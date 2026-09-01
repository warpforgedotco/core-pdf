# SPDX-License-Identifier: AGPL-3.0-only
"""Recover text exported as individual KiCad Newstroke path segments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import Any, cast

import numpy

from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.parse.newstroke_data import NEWSTROKE_ASCII, NEWSTROKE_ASCII_ALTERNATES
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing, CapturedPath

FIT_ERROR = 0.08
FIXED_ERROR = 0.10
CURSOR_ERROR = 0.75
RAW_SPACE_WIDTH = 16.0
MIN_FONT_SIZE = 1.0
MAX_FONT_SIZE = 40.0
MIN_CANDIDATE_SEGMENTS = 10_000
MIN_MATCHED_SEGMENTS = 10_000
MIN_MATCHED_COVERAGE = 0.70
MIN_CHARACTERS = 1_000
MIN_SEQUENCES = 100


@dataclass(frozen=True, slots=True)
class NewstrokeDecode:
    """Deterministic text runs and the evidence used to accept them."""

    runs: tuple[TextRun, ...] = ()
    candidate_segments: int = 0
    matched_segments: int = 0
    glyphs: int = 0
    characters: int = 0
    sequences: int = 0
    maximum_error: float = 0.0

    @property
    def matched_coverage(self) -> float:
        return self.matched_segments / max(1, self.candidate_segments)

    @property
    def trusted(self) -> bool:
        """Require page-level corroboration before replacing OCR with template text."""
        return (
            self.candidate_segments >= MIN_CANDIDATE_SEGMENTS
            and self.matched_segments >= MIN_MATCHED_SEGMENTS
            and self.matched_coverage >= MIN_MATCHED_COVERAGE
            and self.characters >= MIN_CHARACTERS
            and self.sequences >= MIN_SEQUENCES
            and self.maximum_error <= FIXED_ERROR
        )


@dataclass(frozen=True, slots=True)
class internal_Template:
    char: str
    width: float
    segments: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    continuity: tuple[bool, ...]
    solver: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    points: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    centroid_x: float
    centroid_y: float


@dataclass(frozen=True, slots=True)
class internal_TemplateSet:
    all: tuple[internal_Template, ...]
    robust: tuple[internal_Template, ...]
    by_first_delta: dict[tuple[int, int], tuple[internal_Template, ...]]


@dataclass(frozen=True, slots=True)
class internal_Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    style: int
    line_width: float


@dataclass(frozen=True, slots=True)
class internal_Transform:
    matrix: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    inverse: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    scale: float
    x_scale: float
    y_scale: float


@dataclass(frozen=True, slots=True)
class internal_Match:
    char: str
    start: int
    stop: int
    width: float
    transform: internal_Transform
    translation: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    error: float


@cache
def internal_templates() -> internal_TemplateSet:
    templates: list[internal_Template] = []
    by_first_delta: dict[tuple[int, int], list[internal_Template]] = {}
    encoded_glyphs = (
        *((chr(offset + 32), encoded) for offset, encoded in enumerate(NEWSTROKE_ASCII)),
        *NEWSTROKE_ASCII_ALTERNATES.items(),
    )
    for character, encoded in encoded_glyphs:
        start_x = ord(encoded[0]) - ord("R")
        end_x = ord(encoded[1]) - ord("R")
        strokes: list[list[tuple[float, float]]] = []
        stroke: list[tuple[float, float]] = []
        for index in range(2, len(encoded), 2):
            pair = encoded[index : index + 2]
            if pair == " R":
                if stroke:
                    strokes.append(stroke)
                stroke = []
            else:
                stroke.append(
                    (
                        float(ord(pair[0]) - ord("R") - start_x),
                        float(ord(pair[1]) - ord("R") - 8),
                    )
                )
        if stroke:
            strokes.append(stroke)
        raw_segments = [
            (left, right)
            for points in strokes
            for left, right in zip(points, points[1:], strict=False)
        ]
        if not raw_segments:
            continue
        segments = numpy.asarray(raw_segments, dtype=numpy.float64)
        continuity = tuple(
            bool(numpy.array_equal(segments[index, 1], segments[index + 1, 0]))
            for index in range(len(segments) - 1)
        )
        source = segments.reshape((-1, 2))
        design = numpy.column_stack((source, numpy.ones(len(source))))
        template = internal_Template(
            character,
            float(end_x - start_x),
            segments,
            continuity,
            numpy.linalg.pinv(design),
            source,
            float(numpy.mean(source[:, 0])),
            float(numpy.mean(source[:, 1])),
        )
        templates.append(template)
        delta = segments[0, 1] - segments[0, 0]
        by_first_delta.setdefault((int(delta[0]), int(delta[1])), []).append(template)
    all_templates = tuple(templates)
    return internal_TemplateSet(
        all_templates,
        tuple(template for template in all_templates if len(template.segments) >= 5),
        {key: tuple(value) for key, value in by_first_delta.items()},
    )


def internal_color(value: object) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    return tuple(float(cast(Any, component)) for component in value)


def internal_drawing_style(drawing: CapturedDrawing) -> tuple[object, ...] | None:
    style = drawing.stroke_style_key()
    if style is None or style[1] <= 0.0 or style[2] <= 0.0:
        return None
    return style


def internal_segments(
    drawings: tuple[Any, ...],
) -> tuple[tuple[internal_Segment | None, ...], tuple[tuple[object, ...], ...], int]:
    segments: list[internal_Segment | None] = []
    style_ids: dict[tuple[object, ...], int] = {}
    styles: list[tuple[object, ...]] = []
    candidate_count = 0
    for drawing in drawings:
        path = getattr(drawing, "path", None)
        if (
            not isinstance(drawing, CapturedDrawing)
            or drawing.kind != "stroke"
            or type(path) is not CapturedPath
            or len(path.subpaths) != 1
            or path.subpaths[0].closed
            or len(path.subpaths[0].points) != 2
        ):
            segments.append(None)
            continue
        style = internal_drawing_style(drawing)
        if style is None:
            segments.append(None)
            continue
        (x0, y0), (x1, y1) = path.subpaths[0].points
        if abs(x1 - x0) <= 1e-9 and abs(y1 - y0) <= 1e-9:
            segments.append(None)
            continue
        style_id = style_ids.get(style)
        if style_id is None:
            style_id = len(styles)
            style_ids[style] = style_id
            styles.append(style)
        segments.append(
            internal_Segment(
                float(x0),
                float(y0),
                float(x1),
                float(y1),
                style_id,
                cast(float, style[2]),
            )
        )
        candidate_count += 1
    return tuple(segments), tuple(styles), candidate_count


def internal_continuity(segments: tuple[internal_Segment | None, ...]) -> tuple[bool, ...]:
    result: list[bool] = []
    for left, right in zip(segments, segments[1:], strict=False):
        if left is None or right is None or left.style != right.style:
            result.append(False)
            continue
        tolerance = max(0.01, min(left.line_width, right.line_width) * 0.1)
        dx = left.x1 - right.x0
        dy = left.y1 - right.y0
        result.append(dx * dx + dy * dy <= tolerance * tolerance)
    return tuple(result)


def internal_window(
    segments: tuple[internal_Segment | None, ...],
    start: int,
    size: int,
    style: int,
    point_data: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    style_data: numpy.ndarray[Any, numpy.dtype[numpy.int16]],
) -> numpy.ndarray[Any, numpy.dtype[numpy.float64]] | None:
    stop = start + size
    if stop > len(segments) or not numpy.all(style_data[start:stop] == style):
        return None
    return point_data[start:stop]


def internal_fit_match(
    segments: tuple[internal_Segment | None, ...],
    start: int,
    template: internal_Template,
    point_data: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    style_data: numpy.ndarray[Any, numpy.dtype[numpy.int16]],
) -> internal_Match | None:
    first = segments[start]
    if first is None:
        return None
    actual = internal_window(
        segments, start, len(template.segments), first.style, point_data, style_data
    )
    if actual is None:
        return None
    source = template.points
    target = actual.reshape((-1, 2))
    coefficients = template.solver @ target
    matrix = coefficients[:2]
    translation = coefficients[2]
    a = float(matrix[0, 0])
    b = float(matrix[0, 1])
    c = float(matrix[1, 0])
    d = float(matrix[1, 1])
    x_scale = math.hypot(a, b)
    y_scale = math.hypot(c, d)
    scale = max(x_scale, y_scale)
    nominal_size = scale * 21.0
    if not (MIN_FONT_SIZE <= nominal_size <= MAX_FONT_SIZE):
        return None
    min_scale = min(x_scale, y_scale)
    if min_scale <= 0.0 or scale / min_scale > 4.0:
        return None
    orthogonality = abs(a * c + b * d) / (x_scale * y_scale)
    if orthogonality > 0.25:
        return None
    predicted = source @ matrix + translation
    residual = predicted - target
    error = (
        math.sqrt(
            float(numpy.max(residual[:, 0] * residual[:, 0] + residual[:, 1] * residual[:, 1]))
        )
        / scale
    )
    if error > FIT_ERROR:
        return None
    determinant = a * d - b * c
    if determinant == 0.0:
        return None
    inverse = numpy.asarray(((d, -b), (-c, a)), dtype=numpy.float64) / determinant
    transform = internal_Transform(matrix, inverse, scale, x_scale, y_scale)
    return internal_Match(
        template.char,
        start,
        start + len(template.segments),
        template.width,
        transform,
        translation,
        error,
    )


def internal_fixed_template_match(
    segments: tuple[internal_Segment | None, ...],
    continuity: tuple[bool, ...],
    template: internal_Template,
    start: int,
    transform: internal_Transform,
    style: int,
    point_data: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    style_data: numpy.ndarray[Any, numpy.dtype[numpy.int16]],
) -> internal_Match | None:
    size = len(template.segments)
    if continuity[start : start + size - 1] != template.continuity:
        return None
    actual = internal_window(segments, start, size, style, point_data, style_data)
    if actual is None:
        return None
    source = template.points
    target = actual.reshape((-1, 2))
    target_x = float(numpy.mean(target[:, 0]))
    target_y = float(numpy.mean(target[:, 1]))
    matrix = transform.matrix
    translation = numpy.asarray(
        (
            target_x - template.centroid_x * matrix[0, 0] - template.centroid_y * matrix[1, 0],
            target_y - template.centroid_x * matrix[0, 1] - template.centroid_y * matrix[1, 1],
        ),
        dtype=numpy.float64,
    )
    predicted = source @ transform.matrix + translation
    residual = predicted - target
    error = (
        math.sqrt(
            float(numpy.max(residual[:, 0] * residual[:, 0] + residual[:, 1] * residual[:, 1]))
        )
        / transform.scale
    )
    if error > FIXED_ERROR:
        return None
    return internal_Match(
        template.char,
        start,
        start + size,
        template.width,
        transform,
        translation,
        error,
    )


def internal_fixed_match(
    segments: tuple[internal_Segment | None, ...],
    continuity: tuple[bool, ...],
    templates: internal_TemplateSet,
    start: int,
    transform: internal_Transform,
    style: int,
    point_data: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    style_data: numpy.ndarray[Any, numpy.dtype[numpy.int16]],
) -> internal_Match | None:
    first = segments[start]
    if first is None:
        return None
    # Avoid numpy.asarray for 2-element vector: dx*dx_inv + dy*dy_inv
    dx = first.x1 - first.x0
    dy = first.y1 - first.y0
    inverse = transform.inverse
    raw_dx = dx * inverse[0, 0] + dy * inverse[1, 0]
    raw_dy = dx * inverse[0, 1] + dy * inverse[1, 1]
    rounded_dx = round(raw_dx)
    rounded_dy = round(raw_dy)
    if max(abs(raw_dx - rounded_dx), abs(raw_dy - rounded_dy)) > 0.20:
        return None
    candidates: list[internal_Match] = []
    for template in templates.by_first_delta.get((rounded_dx, rounded_dy), ()):
        candidate = internal_fixed_template_match(
            segments,
            continuity,
            template,
            start,
            transform,
            style,
            point_data,
            style_data,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None
    # A one-stroke glyph can be an exact prefix of a richer glyph: I is the
    # first stroke of H and P. Prefer the longest valid template, then error.
    candidates.sort(key=lambda candidate: (-candidate.stop, candidate.error))
    best = candidates[0]
    if (
        len(candidates) > 1
        and candidates[1].stop == best.stop
        and candidates[1].char != best.char
        and candidates[1].error - best.error < 0.01
    ):
        return None
    return best


def internal_cursor_follows(previous: internal_Match, current: internal_Match) -> bool:
    expected = previous.translation + previous.width * previous.transform.matrix[0]
    offset = (current.translation - expected) @ previous.transform.inverse
    y_offset = abs(float(offset[1]))
    x_offset = float(offset[0])
    if y_offset > CURSOR_ERROR or x_offset < -CURSOR_ERROR:
        return False
    if x_offset > RAW_SPACE_WIDTH * 2.0 + CURSOR_ERROR:
        return False
    spaces = round(x_offset / RAW_SPACE_WIDTH)
    return 0 <= spaces <= 2 and abs(x_offset - spaces * RAW_SPACE_WIDTH) <= CURSOR_ERROR


def internal_decode_forward(
    segments: tuple[internal_Segment | None, ...],
    continuity: tuple[bool, ...],
    templates: internal_TemplateSet,
    seed: internal_Match,
    point_data: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    style_data: numpy.ndarray[Any, numpy.dtype[numpy.int16]],
) -> tuple[internal_Match, ...]:
    result = [seed]
    first = segments[seed.start]
    if first is None:
        return ()
    position = seed.stop
    while position < len(segments):
        candidate = internal_fixed_match(
            segments,
            continuity,
            templates,
            position,
            seed.transform,
            first.style,
            point_data,
            style_data,
        )
        if candidate is None or not internal_cursor_follows(result[-1], candidate):
            break
        result.append(candidate)
        position = candidate.stop
    return tuple(result)


def internal_decode_around(
    segments: tuple[internal_Segment | None, ...],
    continuity: tuple[bool, ...],
    templates: internal_TemplateSet,
    seed: internal_Match,
    minimum_start: int,
    point_data: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    style_data: numpy.ndarray[Any, numpy.dtype[numpy.int16]],
) -> tuple[internal_Match, ...]:
    result = list(
        internal_decode_forward(segments, continuity, templates, seed, point_data, style_data)
    )
    first = segments[seed.start]
    if first is None:
        return ()
    position = seed.start
    # Use a separate prepend list to avoid O(n) insert(0) calls.
    prepend: list[internal_Match] = []
    while position > minimum_start:
        candidates: list[internal_Match] = []
        for template in templates.all:
            start = position - len(template.segments)
            if start < minimum_start:
                continue
            candidate = internal_fixed_template_match(
                segments,
                continuity,
                template,
                start,
                seed.transform,
                first.style,
                point_data,
                style_data,
            )
            anchor = prepend[-1] if prepend else result[0]
            if candidate is not None and internal_cursor_follows(candidate, anchor):
                candidates.append(candidate)
        if not candidates:
            break
        candidates.sort(key=lambda candidate: (candidate.start, candidate.error))
        best = candidates[0]
        if (
            len(candidates) > 1
            and candidates[1].start == best.start
            and candidates[1].char != best.char
            and candidates[1].error - best.error < 0.01
        ):
            break
        prepend.append(best)
        position = best.start
    prepend.reverse()
    return tuple(prepend) + tuple(result)


def internal_sequence_text(matches: tuple[internal_Match, ...]) -> str:
    if not matches:
        return ""
    parts = [matches[0].char]
    transform = matches[0].transform
    for previous, current in zip(matches, matches[1:], strict=False):
        expected = previous.translation + previous.width * transform.matrix[0]
        offset = (current.translation - expected) @ transform.inverse
        if float(offset[0]) > RAW_SPACE_WIDTH * 0.5:
            parts.append(" " * max(1, round(float(offset[0]) / RAW_SPACE_WIDTH)))
        parts.append(current.char)
    return "".join(parts)


def internal_sequence_run(
    matches: tuple[internal_Match, ...],
    segments: tuple[internal_Segment | None, ...],
    styles: tuple[tuple[object, ...], ...],
    order: int,
) -> TextRun:
    concrete = tuple(
        segment for segment in segments[matches[0].start : matches[-1].stop] if segment is not None
    )
    # Single pass instead of 5 separate generator passes over `concrete`.
    max_line_width = concrete[0].line_width
    min_x = min(concrete[0].x0, concrete[0].x1)
    max_x = max(concrete[0].x0, concrete[0].x1)
    min_y = min(concrete[0].y0, concrete[0].y1)
    max_y = max(concrete[0].y0, concrete[0].y1)
    for segment in concrete[1:]:
        if segment.line_width > max_line_width:
            max_line_width = segment.line_width
        sx0 = segment.x0
        sx1 = segment.x1
        seg_min_x = sx0 if sx0 < sx1 else sx1
        seg_max_x = sx1 if sx1 > sx0 else sx0
        if seg_min_x < min_x:
            min_x = seg_min_x
        if seg_max_x > max_x:
            max_x = seg_max_x
        sy0 = segment.y0
        sy1 = segment.y1
        seg_min_y = sy0 if sy0 < sy1 else sy1
        seg_max_y = sy1 if sy1 > sy0 else sy0
        if seg_min_y < min_y:
            min_y = seg_min_y
        if seg_max_y > max_y:
            max_y = seg_max_y
    padding = max_line_width * 0.5
    x0 = min_x - padding
    y0 = min_y - padding
    x1 = max_x + padding
    y1 = max_y + padding
    transform = matches[0].transform
    origin = matches[0].translation
    advance = matches[-1].translation + matches[-1].width * transform.matrix[0]
    angle = (
        round(
            math.degrees(math.atan2(float(transform.matrix[0, 1]), float(transform.matrix[0, 0])))
        )
        % 360
    )
    maximum_error = max(match.error for match in matches)
    return TextRun(
        text=internal_sequence_text(matches),
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        tx=float(origin[0]),
        ty=float(origin[1]),
        font_size=transform.scale * 21.0,
        space_width=transform.x_scale * RAW_SPACE_WIDTH,
        order=order,
        stream_order=matches[0].start,
        xobject_depth=0,
        font_name="KiCad Newstroke",
        rotation_angle=angle,
        visible=True,
        seqno=matches[0].start,
        fill_color=cast(tuple[float, ...] | None, styles[concrete[0].style][0]),
        advance_bbox=(x0, y0, x1, y1),
        ink_bbox=(x0, y0, x1, y1),
        baseline=(float(origin[0]), float(origin[1]), float(advance[0]), float(advance[1])),
        provenance=(
            ("unicode_source", "newstroke-template"),
            ("newstroke_max_error", maximum_error),
        ),
        confidence=max(90.0, 100.0 - maximum_error * 100.0),
    )


def decode_newstroke_drawings(drawings: tuple[Any, ...]) -> NewstrokeDecode:
    """Decode a flattened Newstroke page without rasterization or OCR."""
    segments, styles, candidate_count = internal_segments(drawings)
    if candidate_count < MIN_CANDIDATE_SEGMENTS:
        return NewstrokeDecode(candidate_segments=candidate_count)
    templates = internal_templates()
    continuity = internal_continuity(segments)
    point_data = numpy.zeros((len(segments), 2, 2), dtype=numpy.float64)
    style_data = numpy.full(len(segments), -1, dtype=numpy.int16)
    for index, segment in enumerate(segments):
        if segment is not None:
            point_data[index] = ((segment.x0, segment.y0), (segment.x1, segment.y1))
            style_data[index] = segment.style
    sequences: list[tuple[internal_Match, ...]] = []
    known_transforms: dict[int, list[internal_Transform]] = {}
    position = 0
    available_start = 0
    while position < len(segments):
        first = segments[position]
        if first is None:
            position += 1
            continue
        candidates: list[tuple[tuple[int, int, float], tuple[internal_Match, ...]]] = []
        for transform in known_transforms.get(first.style, ()):
            seed = internal_fixed_match(
                segments,
                continuity,
                templates,
                position,
                transform,
                first.style,
                point_data,
                style_data,
            )
            if seed is None:
                continue
            decoded = internal_decode_around(
                segments,
                continuity,
                templates,
                seed,
                available_start,
                point_data,
                style_data,
            )
            span = decoded[-1].stop - decoded[0].start
            candidates.append(
                ((span, len(decoded), -sum(match.error for match in decoded)), decoded)
            )
        if not candidates:
            for template in templates.robust:
                size = len(template.segments)
                if continuity[position : position + size - 1] != template.continuity:
                    continue
                seed = internal_fit_match(segments, position, template, point_data, style_data)
                if seed is None:
                    continue
                decoded = internal_decode_around(
                    segments,
                    continuity,
                    templates,
                    seed,
                    available_start,
                    point_data,
                    style_data,
                )
                span = decoded[-1].stop - decoded[0].start
                candidates.append(
                    ((span, len(decoded), -sum(match.error for match in decoded)), decoded)
                )
        if not candidates:
            position += 1
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        decoded = candidates[0][1]
        span = decoded[-1].stop - decoded[0].start
        if len(decoded) < 2 or span < 10:
            position += 1
            continue
        sequences.append(decoded)
        transforms = known_transforms.setdefault(first.style, [])
        matrix = decoded[0].transform.matrix
        if not any(
            float(numpy.max(numpy.abs(transform.matrix - matrix)))
            <= max(0.001, float(numpy.max(numpy.abs(transform.matrix))) * 0.01)
            for transform in transforms
        ):
            transforms.append(decoded[0].transform)
        position = decoded[-1].stop
        available_start = position

    matched_segments = sum(sequence[-1].stop - sequence[0].start for sequence in sequences)
    glyphs = sum(len(sequence) for sequence in sequences)
    runs = tuple(
        internal_sequence_run(sequence, segments, styles, order)
        for order, sequence in enumerate(sequences)
    )
    characters = sum(not character.isspace() for run in runs for character in run.text)
    maximum_error = max(
        (match.error for sequence in sequences for match in sequence),
        default=0.0,
    )
    return NewstrokeDecode(
        runs=runs,
        candidate_segments=candidate_count,
        matched_segments=matched_segments,
        glyphs=glyphs,
        characters=characters,
        sequences=len(sequences),
        maximum_error=maximum_error,
    )


__all__ = ("NewstrokeDecode", "decode_newstroke_drawings")
