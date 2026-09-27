# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from copy import replace
from types import MappingProxyType

import numpy

from core_pdf.impl.array_views import finite_median
from core_pdf.impl.extract_block_order import (
    REORDER_PASSES,
    full_width_blocks,
    line_bbox,
    topological_block_order,
)
from core_pdf.impl.extract_contracts import (
    ObservationBatch,
    ObservationSource,
    PageFrame,
    ParsedBlock,
    ParsedLine,
    TextReference,
)
from core_pdf.impl.extract_layout_rules import LAYOUT_RULES
from core_pdf.impl.extract_xy_cut import row_order_indexes, xy_cut_regions
from core_pdf.impl.geometry import bbox_tuple, interval_overlap
from core_pdf.impl.layout_lines import LayoutLine
from core_pdf.impl.output_model import TextLine, TextSpan
from core_pdf.impl.runs import TextRun
from core_pdf.impl.text import (
    TextWord,
    collapse_ws,
    reconcile_text_words,
    text_word_tokens,
)
from core_pdf.impl.types import GeneratedRecord, frozen_setattr

NATIVE_SOURCE = int(ObservationSource.NATIVE)

CAPTION_RE = re.compile(r"^(?:figure|fig\.|table|chart|exhibit)\s+\d+\b")
LIST_MARKER_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+")

OBSERVATION_JOIN_NO_SPACE_AFTER = (" ", "-", "/")
OBSERVATION_JOIN_NO_SPACE_BEFORE = (".", ",", ":", ";", ")", "]", "}")
SPAN_JOIN_NO_SPACE_AFTER = ("(", "[", "{", "/", "-")
SPAN_JOIN_NO_SPACE_BEFORE = (".", ",", ";", ":", "!", "?", ")", "]", "}")
SPAN_TRAILING_SPACE = (" ", "\t", "\n")


GroupOrder = Callable[[ObservationBatch, numpy.ndarray], numpy.ndarray]


class LayoutHooks(GeneratedRecord):
    source_labels: Mapping[int, str]
    group_order: GroupOrder | None

    def __init__(
        self,
        source_labels: Mapping[int, str] | None = None,
        group_order: GroupOrder | None = None,
    ) -> None:
        frozen_setattr(
            self,
            "source_labels",
            MappingProxyType({NATIVE_SOURCE: "native"} if source_labels is None else source_labels),
        )
        frozen_setattr(self, "group_order", group_order)

    def __hash__(self) -> int:
        return hash((tuple(self.source_labels.items()), self.group_order))


NATIVE_LAYOUT_HOOKS = LayoutHooks()


class LineGroupPlan(GeneratedRecord):
    indexes: numpy.ndarray
    starts: numpy.ndarray
    stops: numpy.ndarray


class BuiltLines(GeneratedRecord):
    lines: tuple[ParsedLine, ...]
    boxes: numpy.ndarray


def line_group_indexes(observations: ObservationBatch) -> LineGroupPlan:
    if not len(observations):
        empty = numpy.empty(0, dtype=numpy.int64)
        return LineGroupPlan(empty, empty, empty)
    visible_indexes = numpy.flatnonzero(observations.visible)
    indexes = (
        visible_indexes
        if len(visible_indexes)
        else numpy.arange(len(observations), dtype=numpy.int64)
    )
    boxes = observations.bbox[indexes]
    rotations = observations.rotation[indexes]
    vertical = numpy.mod(rotations, 180) != 0
    widths = numpy.maximum(1.0, boxes[:, 2] - boxes[:, 0])
    heights = numpy.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    spans = numpy.where(vertical, widths, heights)
    centers = numpy.where(
        vertical,
        (boxes[:, 0] + boxes[:, 2]) * 0.5,
        (boxes[:, 1] + boxes[:, 3]) * 0.5,
    )
    explicit = observations.line_break_before[indexes]
    breaks = numpy.zeros(len(indexes), dtype=numpy.bool_)
    breaks[0] = True
    if len(indexes) > 1:
        tolerance = numpy.maximum(2.0, numpy.minimum(spans[:-1], spans[1:]) * 0.65)
        breaks[1:] = (
            explicit[1:]
            | (rotations[1:] != rotations[:-1])
            | (numpy.abs(centers[1:] - centers[:-1]) > tolerance)
        )
    starts = numpy.flatnonzero(breaks).astype(numpy.int64, copy=False)
    stops = numpy.empty_like(starts)
    stops[:-1] = starts[1:]
    stops[-1] = len(indexes)
    return LineGroupPlan(indexes, starts, stops)


def style_enabled(reference: object, name: str) -> bool:
    value = getattr(reference, name, False)
    return bool(value() if callable(value) else value)


def group_text_and_words(
    observations: ObservationBatch,
    indexes: numpy.ndarray,
) -> tuple[str, tuple[TextWord, ...]]:
    references = tuple(observations.references[index] for index in indexes)
    if references and all(isinstance(reference, TextRun) for reference in references):
        runs = list(references)
        line = LayoutLine(runs)
        reconstructed = line.reconstructed_text()
        text = reconstructed.text.strip()
        layout_words = line.text_and_words(reconstructed)[1]
        return text, reconcile_text_words(text, layout_words)
    parts: list[str] = []
    candidate_words: list[TextWord] = []
    for index in indexes:
        text = observations.text[index].strip()
        if not text:
            continue
        if (
            parts
            and not parts[-1].endswith(OBSERVATION_JOIN_NO_SPACE_AFTER)
            and not text.startswith(OBSERVATION_JOIN_NO_SPACE_BEFORE)
        ):
            parts.append(" ")
        parts.append(text)
        tokens = text_word_tokens(text)
        bbox = observations.bbox[index]
        word_bbox = bbox_tuple(bbox) if len(tokens) == 1 else None
        candidate_words.extend(TextWord(token, bbox=word_bbox) for token in tokens)
    combined = "".join(parts)
    return combined, reconcile_text_words(combined, tuple(candidate_words))


def color_is_emphasis(color: object) -> bool:
    if not isinstance(color, (tuple, list)) or len(color) < 3:
        return False
    components: list[float] = []
    for component in color[:3]:
        if not isinstance(component, (int, float)):
            return False
        components.append(float(component))
    return max(components) - min(components) >= 0.15


def build_lines(
    observations: ObservationBatch,
    hooks: LayoutHooks | None = None,
) -> BuiltLines:
    hooks = NATIVE_LAYOUT_HOOKS if hooks is None else hooks
    labels = hooks.source_labels
    group_order = hooks.group_order
    line_groups = line_group_indexes(observations)
    if not len(line_groups.starts):
        return BuiltLines((), numpy.empty((0, 4), dtype=numpy.float32))
    selected = line_groups.indexes
    starts = line_groups.starts
    selected_boxes = observations.bbox[selected]
    group_boxes = numpy.column_stack(
        (
            numpy.minimum.reduceat(selected_boxes[:, 0], starts),
            numpy.minimum.reduceat(selected_boxes[:, 1], starts),
            numpy.maximum.reduceat(selected_boxes[:, 2], starts),
            numpy.maximum.reduceat(selected_boxes[:, 3], starts),
        )
    ).astype(numpy.float32, copy=False)
    selected_sources = observations.source[selected]
    source_minimum = numpy.minimum.reduceat(selected_sources, starts)
    source_maximum = numpy.maximum.reduceat(selected_sources, starts)
    group_sequences = numpy.minimum.reduceat(observations.sequence[selected], starts)
    run_styles: dict[str | None, tuple[bool, bool]] = {}
    color_marks: dict[object, bool] = {}

    def reference_style(reference: TextReference) -> tuple[bool, bool]:
        if type(reference) is not TextRun:
            return (style_enabled(reference, "is_bold"), style_enabled(reference, "is_italic"))
        font_name = reference.font_name
        style = run_styles.get(font_name)
        if style is None:
            style = run_styles[font_name] = (reference.is_bold(), reference.is_italic())
        return style

    def emphasis_mark(color: object) -> bool:
        if type(color) is not tuple:
            return color_is_emphasis(color)
        mark = color_marks.get(color)
        if mark is None:
            mark = color_marks[color] = color_is_emphasis(color)
        return mark

    output: list[ParsedLine] = []
    output_boxes: list[numpy.ndarray] = []
    for group_index, (start, stop) in enumerate(
        zip(line_groups.starts, line_groups.stops, strict=True)
    ):
        indexes = selected[int(start) : int(stop)]
        all_native = source_minimum[group_index] == source_maximum[group_index] == NATIVE_SOURCE
        text_indexes = (
            group_order(observations, indexes)
            if group_order is not None and not all_native
            else indexes
        )
        text, words = group_text_and_words(observations, text_indexes)
        if not text:
            continue
        confidences = observations.confidence[indexes]
        font_sizes = observations.font_size[indexes]
        finite_confidences = confidences[numpy.isfinite(confidences)]
        finite_font_sizes = font_sizes[numpy.isfinite(font_sizes) & (font_sizes > 0)]
        native_references = [
            reference
            for reference in (observations.references[index] for index in indexes)
            if reference is not None
        ]
        reference_styles = [reference_style(reference) for reference in native_references]
        bold = bool(native_references) and sum(style[0] for style in reference_styles) * 2 >= len(
            native_references
        )
        italic = bool(native_references) and sum(style[1] for style in reference_styles) * 2 >= len(
            native_references
        )
        span_values: list[TextSpan] = []
        pending_space = False
        for reference, (reference_bold, reference_italic) in zip(
            native_references, reference_styles, strict=True
        ):
            reference_text = reference.text.strip()
            if not reference_text:
                pending_space = True
                continue
            prefix = ""
            if (
                pending_space
                and span_values
                and not span_values[-1].text.endswith(SPAN_JOIN_NO_SPACE_AFTER)
                and not reference_text.startswith(SPAN_JOIN_NO_SPACE_BEFORE)
            ):
                prefix = " "
            span_values.append(
                TextSpan(
                    text=prefix + reference_text,
                    bold=reference_bold,
                    italic=reference_italic,
                    mark=emphasis_mark(getattr(reference, "fill_color", None)),
                )
            )
            pending_space = reference.text.endswith(SPAN_TRAILING_SPACE)
        source_low = int(source_minimum[group_index])
        source_high = int(source_maximum[group_index])
        source = labels.get(source_low, "hybrid") if source_low == source_high else "hybrid"
        words = tuple(
            type(word)(
                word.text,
                word.bbox,
                word.line_index,
                word.word_index,
                word.block_index,
                word.page_number,
                source,
            )
            for word in words
        )
        group_box = group_boxes[group_index]
        output.append(
            ParsedLine(
                TextLine(
                    text,
                    bbox=bbox_tuple(group_box),
                    source=source,
                    confidence=(
                        float(numpy.mean(finite_confidences)) if len(finite_confidences) else None
                    ),
                    bold=bold,
                    italic=italic,
                    spans=tuple(span_values),
                    words=words,
                ),
                sequence=int(group_sequences[group_index]),
                rotation=int(observations.rotation[indexes[0]]),
                font_size=(finite_median(finite_font_sizes) if len(finite_font_sizes) else None),
            )
        )
        output_boxes.append(group_box)
    boxes = (
        numpy.asarray(output_boxes, dtype=numpy.float32).reshape((-1, 4))
        if output_boxes
        else numpy.empty((0, 4), dtype=numpy.float32)
    )
    return BuiltLines(tuple(output), boxes)


def assign_columns(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    if len(blocks) < 2:
        return blocks
    bands: list[list[float]] = []
    assignments: list[int | None] = []
    column_band_overlap = LAYOUT_RULES.reading_order.column_band_overlap
    for block, spans_page in zip(blocks, full_width_blocks(blocks), strict=True):
        if spans_page:
            assignments.append(None)
            continue
        x0, _y0, x1, _y1 = block.bbox
        width = x1 - x0
        best_band: int | None = None
        best_overlap = 0.0
        for band_index, (band_x0, band_x1) in enumerate(bands):
            overlap = interval_overlap(x0, x1, band_x0, band_x1)
            overlap_ratio = overlap / max(1.0, min(width, band_x1 - band_x0))
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_band = band_index
        if best_band is None or best_overlap < column_band_overlap:
            bands.append([x0, x1])
            assignments.append(len(bands) - 1)
        else:
            bands[best_band][0] = min(bands[best_band][0], x0)
            bands[best_band][1] = max(bands[best_band][1], x1)
            assignments.append(best_band)
    ranked_bands = {
        old_index: new_index
        for new_index, old_index in enumerate(
            sorted(range(len(bands)), key=lambda index: bands[index][0])
        )
    }
    return [
        ParsedBlock(
            lines=block.lines,
            bbox=block.bbox,
            column_index=(ranked_bands[assignment] if assignment is not None else None),
            kind=block.kind,
        )
        for block, assignment in zip(blocks, assignments, strict=True)
    ]


def classify_blocks(
    blocks: list[ParsedBlock],
    *,
    body_font_size: float | None,
) -> list[ParsedBlock]:
    rules = LAYOUT_RULES.classification
    classified: list[ParsedBlock] = []
    heading_sizes = sorted(
        {
            line.font_size
            for block in blocks
            for line in block.lines
            if line.font_size is not None and line.font_size > 0
        },
        reverse=True,
    )
    heading_rank = {size: rank for rank, size in enumerate(heading_sizes, start=1)}
    for block in blocks:
        text = " ".join(line.line.text for line in block.lines)
        normalized = collapse_ws(text)
        kind = "paragraph"
        level: int | None = None
        lowered = normalized.casefold()
        if CAPTION_RE.match(lowered):
            kind = "caption"
        elif block.lines and all(
            LIST_MARKER_RE.match(line.line.text.strip()) for line in block.lines
        ):
            kind = "list"
        elif (
            body_font_size is not None
            and len(block.lines) <= rules.heading_max_lines
            and len(normalized) <= rules.heading_max_characters
            and (size := max((line.font_size or 0.0) for line in block.lines))
            >= body_font_size * rules.heading_size_ratio
        ):
            kind = "heading"
            level = min(rules.heading_max_level, heading_rank.get(size, 1))
        classified.append(replace(block, kind=kind, level=level))
    return classified


def semantic_body_font_size(lines: tuple[ParsedLine, ...]) -> float | None:
    sizes = numpy.asarray(
        [line.font_size for line in lines if line.font_size is not None and line.font_size > 0],
        dtype=numpy.float32,
    )
    return finite_median(sizes) if len(sizes) else None


def layout_blocks_with_evidence(
    observations: ObservationBatch,
    *,
    frame: PageFrame,
    obstacles: tuple[tuple[float, float, float, float], ...] = (),
    use_xy_cut: bool = True,
    hooks: LayoutHooks | None = None,
) -> tuple[tuple[ParsedBlock, ...], bool]:
    built_lines = build_lines(observations, hooks)
    lines = built_lines.lines
    if not lines:
        return (), False
    boxes = frame.display_boxes(built_lines.boxes)
    if use_xy_cut:
        if obstacles:
            obstacles = tuple(
                bbox_tuple(box)
                for box in frame.display_boxes(numpy.asarray(obstacles, dtype=numpy.float32))
            )
        blocks = xy_cut_blocks(built_lines, boxes, obstacles)
    else:
        indexes = row_order_indexes(
            numpy.arange(len(lines), dtype=numpy.int64),
            boxes,
        )
        blocks = [
            ParsedBlock(lines=(lines[int(index)],), bbox=line_bbox(lines[int(index)]))
            for index in indexes
        ]
    classified = tuple(
        classify_blocks(assign_columns(blocks), body_font_size=semantic_body_font_size(lines))
    )
    return classified, has_mixed_rotation_block(classified)


def xy_cut_blocks(
    built_lines: BuiltLines,
    boxes: numpy.ndarray,
    obstacles: tuple[tuple[float, float, float, float], ...],
) -> list[ParsedBlock]:
    lines = built_lines.lines
    heights = numpy.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    median_height = max(1.0, finite_median(heights))
    regions = xy_cut_regions(
        numpy.arange(len(lines), dtype=numpy.int64),
        boxes,
        obstacles,
        median_height,
    )
    blocks = [
        ParsedBlock(
            lines=tuple(lines[int(index)] for index in region),
            bbox=(
                float(numpy.min(built_lines.boxes[region, 0])),
                float(numpy.min(built_lines.boxes[region, 1])),
                float(numpy.max(built_lines.boxes[region, 2])),
                float(numpy.max(built_lines.boxes[region, 3])),
            ),
        )
        for region in regions
    ]
    for reorder in REORDER_PASSES:
        blocks = reorder(blocks)
    return topological_block_order(blocks)


def layout_element_order(
    boxes: tuple[tuple[float, float, float, float], ...],
    frame: PageFrame,
) -> tuple[int, ...]:
    if len(boxes) < 2:
        return tuple(range(len(boxes)))
    values = frame.display_boxes(numpy.asarray(boxes, dtype=numpy.float32))
    heights = numpy.maximum(1.0, values[:, 3] - values[:, 1])
    span = max(1.0, float(values[:, 2].max() - values[:, 0].min()))
    full_width_ratio = LAYOUT_RULES.reading_order.full_width_ratio
    obstacles = tuple(
        bbox_tuple(box) for box in values if (box[2] - box[0]) / span >= full_width_ratio
    )
    regions = xy_cut_regions(
        numpy.arange(len(boxes), dtype=numpy.int64),
        values,
        obstacles,
        max(1.0, finite_median(heights)),
    )
    return tuple(int(index) for region in regions for index in region)


def has_mixed_rotation_block(blocks: tuple[ParsedBlock, ...]) -> bool:
    return any(len({line.rotation % 360 for line in block.lines}) > 1 for block in blocks)
