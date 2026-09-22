# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from copy import replace
from heapq import heappop, heappush
from typing import ClassVar, cast

import numpy

from core_pdf.impl.extract.contracts import (
    ObservationBatch,
    ObservationSource,
    ParsedBlock,
    ParsedLine,
    ReadingOrderEvidence,
    bbox_tuple,
)
from core_pdf.impl.layout.lines import LayoutLine
from core_pdf.impl.model.geometry import horizontal_overlap_ratio, interval_overlap
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.model.text import (
    collapse_ws,
    reconcile_text_words,
    text_word_tokens,
)
from core_pdf.impl.output.model import TextLine, TextSpan
from core_pdf.impl.records import Record, ReplaceFields, ReprFields, frozen_setattr
from core_pdf.impl.runtime.array_views import finite_median
from core_pdf.impl.types import TextWord

NATIVE_SOURCE = int(ObservationSource.NATIVE)

CAPTION_RE = re.compile(r"^(?:figure|fig\.|table|chart|exhibit)\s+\d+\b")
LIST_MARKER_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+")


class LineGroupPlan(Record):
    __slots__ = ("indexes", "starts", "stops")

    indexes: numpy.ndarray
    starts: numpy.ndarray
    stops: numpy.ndarray

    __fields__: ClassVar[tuple[str, ...]] = ("indexes", "starts", "stops")
    __match_args__ = ("indexes", "starts", "stops")

    def __init__(self, indexes: numpy.ndarray, starts: numpy.ndarray, stops: numpy.ndarray) -> None:
        frozen_setattr(self, "indexes", indexes)
        frozen_setattr(self, "starts", starts)
        frozen_setattr(self, "stops", stops)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.indexes == other.indexes
            and self.starts == other.starts
            and self.stops == other.stops
        )

    def __hash__(self) -> int:
        return hash((self.indexes, self.starts, self.stops))


class BuiltLines(Record):
    __slots__ = ("lines", "boxes")

    lines: tuple[ParsedLine, ...]
    boxes: numpy.ndarray

    __fields__: ClassVar[tuple[str, ...]] = ("lines", "boxes")
    __match_args__ = ("lines", "boxes")

    def __init__(self, lines: tuple[ParsedLine, ...], boxes: numpy.ndarray) -> None:
        frozen_setattr(self, "lines", lines)
        frozen_setattr(self, "boxes", boxes)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.lines == other.lines and self.boxes == other.boxes

    def __hash__(self) -> int:
        return hash((self.lines, self.boxes))


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
        runs = cast(list[TextRun], list(references))
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
            and not parts[-1].endswith((" ", "-", "/"))
            and not text.startswith((".", ",", ":", ";", ")", "]", "}"))
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
    *,
    source_labels: Mapping[int, str] | None = None,
    group_order: Callable[[ObservationBatch, numpy.ndarray], numpy.ndarray] | None = None,
) -> BuiltLines:
    labels = source_labels if source_labels is not None else {NATIVE_SOURCE: "native"}
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
        reference_styles = [
            (
                style_enabled(reference, "is_bold"),
                style_enabled(reference, "is_italic"),
            )
            for reference in native_references
        ]
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
                and not span_values[-1].text.endswith(("(", "[", "{", "/", "-"))
                and not reference_text.startswith((".", ",", ";", ":", "!", "?", ")", "]", "}"))
            ):
                prefix = " "
            span_values.append(
                TextSpan(
                    text=prefix + reference_text,
                    bold=reference_bold,
                    italic=reference_italic,
                    mark=color_is_emphasis(getattr(reference, "fill_color", None)),
                )
            )
            pending_space = reference.text.endswith((" ", "\t", "\n"))
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
    page_x0 = min(block.bbox[0] for block in blocks)
    page_x1 = max(block.bbox[2] for block in blocks)
    page_width = max(1.0, page_x1 - page_x0)
    bands: list[list[float]] = []
    assignments: list[int | None] = []
    for block in blocks:
        x0, y0, x1, y1 = block.bbox
        width = x1 - x0
        if width / page_width >= 0.70:
            assignments.append(None)
            continue
        best_band: int | None = None
        best_overlap = 0.0
        for band_index, (band_x0, band_x1) in enumerate(bands):
            overlap = interval_overlap(x0, x1, band_x0, band_x1)
            overlap_ratio = overlap / max(1.0, min(width, band_x1 - band_x0))
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_band = band_index
        if best_band is None or best_overlap < 0.50:
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
            and len(block.lines) <= 3
            and len(normalized) <= 240
            and (size := max((line.font_size or 0.0) for line in block.lines))
            >= body_font_size * 1.2
        ):
            kind = "heading"
            level = min(3, heading_rank.get(size, 1))
        classified.append(replace(block, kind=kind, level=level))
    return classified


def semantic_body_font_size(lines: tuple[ParsedLine, ...]) -> float | None:
    sizes = numpy.asarray(
        [line.font_size for line in lines if line.font_size is not None and line.font_size > 0],
        dtype=numpy.float32,
    )
    return finite_median(sizes) if len(sizes) else None


def display_boxes(
    boxes: numpy.ndarray, rotation: int, width: float, height: float
) -> numpy.ndarray:
    rotation %= 360
    if rotation == 0 or not len(boxes):
        return boxes
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if rotation == 90:
        corners = (y0, width - x1, y1, width - x0)
    elif rotation == 180:
        corners = (width - x1, height - y1, width - x0, height - y0)
    elif rotation == 270:
        corners = (height - y1, x0, height - y0, x1)
    else:
        return boxes
    rotated = numpy.column_stack(corners).astype(boxes.dtype, copy=False)
    return numpy.column_stack(
        (
            numpy.minimum(rotated[:, 0], rotated[:, 2]),
            numpy.minimum(rotated[:, 1], rotated[:, 3]),
            numpy.maximum(rotated[:, 0], rotated[:, 2]),
            numpy.maximum(rotated[:, 1], rotated[:, 3]),
        )
    ).astype(boxes.dtype, copy=False)


def layout_blocks(
    observations: ObservationBatch,
    *,
    obstacles: tuple[tuple[float, float, float, float], ...] = (),
    use_xy_cut: bool = True,
    rotation: int = 0,
    page_width: float = 0.0,
    page_height: float = 0.0,
    source_labels: Mapping[int, str] | None = None,
    group_order: Callable[[ObservationBatch, numpy.ndarray], numpy.ndarray] | None = None,
) -> tuple[ParsedBlock, ...]:
    built_lines = build_lines(observations, source_labels=source_labels, group_order=group_order)
    lines = built_lines.lines
    if not lines:
        return ()
    boxes = display_boxes(
        built_lines.boxes,
        rotation,
        page_width,
        page_height,
    )
    if obstacles:
        obstacles = tuple(
            bbox_tuple(box)
            for box in display_boxes(
                numpy.asarray(obstacles, dtype=numpy.float32),
                rotation,
                page_width,
                page_height,
            )
        )
    if not use_xy_cut:
        indexes = row_order_indexes(
            numpy.arange(len(lines), dtype=numpy.int64),
            boxes,
        )
        blocks = [
            ParsedBlock(lines=(lines[int(index)],), bbox=line_bbox(lines[int(index)]))
            for index in indexes
        ]
        return tuple(
            classify_blocks(
                assign_columns(blocks),
                body_font_size=semantic_body_font_size(lines),
            )
        )
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
    blocks = interleave_columnar_blocks(blocks)
    blocks = transpose_numeric_table_blocks(blocks)
    blocks = column_major_prose(blocks)
    blocks = topological_block_order(blocks)
    return tuple(
        classify_blocks(assign_columns(blocks), body_font_size=semantic_body_font_size(lines))
    )


def layout_blocks_with_evidence(
    observations: ObservationBatch,
    *,
    obstacles: tuple[tuple[float, float, float, float], ...] = (),
    use_xy_cut: bool = True,
    rotation: int = 0,
    page_width: float = 0.0,
    page_height: float = 0.0,
    source_labels: Mapping[int, str] | None = None,
    group_order: Callable[[ObservationBatch, numpy.ndarray], numpy.ndarray] | None = None,
) -> tuple[tuple[ParsedBlock, ...], ReadingOrderEvidence]:
    blocks = layout_blocks(
        observations,
        obstacles=obstacles,
        use_xy_cut=use_xy_cut,
        rotation=rotation,
        page_width=page_width,
        page_height=page_height,
        source_labels=source_labels,
        group_order=group_order,
    )
    return blocks, reading_order_evidence(blocks)


def layout_element_order(
    boxes: tuple[tuple[float, float, float, float], ...],
    rotation: int = 0,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> tuple[int, ...]:
    if len(boxes) < 2:
        return tuple(range(len(boxes)))
    values = display_boxes(
        numpy.asarray(boxes, dtype=numpy.float32), rotation, page_width, page_height
    )
    heights = numpy.maximum(1.0, values[:, 3] - values[:, 1])
    span = max(1.0, float(values[:, 2].max() - values[:, 0].min()))
    obstacles = tuple(bbox_tuple(box) for box in values if (box[2] - box[0]) / span >= 0.70)
    regions = xy_cut_regions(
        numpy.arange(len(boxes), dtype=numpy.int64),
        values,
        obstacles,
        max(1.0, finite_median(heights)),
    )
    return tuple(int(index) for region in regions for index in region)


def line_bbox(line: ParsedLine) -> tuple[float, float, float, float]:
    bbox = line.line.bbox
    assert bbox is not None
    return bbox


def block_bbox(lines: tuple[ParsedLine, ...]) -> tuple[float, float, float, float]:
    boxes = numpy.asarray(tuple(line_bbox(line) for line in lines), dtype=numpy.float32)
    return (
        float(numpy.min(boxes[:, 0])),
        float(numpy.min(boxes[:, 1])),
        float(numpy.max(boxes[:, 2])),
        float(numpy.max(boxes[:, 3])),
    )


def inversion_count(values: tuple[int, ...]) -> int:
    if len(values) < 2:
        return 0
    ranks = {value: rank + 1 for rank, value in enumerate(sorted(values))}
    tree = [0] * (len(values) + 1)
    inversions = 0
    for seen, value in enumerate(values):
        rank = ranks[value]
        prefix = 0
        index = rank
        while index:
            prefix += tree[index]
            index -= index & -index
        inversions += seen - prefix
        index = rank
        while index < len(tree):
            tree[index] += 1
            index += index & -index
    return inversions


def reading_order_evidence(
    blocks: tuple[ParsedBlock, ...],
) -> ReadingOrderEvidence:
    lines = tuple(line for block in blocks for line in block.lines)
    sequences = tuple(line.sequence for line in lines)
    inversions = inversion_count(sequences)
    maximum = len(lines) * (len(lines) - 1) // 2
    rotations = {line.rotation % 360 for line in lines}
    mixed_rotation_block = any(
        len({line.rotation % 360 for line in block.lines}) > 1 for block in blocks
    )
    columns = {block.column_index for block in blocks if block.column_index is not None}
    repaired = inversions > 0
    ambiguous = mixed_rotation_block
    confidence = 0.5 if ambiguous else (0.85 if len(rotations) > 1 else 1.0)
    return ReadingOrderEvidence(
        line_count=len(lines),
        source_inversions=inversions,
        source_inversion_ratio=inversions / maximum if maximum else 0.0,
        column_count=max(1, len(columns)) if lines else 0,
        rotation_count=len(rotations),
        repaired=repaired,
        ambiguous=ambiguous,
        confidence=confidence,
        strategy="geometric-repair" if repaired else "source-stable",
    )


def interval_overlap_pairs(starts: numpy.ndarray, ends: numpy.ndarray) -> set[tuple[int, int]]:
    order = numpy.argsort(starts, kind="stable")
    active: set[int] = set()
    ending: list[tuple[float, int]] = []
    pairs: set[tuple[int, int]] = set()
    for raw_index in order:
        index = int(raw_index)
        start = float(starts[index])
        while ending and ending[0][0] <= start:
            _end, expired = heappop(ending)
            active.discard(expired)
        for other in active:
            pairs.add((other, index) if other < index else (index, other))
        active.add(index)
        heappush(ending, (float(ends[index]), index))
    return pairs


def sparse_block_candidate_pairs(
    blocks: list[ParsedBlock], full_width: list[bool]
) -> list[tuple[int, int]]:
    boxes = numpy.asarray(tuple(block.bbox for block in blocks), dtype=numpy.float64)
    pairs = interval_overlap_pairs(boxes[:, 0], boxes[:, 2])
    pairs.update(interval_overlap_pairs(boxes[:, 1], boxes[:, 3]))
    full_width_indexes = [index for index, value in enumerate(full_width) if value]
    for index in full_width_indexes:
        pairs.update(
            (min(index, other), max(index, other)) for other in range(len(blocks)) if other != index
        )
    return sorted(pairs)


def full_width_blocks(blocks: list[ParsedBlock]) -> list[bool]:
    page_x0 = min(block.bbox[0] for block in blocks)
    page_x1 = max(block.bbox[2] for block in blocks)
    page_width = max(1.0, page_x1 - page_x0)
    return [(block.bbox[2] - block.bbox[0]) / page_width >= 0.70 for block in blocks]


def topological_block_order_from_pairs(
    blocks: list[ParsedBlock], pairs: Iterable[tuple[int, int]], full_width: list[bool]
) -> list[ParsedBlock]:
    if len(blocks) <= 2:
        return blocks
    n = len(blocks)
    in_degree = [0] * n
    graph: dict[int, list[int]] = defaultdict(list)

    for i, j in pairs:
        ax0, ay0, ax1, ay1 = blocks[i].bbox
        a_is_full_width = full_width[i]
        bx0, by0, bx1, by1 = blocks[j].bbox
        b_is_full_width = full_width[j]

        if a_is_full_width and not b_is_full_width and ay0 >= by1 - 2.0:
            graph[i].append(j)
            in_degree[j] += 1
        elif b_is_full_width and not a_is_full_width and by0 >= ay1 - 2.0:
            graph[j].append(i)
            in_degree[i] += 1
        elif not a_is_full_width and not b_is_full_width:
            if horizontal_overlap_ratio(blocks[i].bbox, blocks[j].bbox) >= 0.45:
                if ay0 >= by1 - 2.0:
                    graph[i].append(j)
                    in_degree[j] += 1
                elif by0 >= ay1 - 2.0:
                    graph[j].append(i)
                    in_degree[i] += 1
            elif ax1 <= bx0 + 2.0 and interval_overlap(ay0, ay1, by0, by1) > 4.0:
                graph[i].append(j)
                in_degree[j] += 1
            elif bx1 <= ax0 + 2.0 and interval_overlap(ay0, ay1, by0, by1) > 4.0:
                graph[j].append(i)
                in_degree[i] += 1

    ready: list[tuple[float, float, int, int]] = []
    serial = 0
    for index in range(n):
        if in_degree[index] == 0:
            heappush(ready, (-blocks[index].bbox[3], blocks[index].bbox[0], serial, index))
            serial += 1
    result: list[int] = []

    while ready:
        _negative_top, _left, _serial, curr = heappop(ready)
        result.append(curr)
        for nxt in graph[curr]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                heappush(ready, (-blocks[nxt].bbox[3], blocks[nxt].bbox[0], serial, nxt))
                serial += 1

    if len(result) == n:
        return [blocks[i] for i in result]
    return blocks


def topological_block_order(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    if len(blocks) <= 2:
        return blocks
    full_width = full_width_blocks(blocks)
    pairs = sparse_block_candidate_pairs(blocks, full_width)
    return topological_block_order_from_pairs(blocks, pairs, full_width)


def interleave_columnar_blocks(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    candidates = [block for block in blocks if len(block.lines) >= 20]
    if len(candidates) < 3:
        return blocks
    x0s = [block.bbox[0] for block in candidates]
    x1s = [block.bbox[2] for block in candidates]
    if max(x0s) - min(x0s) > 20.0 or max(x1s) - min(x1s) > 30.0:
        return blocks
    merged_lines = tuple(line for block in candidates for line in block.lines)
    boxes = numpy.asarray(tuple(line_bbox(line) for line in merged_lines), dtype=numpy.float32)
    ordered = row_order_indexes(numpy.arange(len(merged_lines)), boxes)
    merged = ParsedBlock(
        lines=tuple(merged_lines[int(index)] for index in ordered),
        bbox=block_bbox(merged_lines),
    )
    candidate_ids = {id(block) for block in candidates}
    return [merged, *(block for block in blocks if id(block) not in candidate_ids)]


def column_major_prose(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    output: list[ParsedBlock] = []
    for block in blocks:
        if len(block.lines) < 80:
            output.append(block)
            continue
        alphabetic = 0
        total = 0
        for line in block.lines:
            for character in line.line.text:
                is_alpha = character.isalpha()
                alphabetic += is_alpha
                total += is_alpha or character.isdigit()
        line_starts = numpy.fromiter(
            (line_bbox(line)[0] for line in block.lines), dtype=numpy.float64
        )
        starts = numpy.sort(line_starts)
        clusters: list[float] = []
        for start in starts:
            if not clusters or start - clusters[-1] > 40.0:
                clusters.append(float(start))
        if len(clusters) < 3 or alphabetic / max(1, total) < 0.45:
            output.append(block)
            continue
        cluster_values = numpy.asarray(clusters, dtype=numpy.float64)
        insertion = numpy.searchsorted(cluster_values, line_starts)
        left = numpy.maximum(0, insertion - 1)
        right = numpy.minimum(len(cluster_values) - 1, insertion)
        choose_right = numpy.abs(line_starts - cluster_values[right]) < numpy.abs(
            line_starts - cluster_values[left]
        )
        line_clusters = numpy.where(choose_right, right, left)
        transitions = sum(
            left != right for left, right in zip(line_clusters, line_clusters[1:], strict=False)
        )
        if transitions / max(1, len(line_clusters) - 1) < 0.25:
            output.append(block)
            continue
        columns: list[list[ParsedLine]] = [[] for cluster in clusters]
        for line, assigned in zip(block.lines, line_clusters, strict=True):
            columns[int(assigned)].append(line)
        ordered = tuple(
            line
            for column in columns
            for line in sorted(column, key=lambda item: -line_bbox(item)[1])
        )
        output.append(replace(block, lines=ordered))
    return output


def transpose_numeric_table_blocks(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    output: list[ParsedBlock] = []
    for block in blocks:
        if len(block.lines) < 300:
            output.append(block)
            continue
        text = " ".join(line.line.text for line in block.lines)
        numeric = sum(character.isdigit() for character in text)
        alphanumeric = sum(character.isalnum() for character in text)
        starts = sorted(line_bbox(line)[0] for line in block.lines)
        columns: list[float] = []
        for start in starts:
            if not columns or start - columns[-1] > 8.0:
                columns.append(start)
        if numeric / max(1, alphanumeric) < 0.25 or len(columns) < 20:
            output.append(block)
            continue
        boxes = numpy.asarray(tuple(line_bbox(line) for line in block.lines))
        indexes = row_order_indexes(numpy.arange(len(block.lines)), boxes)
        ordered = tuple(block.lines[int(index)] for index in indexes)
        output.append(replace(block, lines=ordered))
    return output


def has_repeated_block_columns(blocks: tuple[ParsedBlock, ...]) -> bool:
    bounded = tuple(block.bbox for block in blocks if block.bbox is not None)
    if len(bounded) < 6:
        return False
    top = max(box[3] for box in bounded)
    bottom = min(box[1] for box in bounded)
    cutoff = top - (top - bottom) * 0.55
    starts = sorted(box[0] for box in bounded if box[3] >= cutoff)
    if len(starts) < 6:
        return False
    clusters: list[list[float]] = []
    for start in starts:
        if clusters and start - clusters[-1][-1] <= 16.0:
            clusters[-1].append(start)
        else:
            clusters.append([start])
    return sum(len(cluster) >= 3 for cluster in clusters) >= 3


class LayoutRegion(Record):
    __slots__ = ("indexes", "x_start_order", "y_start_order", "y_center_order")

    indexes: numpy.ndarray
    x_start_order: numpy.ndarray
    y_start_order: numpy.ndarray
    y_center_order: numpy.ndarray

    __fields__: ClassVar[tuple[str, ...]] = (
        "indexes",
        "x_start_order",
        "y_start_order",
        "y_center_order",
    )
    __match_args__ = ("indexes", "x_start_order", "y_start_order", "y_center_order")

    def __init__(
        self,
        indexes: numpy.ndarray,
        x_start_order: numpy.ndarray,
        y_start_order: numpy.ndarray,
        y_center_order: numpy.ndarray,
    ) -> None:
        frozen_setattr(self, "indexes", indexes)
        frozen_setattr(self, "x_start_order", x_start_order)
        frozen_setattr(self, "y_start_order", y_start_order)
        frozen_setattr(self, "y_center_order", y_center_order)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.indexes == other.indexes
            and self.x_start_order == other.x_start_order
            and self.y_start_order == other.y_start_order
            and self.y_center_order == other.y_center_order
        )

    def __hash__(self) -> int:
        return hash((self.indexes, self.x_start_order, self.y_start_order, self.y_center_order))


class LayoutGeometry(ReplaceFields, ReprFields):
    __slots__ = ("boxes", "x_centers", "y_centers", "heights", "marks", "row_ids")

    boxes: numpy.ndarray
    x_centers: numpy.ndarray
    y_centers: numpy.ndarray
    heights: numpy.ndarray
    marks: numpy.ndarray
    row_ids: numpy.ndarray

    __fields__: ClassVar[tuple[str, ...]] = (
        "boxes",
        "x_centers",
        "y_centers",
        "heights",
        "marks",
        "row_ids",
    )
    __match_args__ = ("boxes", "x_centers", "y_centers", "heights", "marks", "row_ids")

    def __init__(
        self,
        boxes: numpy.ndarray,
        x_centers: numpy.ndarray,
        y_centers: numpy.ndarray,
        heights: numpy.ndarray,
        marks: numpy.ndarray,
        row_ids: numpy.ndarray,
    ) -> None:
        self.boxes = boxes
        self.x_centers = x_centers
        self.y_centers = y_centers
        self.heights = heights
        self.marks = marks
        self.row_ids = row_ids

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.boxes == other.boxes
            and self.x_centers == other.x_centers
            and self.y_centers == other.y_centers
            and self.heights == other.heights
            and self.marks == other.marks
            and self.row_ids == other.row_ids
        )

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def create(cls, boxes: numpy.ndarray) -> LayoutGeometry:
        x_centers = (boxes[:, 0] + boxes[:, 2]) * 0.5
        y_centers = (boxes[:, 1] + boxes[:, 3]) * 0.5
        return cls(
            boxes=boxes,
            x_centers=x_centers,
            y_centers=y_centers,
            heights=numpy.maximum(1.0, boxes[:, 3] - boxes[:, 1]),
            marks=numpy.zeros(len(boxes), dtype=numpy.bool_),
            row_ids=numpy.empty(len(boxes), dtype=numpy.int64),
        )

    def region(self, indexes: numpy.ndarray, parent: LayoutRegion | None = None) -> LayoutRegion:
        if parent is None:
            return LayoutRegion(
                indexes=indexes,
                x_start_order=indexes[numpy.argsort(self.boxes[indexes, 0], kind="stable")],
                y_start_order=indexes[numpy.argsort(self.boxes[indexes, 1], kind="stable")],
                y_center_order=indexes[numpy.argsort(-self.y_centers[indexes], kind="stable")],
            )
        self.marks[indexes] = True
        region = LayoutRegion(
            indexes=indexes,
            x_start_order=parent.x_start_order[self.marks[parent.x_start_order]],
            y_start_order=parent.y_start_order[self.marks[parent.y_start_order]],
            y_center_order=parent.y_center_order[self.marks[parent.y_center_order]],
        )
        self.marks[indexes] = False
        return region


def projection_gap_from_sorted(
    sorted_starts: numpy.ndarray,
    sorted_ends: numpy.ndarray,
    axis: int,
    minimum_gap: float,
) -> tuple[float, float] | None:
    previous_ends = numpy.maximum.accumulate(sorted_ends)[:-1]
    gaps = sorted_starts[1:] - previous_ends
    if not len(gaps):
        return None
    best_index = (
        len(gaps) - 1 - int(numpy.argmax(gaps[::-1])) if axis == 1 else int(numpy.argmax(gaps))
    )
    best_gap = float(gaps[best_index])
    best_cut = float((sorted_starts[best_index + 1] + previous_ends[best_index]) * 0.5)
    return (best_gap, best_cut) if best_gap >= minimum_gap else None


def best_projection_gap(
    boxes: numpy.ndarray,
    axis: int,
    minimum_gap: float,
) -> tuple[float, float] | None:
    starts = boxes[:, axis]
    ends = boxes[:, axis + 2]
    order = numpy.argsort(starts, kind="stable")
    return projection_gap_from_sorted(starts[order], ends[order], axis, minimum_gap)


def best_region_projection_gap(
    geometry: LayoutGeometry,
    region: LayoutRegion,
    axis: int,
    minimum_gap: float,
) -> tuple[float, float] | None:
    order = region.x_start_order if axis == 0 else region.y_start_order
    return projection_gap_from_sorted(
        geometry.boxes[order, axis],
        geometry.boxes[order, axis + 2],
        axis,
        minimum_gap,
    )


def gutter_tolerating_contained_boxes(
    region_boxes: numpy.ndarray, minimum_gap: float
) -> float | None:
    count = len(region_boxes)
    if count < 8:
        return None
    left = float(region_boxes[:, 0].min())
    right = float(region_boxes[:, 2].max())
    if right - left <= minimum_gap * 2:
        return None
    positions = numpy.linspace(left, right, 256)
    crossing = interval_crossing_counts(region_boxes, positions)
    allowed = max(1, count // 20)
    quiet = crossing <= allowed
    if not quiet.any():
        return None
    padded = numpy.concatenate(([False], quiet, [False]))
    edges = numpy.flatnonzero(padded[1:] != padded[:-1])
    margin = float(positions[1] - positions[0]) * 0.5 if len(positions) > 1 else 0.0
    runs = [
        (float(positions[run_start]) - margin, float(positions[run_end - 1]) + margin)
        for run_start, run_end in zip(edges[0::2], edges[1::2], strict=True)
        if float(positions[run_start]) > left and float(positions[run_end - 1]) < right
    ]
    if not runs:
        return None

    def unspanned(low: float, high: float) -> bool:
        spanning = (region_boxes[:, 0] < low) & (region_boxes[:, 2] > high)
        return not bool(spanning.any())

    def enclosed(low: float, high: float) -> int:
        inside = (region_boxes[:, 0] >= low) & (region_boxes[:, 2] <= high)
        return int(inside.sum())

    best: tuple[float, float] | None = None
    for index, (low, _high) in enumerate(runs):
        span_high = runs[index][1]
        for next_low, next_high in runs[index + 1 :]:
            if not unspanned(low, next_high) or enclosed(low, next_high) > allowed:
                break
            span_high = next_high
        if (
            unspanned(low, span_high)
            and enclosed(low, span_high) <= allowed
            and span_high - low >= minimum_gap
            and (best is None or span_high - low > best[1] - best[0])
        ):
            best = (low, span_high)
    if best is None:
        return None
    return (best[0] + best[1]) * 0.5


def interval_crossing_counts(boxes: numpy.ndarray, positions: numpy.ndarray) -> numpy.ndarray:
    sorted_starts = numpy.sort(boxes[:, 0])
    sorted_ends = numpy.sort(boxes[:, 2])
    return numpy.searchsorted(sorted_starts, positions, side="left") - numpy.searchsorted(
        sorted_ends, positions, side="right"
    )


def column_gap_minimum(region_boxes: numpy.ndarray) -> float:
    if not len(region_boxes):
        return 12.0
    median_width = finite_median(region_boxes[:, 2] - region_boxes[:, 0])
    return max(12.0, median_width * 0.05)


def narrow_column_gap_minimum(region_boxes: numpy.ndarray) -> float:
    if not len(region_boxes):
        return 12.0
    median_width = finite_median(region_boxes[:, 2] - region_boxes[:, 0])
    median_height = finite_median(region_boxes[:, 3] - region_boxes[:, 1])
    return max(6.0, min(12.0, median_width * 0.08), median_height * 0.9)


def columnar_split_alignment(region_boxes: numpy.ndarray, cut: float) -> bool:
    centers = (region_boxes[:, 0] + region_boxes[:, 2]) * 0.5
    left = region_boxes[centers < cut]
    right = region_boxes[centers >= cut]
    if len(left) < 4 or len(right) < 4:
        return False

    def aligned(side: numpy.ndarray) -> bool:
        starts = side[:, 0]
        median_start = finite_median(starts)
        return float(numpy.mean(numpy.abs(starts - median_start) <= 3.0)) >= 0.6

    if not (aligned(left) and aligned(right)):
        return False
    left_width = finite_median(left[:, 2] - left[:, 0])
    right_width = finite_median(right[:, 2] - right[:, 0])
    return min(left_width, right_width) >= max(left_width, right_width) * 0.5


def narrow_projection_gap(
    region_boxes: numpy.ndarray,
) -> tuple[float, float] | None:
    narrow_minimum = narrow_column_gap_minimum(region_boxes)
    if narrow_minimum >= column_gap_minimum(region_boxes):
        return None
    narrow = best_projection_gap(region_boxes, 0, narrow_minimum)
    if narrow is None or not columnar_split_alignment(region_boxes, narrow[1]):
        return None
    return narrow


def peel_spanning_band(
    indexes: numpy.ndarray,
    boxes: numpy.ndarray,
    median_height: float,
    *,
    from_bottom: bool = False,
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    if len(indexes) < 4:
        return None
    region = boxes[indexes]
    if from_bottom:
        edges = region[:, 1]
        order = numpy.argsort(edges, kind="stable")
    else:
        edges = region[:, 3]
        order = numpy.argsort(-edges, kind="stable")
    limit = len(indexes) // 2
    taken = 0
    while taken < limit:
        band_edge = float(edges[order[taken]])
        band_end = taken
        while band_end < len(order) and abs(band_edge - float(edges[order[band_end]])) <= max(
            1.0, median_height * 0.5
        ):
            band_end += 1
        if band_end >= len(order):
            return None
        taken = band_end
        remainder_indexes = indexes[order[taken:]]
        if len(remainder_indexes) < 2:
            return None
        remainder = boxes[remainder_indexes]
        gutter = best_projection_gap(remainder, 0, column_gap_minimum(remainder))
        if gutter is None:
            gutter = narrow_projection_gap(remainder)
        if gutter is not None:
            if from_bottom:
                if not columnar_split_alignment(remainder, gutter[1]):
                    continue
                centers = (remainder[:, 0] + remainder[:, 2]) * 0.5
                left_count = int(numpy.count_nonzero(centers < gutter[1]))
                if min(left_count, len(remainder) - left_count) < 18:
                    continue
            return indexes[order[:taken]], remainder_indexes
    return None


def assign_row_bands(
    order: numpy.ndarray,
    centers: numpy.ndarray,
    tolerance: float,
    row_ids: numpy.ndarray,
) -> None:
    current_row = 0
    row_center = float(centers[order[0]])
    for position, raw_item in enumerate(order):
        item = int(raw_item)
        center = float(centers[item])
        if position and row_center - center > tolerance:
            current_row += 1
            row_center = center
        row_ids[item] = current_row


def row_order_indexes(indexes: numpy.ndarray, boxes: numpy.ndarray) -> numpy.ndarray:
    geometry = LayoutGeometry.create(boxes)
    return row_order_region(geometry, geometry.region(indexes))


def row_order_region(geometry: LayoutGeometry, region: LayoutRegion) -> numpy.ndarray:
    indexes = region.indexes
    if len(indexes) < 2:
        return indexes
    tolerance = max(1.0, finite_median(geometry.heights[indexes]) * 0.5)
    if not math.isfinite(tolerance):
        tolerance = 1.0
    assign_row_bands(
        region.y_center_order,
        geometry.y_centers,
        tolerance,
        geometry.row_ids,
    )
    return indexes[numpy.lexsort((geometry.boxes[indexes, 0], geometry.row_ids[indexes]))]


def partition_by_obstacles(
    indexes: numpy.ndarray,
    boxes: numpy.ndarray,
    obstacles: tuple[tuple[float, float, float, float], ...],
    used_obstacles: frozenset[int] = frozenset(),
) -> tuple[tuple[numpy.ndarray, ...], int] | None:
    if not obstacles or len(indexes) < 3:
        return None
    region = boxes[indexes]
    region_box = (
        float(numpy.min(region[:, 0])),
        float(numpy.min(region[:, 1])),
        float(numpy.max(region[:, 2])),
        float(numpy.max(region[:, 3])),
    )
    region_width = max(1.0, region_box[2] - region_box[0])
    region_height = max(1.0, region_box[3] - region_box[1])
    centers_x = (region[:, 0] + region[:, 2]) * 0.5
    centers_y = (region[:, 1] + region[:, 3]) * 0.5
    for current_obstacle_index, obstacle in enumerate(obstacles):
        if current_obstacle_index in used_obstacles:
            continue
        x0, y0, x1, y1 = obstacle
        obstacle_width = max(0.0, x1 - x0)
        obstacle_height = max(0.0, y1 - y0)
        if obstacle_width / region_width >= 0.70:
            groups = (
                indexes[centers_y > y1],
                indexes[(centers_y >= y0) & (centers_y <= y1)],
                indexes[centers_y < y0],
            )
        elif obstacle_height / region_height >= 0.70:
            groups = (
                indexes[centers_x < x0],
                indexes[(centers_x >= x0) & (centers_x <= x1)],
                indexes[centers_x > x1],
            )
        else:
            continue
        populated = tuple(group for group in groups if len(group))
        if len(populated) >= 2:
            return populated, current_obstacle_index
    return None


def xy_cut_regions(
    indexes: numpy.ndarray,
    boxes: numpy.ndarray,
    obstacles: tuple[tuple[float, float, float, float], ...],
    median_height: float,
    *,
    depth: int = 0,
    used_obstacles: frozenset[int] = frozenset(),
    geometry: LayoutGeometry | None = None,
    parent_region: LayoutRegion | None = None,
) -> list[numpy.ndarray]:
    if geometry is None:
        geometry = LayoutGeometry.create(boxes)
    current_region = geometry.region(indexes, parent_region)
    if len(indexes) <= 2 or depth >= 32:
        return [row_order_region(geometry, current_region)]

    def recurse(
        group: numpy.ndarray,
        used: frozenset[int] = used_obstacles,
    ) -> list[numpy.ndarray]:
        return xy_cut_regions(
            group,
            boxes,
            obstacles,
            median_height,
            depth=depth + 1,
            used_obstacles=used,
            geometry=geometry,
            parent_region=current_region,
        )

    obstacle_partition = partition_by_obstacles(
        indexes,
        boxes,
        obstacles,
        used_obstacles,
    )
    if obstacle_partition is not None:
        groups, used_obstacle = obstacle_partition
        next_used_obstacles = used_obstacles | {used_obstacle}
        return [region for group in groups for region in recurse(group, next_used_obstacles)]

    region_boxes = boxes[indexes]
    horizontal = best_region_projection_gap(
        geometry, current_region, 1, max(3.0, median_height * 0.90)
    )
    vertical = best_region_projection_gap(
        geometry, current_region, 0, column_gap_minimum(region_boxes)
    )
    if vertical is None:
        vertical = narrow_projection_gap(region_boxes)
    candidates: list[tuple[float, int, float]] = []
    if horizontal is not None:
        candidates.append((horizontal[0] / median_height * 1.15, 1, horizontal[1]))
    if vertical is not None:
        candidates.append((vertical[0] / median_height, 0, vertical[1]))
    if not candidates:
        tolerant_cut = gutter_tolerating_contained_boxes(
            region_boxes, column_gap_minimum(region_boxes)
        )
        if tolerant_cut is not None:
            centers_x = geometry.x_centers[indexes]
            left = indexes[centers_x < tolerant_cut]
            right = indexes[centers_x >= tolerant_cut]
            if len(left) and len(right):
                return [region for group in (left, right) for region in recurse(group)]
        peeled = peel_spanning_band(indexes, boxes, median_height)
        if peeled is not None:
            band, remainder = peeled
            return [
                row_order_region(geometry, geometry.region(band, current_region)),
                *recurse(remainder),
            ]
        peeled = peel_spanning_band(indexes, boxes, median_height, from_bottom=True)
        if peeled is not None:
            band, remainder = peeled
            return [
                *recurse(remainder),
                row_order_region(geometry, geometry.region(band, current_region)),
            ]
        return [row_order_region(geometry, current_region)]

    _, axis, cut = max(candidates, key=lambda item: item[0])
    centers = (geometry.x_centers if axis == 0 else geometry.y_centers)[indexes]
    first = indexes[centers < cut]
    second = indexes[centers >= cut]
    if not len(first) or not len(second):
        return [row_order_region(geometry, current_region)]
    ordered_groups = (second, first) if axis == 1 else (first, second)
    return [region for group in ordered_groups for region in recurse(group)]
