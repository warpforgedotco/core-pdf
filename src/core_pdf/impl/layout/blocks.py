# SPDX-License-Identifier: AGPL-3.0-only
"""Group runs into lines and blocks; determine reading order."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from typing import cast

import numpy

from core_pdf.impl.extract.contracts import (
    ObservationBatch,
    ObservationSource,
    ParsedBlock,
    ParsedLine,
    ReadingOrderEvidence,
    internal_bbox_tuple,
)
from core_pdf.impl.layout import order as layout_order
from core_pdf.impl.layout import regions as layout_regions
from core_pdf.impl.layout.lines import LayoutLine
from core_pdf.impl.layout.spatial import (
    SpatialIndex,
)
from core_pdf.impl.model.geometry import interval_overlap
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.output import (
    TextSpan,
)
from core_pdf.impl.records import (
    TextWord,
    internal_reconcile_text_words,
    internal_text_word_tokens,
)
from core_pdf.impl.runtime.array_views import finite_median
from core_pdf.impl.text import collapse_leader_runs, collapse_ws

# ``ObservationBatch.source`` is a ``uint8`` column, so the OCR test is a vectorized
# comparison against a preconverted scalar rather than a per-observation Python loop.
internal_OCR_SOURCE = numpy.uint8(ObservationSource.OCR)
internal_NATIVE_SOURCE = int(ObservationSource.NATIVE)

internal_CAPTION_RE = re.compile(r"^(?:figure|fig\.|table|chart|exhibit)\s+\d+\b")
internal_LIST_MARKER_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+")


@dataclass(frozen=True, slots=True)
class internal_LineGroupPlan:
    indexes: numpy.ndarray
    starts: numpy.ndarray
    stops: numpy.ndarray


@dataclass(frozen=True, slots=True)
class internal_BuiltLines:
    lines: tuple[ParsedLine, ...]
    boxes: numpy.ndarray


def internal_line_group_indexes(observations: ObservationBatch) -> internal_LineGroupPlan:
    if not len(observations):
        empty = numpy.empty(0, dtype=numpy.int64)
        return internal_LineGroupPlan(empty, empty, empty)
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
    return internal_LineGroupPlan(indexes, starts, stops)


def internal_style_enabled(reference: object, name: str) -> bool:
    value = getattr(reference, name, False)
    return bool(value() if callable(value) else value)


def internal_group_text_and_words(
    observations: ObservationBatch,
    indexes: numpy.ndarray,
    *,
    may_contain_ocr: bool = True,
) -> tuple[str, tuple[TextWord, ...]]:
    # A group whose source range collapses to NATIVE cannot hold an OCR observation,
    # and the caller already has that range from its columnar reduction.  Native pages
    # therefore reach the reordering test without touching the source column at all.
    if may_contain_ocr and bool((observations.source[indexes] == internal_OCR_SOURCE).any()):
        rotation = int(observations.rotation[indexes[0]]) % 360
        boxes = observations.bbox[indexes]
        if rotation == 90:
            positions = (boxes[:, 1] + boxes[:, 3]) * 0.5
        elif rotation == 180:
            positions = -(boxes[:, 0] + boxes[:, 2]) * 0.5
        elif rotation == 270:
            positions = -(boxes[:, 1] + boxes[:, 3]) * 0.5
        else:
            positions = (boxes[:, 0] + boxes[:, 2]) * 0.5
        position_values = positions.tolist()

        # One pass over the text counts both directions; ASCII text can only
        # contribute L characters, and only letters carry a strong class.
        rtl = 0
        ltr = 0
        bidirectional = unicodedata.bidirectional
        for index in indexes:
            observation_text = observations.text[index]
            if observation_text.isascii():
                ltr += sum(map(str.isalpha, observation_text))
                continue
            for character in observation_text:
                direction_class = bidirectional(character)
                if direction_class == "L":
                    ltr += 1
                elif direction_class in {"R", "AL", "AN"}:
                    rtl += 1
        order = sorted(
            range(len(position_values)), key=position_values.__getitem__, reverse=rtl > ltr
        )
        index_values = indexes.tolist()
        indexes = cast(numpy.ndarray, numpy.asarray([index_values[position] for position in order]))
    references = tuple(observations.references[index] for index in indexes)
    if references and all(isinstance(reference, TextRun) for reference in references):
        runs = cast(list[TextRun], list(references))
        line = LayoutLine(runs)
        text = line.reconstructed_text().text.strip()
        layout_words = line.cached_text_and_words()[1]
        return text, internal_reconcile_text_words(text, layout_words)
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
        tokens = internal_text_word_tokens(text)
        bbox = observations.bbox[index]
        word_bbox = internal_bbox_tuple(bbox) if len(tokens) == 1 else None
        candidate_words.extend(TextWord(token, bbox=word_bbox) for token in tokens)
    combined = "".join(parts)
    return combined, internal_reconcile_text_words(combined, tuple(candidate_words))


def internal_looks_like_native_artifact(text: str) -> bool:
    """Reject symbol-heavy native lines produced by damaged text layers.

    Some PDFs expose decorative rules, malformed glyph mappings, and dotted
    leaders as ordinary text runs.  They are not OCR observations and are
    therefore safe to reject only after line reconstruction, where the whole
    artifact is visible.  Requiring a small alphanumeric count keeps compact
    identifiers and schematic labels intact.
    """
    # Unicode punctuation and scripts can be valid standalone text runs.  The
    # damaged mappings this targets are emitted as ASCII-looking rules and
    # dotted leaders, so leave non-ASCII lines untouched.
    if not text.isascii():
        return False
    nonspace_count = 0
    alphanumeric = 0
    for character in text:
        if character.isspace():
            continue
        nonspace_count += 1
        if character.isalnum():
            alphanumeric += 1
            if alphanumeric >= 12:
                return False
    if not nonspace_count:
        return False
    return (nonspace_count - alphanumeric) / nonspace_count >= 0.60


def internal_repeated_native_label_tokens(
    observations: ObservationBatch,
    indexes: numpy.ndarray,
) -> frozenset[str]:
    counts: Counter[str] = Counter()
    for index in indexes:
        text = collapse_ws(observations.text[index])
        if len(text) == 1 and text.isascii() and text.isalpha():
            counts[text.casefold()] += 1
    return frozenset(token for token, count in counts.items() if count >= 4)


def internal_is_repeated_native_label(text: str, repeated_tokens: frozenset[str]) -> bool:
    parts = text.casefold().split()
    return bool(parts) and all(len(part) == 1 and part in repeated_tokens for part in parts)


def internal_clean_native_punctuation_runs(text: str) -> str:
    return collapse_leader_runs(text)


def internal_color_is_emphasis(color: object) -> bool:
    if not isinstance(color, (tuple, list)) or len(color) < 3:
        return False
    components: list[float] = []
    for component in color[:3]:
        if not isinstance(component, (int, float)):
            return False
        components.append(float(component))
    return max(components) - min(components) >= 0.15


def internal_build_lines(observations: ObservationBatch) -> internal_BuiltLines:
    line_groups = internal_line_group_indexes(observations)
    if not len(line_groups.starts):
        return internal_BuiltLines((), numpy.empty((0, 4), dtype=numpy.float32))
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
    repeated_native_labels = internal_repeated_native_label_tokens(
        observations,
        selected,
    )
    output: list[ParsedLine] = []
    output_boxes: list[numpy.ndarray] = []
    for group_index, (start, stop) in enumerate(
        zip(line_groups.starts, line_groups.stops, strict=True)
    ):
        indexes = selected[int(start) : int(stop)]
        all_native = (
            source_minimum[group_index] == source_maximum[group_index] == internal_NATIVE_SOURCE
        )
        text, words = internal_group_text_and_words(
            observations,
            indexes,
            may_contain_ocr=not all_native,
        )
        if not text:
            continue
        if all_native and internal_looks_like_native_artifact(text):
            continue
        if (
            repeated_native_labels
            and all_native
            and internal_is_repeated_native_label(text, repeated_native_labels)
        ):
            continue
        if all_native:
            text = internal_clean_native_punctuation_runs(text)
            if not text:
                continue
        words = internal_reconcile_text_words(text, words)
        confidences = observations.confidence[indexes]
        font_sizes = observations.font_size[indexes]
        finite_confidences = confidences[numpy.isfinite(confidences)]
        finite_font_sizes = font_sizes[numpy.isfinite(font_sizes) & (font_sizes > 0)]
        native_references = [
            reference
            for reference in (observations.references[index] for index in indexes)
            if reference is not None
        ]
        # Each reference's bold/italic state was resolved four times below -- twice for
        # the majority vote and twice when its span is built.  Resolve each once.
        reference_styles = [
            (
                internal_style_enabled(reference, "is_bold"),
                internal_style_enabled(reference, "is_italic"),
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
                    mark=internal_color_is_emphasis(getattr(reference, "fill_color", None)),
                )
            )
            pending_space = reference.text.endswith((" ", "\t", "\n"))
        spans = tuple(span_values)
        if not spans or "".join(span.text for span in spans) != text:
            spans = ()
        source_low = int(source_minimum[group_index])
        source_high = int(source_maximum[group_index])
        if source_low == source_high == int(ObservationSource.NATIVE):
            source = "native"
        elif source_low == source_high == int(ObservationSource.OCR):
            source = "ocr"
        else:
            source = "hybrid"
        words = tuple(replace(word, source=source) for word in words)
        group_box = group_boxes[group_index]
        output.append(
            ParsedLine(
                text=text,
                bbox=(
                    float(group_box[0]),
                    float(group_box[1]),
                    float(group_box[2]),
                    float(group_box[3]),
                ),
                source=source,
                confidence=(
                    float(numpy.mean(finite_confidences)) if len(finite_confidences) else None
                ),
                sequence=int(group_sequences[group_index]),
                rotation=int(observations.rotation[indexes[0]]),
                font_size=(finite_median(finite_font_sizes) if len(finite_font_sizes) else None),
                bold=bold,
                italic=italic,
                spans=spans,
                words=words,
            )
        )
        output_boxes.append(group_box)
    boxes = (
        numpy.asarray(output_boxes, dtype=numpy.float32).reshape((-1, 4))
        if output_boxes
        else numpy.empty((0, 4), dtype=numpy.float32)
    )
    return internal_BuiltLines(tuple(output), boxes)


def internal_assign_columns(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    if len(blocks) < 2:
        return blocks
    page_x0 = min(block.bbox[0] for block in blocks)
    page_x1 = max(block.bbox[2] for block in blocks)
    page_width = max(1.0, page_x1 - page_x0)
    bands: list[list[float]] = []
    assignments: list[int | None] = []
    for block in blocks:
        x0, internal_y0, x1, internal_y1 = block.bbox
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


def internal_classify_blocks(
    blocks: list[ParsedBlock],
    *,
    body_font_size: float | None,
) -> list[ParsedBlock]:
    """Add conservative semantic roles using typography and stable text cues."""
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
        text = " ".join(line.text for line in block.lines)
        normalized = collapse_ws(text)
        kind = "paragraph"
        level: int | None = None
        lowered = normalized.casefold()
        if internal_CAPTION_RE.match(lowered):
            kind = "caption"
        elif block.lines and all(
            internal_LIST_MARKER_RE.match(line.text.strip()) for line in block.lines
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


def internal_semantic_body_font_size(lines: tuple[ParsedLine, ...]) -> float | None:
    sizes = numpy.asarray(
        [line.font_size for line in lines if line.font_size is not None and line.font_size > 0],
        dtype=numpy.float32,
    )
    return finite_median(sizes) if len(sizes) else None


def internal_display_boxes(
    boxes: numpy.ndarray, rotation: int, width: float, height: float
) -> numpy.ndarray:
    """Map boxes into the frame the page is displayed in.

    Reading order is a statement about what a reader sees, so it has to be
    decided in the rotated frame. A page carrying /Rotate 180 stores its first
    line at the bottom of unrotated space, and ordering there walks the page
    backwards -- one benchmark page came out reading its outline items l, k, j,
    i, h. Ordering alone is rotated: the boxes handed back to callers stay in
    page space, where the rest of the engine expects them.
    """
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
) -> tuple[ParsedBlock, ...]:
    """Reduce fused observations into geometrically ordered, structured blocks."""
    built_lines = internal_build_lines(observations)
    lines = built_lines.lines
    if not lines:
        return ()
    boxes = internal_display_boxes(
        built_lines.boxes,
        rotation,
        page_width,
        page_height,
    )
    if obstacles:
        obstacles = tuple(
            internal_bbox_tuple(box)
            for box in internal_display_boxes(
                numpy.asarray(obstacles, dtype=numpy.float32),
                rotation,
                page_width,
                page_height,
            )
        )
    if not use_xy_cut:
        indexes = layout_regions.internal_row_order_indexes(
            numpy.arange(len(lines), dtype=numpy.int64),
            boxes,
        )
        blocks = [
            ParsedBlock(lines=(lines[int(index)],), bbox=lines[int(index)].bbox)
            for index in indexes
        ]
        return tuple(
            internal_classify_blocks(
                internal_assign_columns(blocks),
                body_font_size=internal_semantic_body_font_size(lines),
            )
        )
    heights = numpy.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    median_height = max(1.0, finite_median(heights))
    obstacle_index = (
        SpatialIndex(((index, obstacle) for index, obstacle in enumerate(obstacles)))
        if obstacles
        else None
    )
    regions = layout_regions.internal_xy_cut_regions(
        numpy.arange(len(lines), dtype=numpy.int64),
        boxes,
        obstacles,
        median_height,
        obstacle_index=obstacle_index,
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
    blocks = layout_order.internal_interleave_columnar_blocks(blocks)
    blocks = layout_order.internal_transpose_numeric_table_blocks(blocks)
    blocks = layout_order.internal_column_major_prose(blocks)
    blocks = layout_order.internal_topological_block_order(blocks)
    return tuple(
        internal_classify_blocks(
            internal_assign_columns(blocks), body_font_size=internal_semantic_body_font_size(lines)
        )
    )


def layout_blocks_with_evidence(
    observations: ObservationBatch,
    *,
    obstacles: tuple[tuple[float, float, float, float], ...] = (),
    use_xy_cut: bool = True,
    rotation: int = 0,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> tuple[tuple[ParsedBlock, ...], ReadingOrderEvidence]:
    """Return ordered blocks together with validation evidence."""
    blocks = layout_blocks(
        observations,
        obstacles=obstacles,
        use_xy_cut=use_xy_cut,
        rotation=rotation,
        page_width=page_width,
        page_height=page_height,
    )
    return blocks, layout_order.internal_reading_order_evidence(blocks)


def layout_element_order(
    boxes: tuple[tuple[float, float, float, float], ...],
    rotation: int = 0,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> tuple[int, ...]:
    """Return reading order for arbitrary page elements represented by boxes."""
    if len(boxes) < 2:
        return tuple(range(len(boxes)))
    values = internal_display_boxes(
        numpy.asarray(boxes, dtype=numpy.float32), rotation, page_width, page_height
    )
    heights = numpy.maximum(1.0, values[:, 3] - values[:, 1])
    # A full-width element -- a table or figure set across the text -- is itself
    # the obstacle that divides the page above it from the page below. Ordering
    # elements without saying so leaves its box bridging the column gutter, so
    # no column split is available and a two-column page falls back to row order
    # with its columns interleaved.
    span = max(1.0, float(values[:, 2].max() - values[:, 0].min()))
    obstacles = tuple(
        internal_bbox_tuple(box) for box in values if (box[2] - box[0]) / span >= 0.70
    )
    regions = layout_regions.internal_xy_cut_regions(
        numpy.arange(len(boxes), dtype=numpy.int64),
        values,
        obstacles,
        max(1.0, finite_median(heights)),
        obstacle_index=(
            SpatialIndex(((index, obstacle) for index, obstacle in enumerate(obstacles)))
            if obstacles
            else None
        ),
    )
    return tuple(int(index) for region in regions for index in region)
