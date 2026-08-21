# SPDX-License-Identifier: AGPL-3.0-only
"""Group runs into lines and blocks; determine reading order."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import cast

import numpy

from core_pdf.impl.engine.array_views import finite_median
from core_pdf.impl.engine.layout.models import TextRun, reconstruct_cached_layout_line_text
from core_pdf.impl.engine.layout.spatial import (
    SpatialIndex,
)
from core_pdf.impl.engine.parse.model import (
    ObservationBatch,
    ObservationSource,
    ParsedBlock,
    ParsedLine,
    ReadingOrderEvidence,
)
from core_pdf.impl.engine.structured import (
    TextSpan,
)
from core_pdf.impl.text import collapse_ws

internal_NATIVE_DOTTED_LEADER_RE = re.compile(r"\.{2,}")
internal_NATIVE_DASH_RULE_RE = re.compile(r"(?:\s*-\s*){2,}")


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


def internal_group_text(observations: ObservationBatch, indexes: numpy.ndarray) -> str:
    if any(int(observations.source[index]) == int(ObservationSource.OCR) for index in indexes):
        rotation = int(observations.rotation[indexes[0]]) % 360

        def baseline_position(index: int) -> float:
            box = observations.bbox[index]
            if rotation == 90:
                return float((box[1] + box[3]) * 0.5)
            if rotation == 180:
                return -float((box[0] + box[2]) * 0.5)
            if rotation == 270:
                return -float((box[1] + box[3]) * 0.5)
            return float((box[0] + box[2]) * 0.5)

        rtl = sum(
            unicodedata.bidirectional(character) in {"R", "AL", "AN"}
            for index in indexes
            for character in observations.text[index]
        )
        ltr = sum(
            unicodedata.bidirectional(character) == "L"
            for index in indexes
            for character in observations.text[index]
        )
        indexes = numpy.asarray(sorted(indexes, key=baseline_position, reverse=rtl > ltr))
    references = tuple(observations.references[index] for index in indexes)
    if references and all(reference is not None for reference in references):
        runs = cast(list[TextRun], list(references))
        return reconstruct_cached_layout_line_text(runs).text.strip()
    parts: list[str] = []
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
    return "".join(parts)


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
    if any(ord(character) > 127 for character in text):
        return False
    nonspace = [character for character in text if not character.isspace()]
    if not nonspace:
        return False
    alphanumeric = sum(character.isalnum() for character in nonspace)
    if alphanumeric >= 12:
        return False
    return (len(nonspace) - alphanumeric) / len(nonspace) >= 0.60


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
    text = internal_NATIVE_DOTTED_LEADER_RE.sub(" ", text)
    text = internal_NATIVE_DASH_RULE_RE.sub(" ", text)
    return collapse_ws(text)


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
        text = internal_group_text(observations, indexes)
        if not text:
            continue
        all_native = (
            source_minimum[group_index]
            == source_maximum[group_index]
            == int(ObservationSource.NATIVE)
        )
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
        confidences = observations.confidence[indexes]
        font_sizes = observations.font_size[indexes]
        finite_confidences = confidences[numpy.isfinite(confidences)]
        finite_font_sizes = font_sizes[numpy.isfinite(font_sizes) & (font_sizes > 0)]
        native_references = [
            reference
            for reference in (observations.references[index] for index in indexes)
            if reference is not None
        ]

        def style_enabled(reference: object, name: str) -> bool:
            value = getattr(reference, name, False)
            return bool(value() if callable(value) else value)

        bold = bool(native_references) and sum(
            style_enabled(reference, "is_bold") for reference in native_references
        ) * 2 >= len(native_references)
        italic = bool(native_references) and sum(
            style_enabled(reference, "is_italic") for reference in native_references
        ) * 2 >= len(native_references)
        span_values: list[TextSpan] = []
        pending_space = False
        for reference in native_references:
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
                    bold=style_enabled(reference, "is_bold"),
                    italic=style_enabled(reference, "is_italic"),
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
            )
        )
        output_boxes.append(group_box)
    boxes = (
        numpy.asarray(output_boxes, dtype=numpy.float32).reshape((-1, 4))
        if output_boxes
        else numpy.empty((0, 4), dtype=numpy.float32)
    )
    return internal_BuiltLines(tuple(output), boxes)


def internal_best_projection_gap(
    boxes: numpy.ndarray,
    axis: int,
    minimum_gap: float,
) -> tuple[float, float] | None:
    starts = boxes[:, axis]
    ends = boxes[:, axis + 2]
    order = numpy.argsort(starts, kind="stable")
    sorted_starts = starts[order]
    sorted_ends = ends[order]
    previous_ends = numpy.maximum.accumulate(sorted_ends)[:-1]
    gaps = sorted_starts[1:] - previous_ends
    if not len(gaps):
        return None
    # PDF page space is bottom-left based.  For equal horizontal whitespace,
    # split at the uppermost gap first so a full-width header is detached before
    # column detection runs on the body below it.
    best_index = (
        len(gaps) - 1 - int(numpy.argmax(gaps[::-1])) if axis == 1 else int(numpy.argmax(gaps))
    )
    best_gap = float(gaps[best_index])
    best_cut = float((sorted_starts[best_index + 1] + previous_ends[best_index]) * 0.5)
    return (best_gap, best_cut) if best_gap >= minimum_gap else None


def internal_gutter_tolerating_contained_boxes(
    region_boxes: numpy.ndarray, minimum_gap: float
) -> float | None:
    """Find a column boundary that almost nothing crosses.

    A projection gap has to be completely empty, so a single mark lying in the
    gutter hides it. That happens constantly on recognised pages: a centred
    page number, a speck of noise, a stray accent. On one patent, four boxes
    26 units wide -- against a median line width of 860 -- hid the boundary
    from 135 lines and interleaved both columns for the whole page.

    Coverage tells the story the projection cannot: it runs at fifty-odd boxes
    across each column and collapses to one or two between them. So look for
    where the fewest boxes cross rather than where none do, and require the
    collapse to be near total, so ordinary text never splits down the middle.
    """
    count = len(region_boxes)
    if count < 8:
        return None
    left = float(region_boxes[:, 0].min())
    right = float(region_boxes[:, 2].max())
    if right - left <= minimum_gap * 2:
        return None
    positions = numpy.linspace(left, right, 256)
    # Count intervals crossing each sample without materializing a 256 x N
    # boolean matrix. Strict inequalities match the former broadcast exactly:
    # starts below the position minus ends at or below the position.
    crossing = internal_interval_crossing_counts(region_boxes, positions)
    # "Almost nothing" has to stay a small share of the region, or a page with
    # few lines would split on a single crossing.
    allowed = max(1, count // 20)
    quiet = crossing <= allowed
    if not quiet.any():
        return None
    # Take the widest quiet run, ignoring the margins outside the text.
    padded = numpy.concatenate(([False], quiet, [False]))
    edges = numpy.flatnonzero(padded[1:] != padded[:-1])
    # A run is measured between sampled points, so it stops short of the real
    # gutter by up to one step at each end. The sample count is fixed while the
    # region is not, so the wider the region the coarser that step and the more
    # every gutter in it is understated -- on a three-column page a true 17pt
    # gutter measured 11.98 and lost to a 12pt floor. The boundary lies between
    # the last quiet sample and the first busy one, so credit half a step.
    margin = float(positions[1] - positions[0]) * 0.5 if len(positions) > 1 else 0.0
    runs = [
        (float(positions[run_start]) - margin, float(positions[run_end - 1]) + margin)
        for run_start, run_end in zip(edges[0::2], edges[1::2], strict=True)
        if float(positions[run_start]) > left and float(positions[run_end - 1]) < right
    ]
    if not runs:
        return None

    def unspanned(low: float, high: float) -> bool:
        """Does nothing reach across this span into the text on both sides?

        A box poking in from one side is the last word of a line, not a
        heading: only one that starts left of the gutter and ends right of it
        joins the columns together.
        """
        spanning = (region_boxes[:, 0] < low) & (region_boxes[:, 2] > high)
        return not bool(spanning.any())

    def enclosed(low: float, high: float) -> int:
        """How much text sits wholly inside this span?

        Nothing spans a stretch drawn from the gutter on one side of a column to
        the gutter on the other: that column's own lines start and end within it
        and so cross neither edge. ``unspanned`` therefore cannot tell a mark
        left in the gutter from an entire column, and on a three-column page it
        will merge both gutters into one span and cut the middle column in half.
        What is enclosed separates them -- a speck is a handful of boxes, a
        column is most of the region.
        """
        inside = (region_boxes[:, 0] >= low) & (region_boxes[:, 2] <= high)
        return int(inside.sum())

    # A mark sitting in the gutter interrupts the quiet stretch without ending
    # it, so join neighbouring runs while everything crossing the join stays
    # inside the result. A box reaching into the columns on either side stops
    # the merge, which is what keeps ordinary text from splitting mid-line, and
    # so does a join that would swallow a column whole.
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


def internal_interval_crossing_counts(
    boxes: numpy.ndarray, positions: numpy.ndarray
) -> numpy.ndarray:
    """Count horizontal box intervals that strictly cross each position."""
    sorted_starts = numpy.sort(boxes[:, 0])
    sorted_ends = numpy.sort(boxes[:, 2])
    return numpy.searchsorted(sorted_starts, positions, side="left") - numpy.searchsorted(
        sorted_ends, positions, side="right"
    )


def internal_column_gap_minimum(region_boxes: numpy.ndarray) -> float:
    """Return the smallest horizontal gap that counts as a column gutter.

    Scaling this by element height, as the row threshold rightly does, is a
    category error: it judges a horizontal distance by a vertical one. It
    survives at line level only because a line's height is close to its text
    size, and breaks completely once the boxes are blocks, where a 141pt tall
    column demanded a 211pt gutter to separate from its neighbour 26pt away.

    A gutter is a horizontal distance, so scale it by one: the width of a
    typical box in the region. The floor keeps ordinary word spacing from
    qualifying on pages whose boxes are narrow.
    """
    if not len(region_boxes):
        return 12.0
    median_width = finite_median(region_boxes[:, 2] - region_boxes[:, 0])
    return max(12.0, median_width * 0.05)


def internal_narrow_column_gap_minimum(region_boxes: numpy.ndarray) -> float:
    """Return the narrower gutter minimum a well-aligned split may use.

    Narrow columns draw proportionally narrow gutters: a two-column
    small-print callout with 113pt columns separates them by 11pt, which
    the flat 12pt floor rejects, and the columns interleave line by line.
    A narrower gap is only trustworthy alongside the margin-alignment
    evidence checked by ``internal_columnar_split_alignment`` -- ragged
    recognized text offers same-width gaps purely by accident.

    The absolute floor only means something at native page scale; scanned
    pages measure several units per point, so anchor the floor to the
    region's own line height as well -- no real gutter is narrower than
    the text it separates is tall.
    """
    if not len(region_boxes):
        return 12.0
    median_width = finite_median(region_boxes[:, 2] - region_boxes[:, 0])
    median_height = finite_median(region_boxes[:, 3] - region_boxes[:, 1])
    return max(6.0, min(12.0, median_width * 0.08), median_height * 0.9)


def internal_columnar_split_alignment(region_boxes: numpy.ndarray, cut: float) -> bool:
    """Report whether both sides of a vertical cut read as real columns.

    Genuine columns hang their lines from a shared left margin, so most
    boxes on each side start within a few points of that side's median
    start. Accidental gaps through ragged text leave scattered starts and
    fail this immediately. Twin text columns are also close in width;
    a skinny aligned strip beside a wide one is a marker rail -- claim
    numbers, list bullets, row labels -- whose gap to the body text is
    not a gutter, so require the sides to balance as well.
    """
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


def internal_narrow_projection_gap(
    region_boxes: numpy.ndarray,
) -> tuple[float, float] | None:
    """Find a sub-12pt column gutter backed by margin-alignment evidence."""
    narrow_minimum = internal_narrow_column_gap_minimum(region_boxes)
    if narrow_minimum >= internal_column_gap_minimum(region_boxes):
        return None
    narrow = internal_best_projection_gap(region_boxes, 0, narrow_minimum)
    if narrow is None or not internal_columnar_split_alignment(region_boxes, narrow[1]):
        return None
    return narrow


def internal_peel_spanning_band(
    indexes: numpy.ndarray,
    boxes: numpy.ndarray,
    median_height: float,
    *,
    from_bottom: bool = False,
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    """Split an edge row band off a region that admits no projection cut.

    A projection gap only counts when nothing straddles it, so a single
    full-width line -- a heading centred over two columns is the usual case --
    hides the gutter beneath it from the whole region. There is no horizontal
    gap either, because between two columns every row band holds text. The
    region then falls back to row order and the columns interleave for good.

    Such an element cannot take part in a column split, so remove it and let
    the caller re-test what is left. A heading is often set over several lines,
    and a banner may carry a subtitle, so keep taking bands off the edge while
    the columns stay hidden -- stopping once half the region is gone, since
    past that the region is not a heading above columns at all.

    Spanners sit under columns as readily as over them -- a banner or footer
    across the page bottom hides the gutter exactly the same way -- so the
    caller may peel from either edge; it emits a bottom band after the
    remainder instead of before it.

    Returning None when no amount of peeling helps leaves regions that are
    genuinely unsplittable exactly as they were.
    """
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
        # Take everything sharing the outermost remaining band, so a heading
        # set across a line is removed whole rather than a word at a time.
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
        gutter = internal_best_projection_gap(remainder, 0, internal_column_gap_minimum(remainder))
        if gutter is None:
            gutter = internal_narrow_projection_gap(remainder)
        if gutter is not None:
            # Bottom peeling is speculative in a way top peeling is not:
            # receipts and invoices end in totals rows whose removal exposes a
            # label/value gap that reads row-major. Demand real column
            # evidence -- aligned margins, balanced widths, and a paragraph's
            # worth of lines on each side -- before trusting a gutter
            # uncovered from below.
            if from_bottom:
                if not internal_columnar_split_alignment(remainder, gutter[1]):
                    continue
                centers = (remainder[:, 0] + remainder[:, 2]) * 0.5
                left_count = int(numpy.count_nonzero(centers < gutter[1]))
                # A paragraph column runs a couple dozen lines; the aligned
                # key/value panels of invoices and receipts do not.
                if min(left_count, len(remainder) - left_count) < 18:
                    continue
            return indexes[order[:taken]], remainder_indexes
    return None


def internal_row_order_indexes(indexes: numpy.ndarray, boxes: numpy.ndarray) -> numpy.ndarray:
    """Order boxes into reading order: row bands top-to-bottom, then left-to-right."""
    region = boxes[indexes]
    if len(indexes) < 2:
        return indexes
    heights = numpy.maximum(1.0, region[:, 3] - region[:, 1])
    tolerance = max(1.0, finite_median(heights) * 0.5)
    if not math.isfinite(tolerance):
        tolerance = 1.0

    # Band by how far each box sits from the row already being built, rather
    # than by rounding its absolute position to a grid. Rounding puts the
    # boundary at an arbitrary offset, so two cells a point apart can land in
    # different rows while two a whole line apart share one -- which scrambles
    # the reading order of any table whose cells vary in height.
    centers = (region[:, 1] + region[:, 3]) * 0.5
    order = numpy.argsort(-centers, kind="stable")
    row_ids = numpy.empty(len(indexes), dtype=numpy.int64)
    current_row = 0
    row_center = float(centers[order[0]])
    for position, item in enumerate(order):
        center = float(centers[item])
        if position and row_center - center > tolerance:
            current_row += 1
            row_center = center
        row_ids[item] = current_row
    return indexes[numpy.lexsort((region[:, 0], row_ids))]


def internal_obstacle_partition(
    indexes: numpy.ndarray,
    boxes: numpy.ndarray,
    obstacles: tuple[tuple[float, float, float, float], ...],
    obstacle_index: SpatialIndex[int] | None = None,
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
    obstacle_indexes: Iterable[int] = (
        obstacle_index.intersecting(region_box)
        if obstacle_index is not None
        else range(len(obstacles))
    )
    for raw_obstacle_index in obstacle_indexes:
        current_obstacle_index = int(raw_obstacle_index)
        if current_obstacle_index in used_obstacles:
            continue
        obstacle = obstacles[current_obstacle_index]
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


def internal_xy_cut_regions(
    indexes: numpy.ndarray,
    boxes: numpy.ndarray,
    obstacles: tuple[tuple[float, float, float, float], ...],
    median_height: float,
    *,
    depth: int = 0,
    obstacle_index: SpatialIndex[int] | None = None,
    used_obstacles: frozenset[int] = frozenset(),
) -> list[numpy.ndarray]:
    if len(indexes) <= 2 or depth >= 32:
        return [internal_row_order_indexes(indexes, boxes)]

    obstacle_partition = internal_obstacle_partition(
        indexes,
        boxes,
        obstacles,
        obstacle_index,
        used_obstacles,
    )
    if obstacle_partition is not None:
        groups, used_obstacle = obstacle_partition
        next_used_obstacles = used_obstacles | {used_obstacle}
        return [
            region
            for group in groups
            for region in internal_xy_cut_regions(
                group,
                boxes,
                obstacles,
                median_height,
                depth=depth + 1,
                obstacle_index=obstacle_index,
                used_obstacles=next_used_obstacles,
            )
        ]

    region_boxes = boxes[indexes]
    horizontal = internal_best_projection_gap(region_boxes, 1, max(3.0, median_height * 0.90))
    vertical = internal_best_projection_gap(
        region_boxes, 0, internal_column_gap_minimum(region_boxes)
    )
    if vertical is None:
        vertical = internal_narrow_projection_gap(region_boxes)
    candidates: list[tuple[float, int, float]] = []
    if horizontal is not None:
        candidates.append((horizontal[0] / median_height * 1.15, 1, horizontal[1]))
    if vertical is not None:
        candidates.append((vertical[0] / median_height, 0, vertical[1]))
    if not candidates:
        tolerant_cut = internal_gutter_tolerating_contained_boxes(
            region_boxes, internal_column_gap_minimum(region_boxes)
        )
        if tolerant_cut is not None:
            centers_x = (region_boxes[:, 0] + region_boxes[:, 2]) * 0.5
            left = indexes[centers_x < tolerant_cut]
            right = indexes[centers_x >= tolerant_cut]
            if len(left) and len(right):
                return [
                    region
                    for group in (left, right)
                    for region in internal_xy_cut_regions(
                        group,
                        boxes,
                        obstacles,
                        median_height,
                        depth=depth + 1,
                        obstacle_index=obstacle_index,
                        used_obstacles=used_obstacles,
                    )
                ]
        peeled = internal_peel_spanning_band(indexes, boxes, median_height)
        if peeled is not None:
            band, remainder = peeled
            return [
                internal_row_order_indexes(band, boxes),
                *internal_xy_cut_regions(
                    remainder,
                    boxes,
                    obstacles,
                    median_height,
                    depth=depth + 1,
                    obstacle_index=obstacle_index,
                    used_obstacles=used_obstacles,
                ),
            ]
        peeled = internal_peel_spanning_band(indexes, boxes, median_height, from_bottom=True)
        if peeled is not None:
            band, remainder = peeled
            return [
                *internal_xy_cut_regions(
                    remainder,
                    boxes,
                    obstacles,
                    median_height,
                    depth=depth + 1,
                    obstacle_index=obstacle_index,
                    used_obstacles=used_obstacles,
                ),
                internal_row_order_indexes(band, boxes),
            ]
        return [internal_row_order_indexes(indexes, boxes)]

    internal_score, axis, cut = max(candidates, key=lambda item: item[0])
    centers = (region_boxes[:, axis] + region_boxes[:, axis + 2]) * 0.5
    first = indexes[centers < cut]
    second = indexes[centers >= cut]
    if not len(first) or not len(second):
        return [internal_row_order_indexes(indexes, boxes)]
    ordered_groups = (second, first) if axis == 1 else (first, second)
    return [
        region
        for group in ordered_groups
        for region in internal_xy_cut_regions(
            group,
            boxes,
            obstacles,
            median_height,
            depth=depth + 1,
            obstacle_index=obstacle_index,
            used_obstacles=used_obstacles,
        )
    ]


def internal_block_bbox(lines: tuple[ParsedLine, ...]) -> tuple[float, float, float, float]:
    boxes = numpy.asarray(tuple(line.bbox for line in lines), dtype=numpy.float32)
    return (
        float(numpy.min(boxes[:, 0])),
        float(numpy.min(boxes[:, 1])),
        float(numpy.max(boxes[:, 2])),
        float(numpy.max(boxes[:, 3])),
    )


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
            overlap = max(0.0, min(x1, band_x1) - max(x0, band_x0))
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
    for block in blocks:
        text = " ".join(line.text for line in block.lines)
        normalized = collapse_ws(text)
        kind = "paragraph"
        level: int | None = None
        lowered = normalized.casefold()
        if re.match(r"^(?:figure|fig\.|table|chart|exhibit)\s+\d+\b", lowered):
            kind = "caption"
        elif block.lines and all(
            re.match(r"^(?:[-*•]|\d+[.)])\s+", line.text.strip()) for line in block.lines
        ):
            kind = "list"
        elif (
            body_font_size is not None
            and len(block.lines) <= 3
            and len(normalized) <= 240
            and max((line.font_size or 0.0) for line in block.lines) >= body_font_size * 1.2
        ):
            kind = "heading"
            size = max((line.font_size or 0.0) for line in block.lines)
            level = min(3, heading_sizes.index(size) + 1) if size in heading_sizes else 1
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
            (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            for box in internal_display_boxes(
                numpy.asarray(obstacles, dtype=numpy.float32),
                rotation,
                page_width,
                page_height,
            )
        )
    if not use_xy_cut:
        indexes = internal_row_order_indexes(
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
    regions = internal_xy_cut_regions(
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
    blocks = internal_interleave_columnar_blocks(blocks)
    blocks = internal_transpose_numeric_table_blocks(blocks)
    blocks = internal_column_major_prose(blocks)
    blocks = internal_topological_block_order(blocks)
    return tuple(
        internal_classify_blocks(
            internal_assign_columns(blocks), body_font_size=internal_semantic_body_font_size(lines)
        )
    )


def internal_inversion_count(values: tuple[int, ...]) -> int:
    """Count source-order inversions in O(n log n) time and O(n) memory."""
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


def internal_reading_order_evidence(
    blocks: tuple[ParsedBlock, ...],
) -> ReadingOrderEvidence:
    """Summarize repair strength and ambiguity for an ordered block sequence."""
    lines = tuple(line for block in blocks for line in block.lines)
    sequences = tuple(line.sequence for line in lines)
    inversions = internal_inversion_count(sequences)
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
    return blocks, internal_reading_order_evidence(blocks)


def internal_topological_block_order(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    """Sort blocks into topological reading order using a spatial predecessor DAG.

    Full-width header blocks enforce strict vertical precedence over all child columns,
    and column blocks are ordered left-to-right while preserving top-down sequence inside
    each column.
    """
    if len(blocks) <= 2:
        return blocks
    page_x0 = min(block.bbox[0] for block in blocks)
    page_x1 = max(block.bbox[2] for block in blocks)
    page_width = max(1.0, page_x1 - page_x0)

    n = len(blocks)
    in_degree = [0] * n
    graph: dict[int, list[int]] = defaultdict(list)

    for i in range(n):
        ax0, ay0, ax1, ay1 = blocks[i].bbox
        awidth = ax1 - ax0
        a_is_full_width = (awidth / page_width) >= 0.70
        for j in range(i + 1, n):
            bx0, by0, bx1, by1 = blocks[j].bbox
            bwidth = bx1 - bx0
            b_is_full_width = (bwidth / page_width) >= 0.70

            # Rule 1: Full-width header preceding child blocks below it
            if a_is_full_width and not b_is_full_width and ay0 >= by1 - 2.0:
                graph[i].append(j)
                in_degree[j] += 1
            elif b_is_full_width and not a_is_full_width and by0 >= ay1 - 2.0:
                graph[j].append(i)
                in_degree[i] += 1
            elif not a_is_full_width and not b_is_full_width:
                overlap_x = max(0.0, min(ax1, bx1) - max(ax0, bx0))
                min_w = max(1.0, min(awidth, bwidth))
                if overlap_x / min_w >= 0.45:
                    if ay0 >= by1 - 2.0:
                        graph[i].append(j)
                        in_degree[j] += 1
                    elif by0 >= ay1 - 2.0:
                        graph[j].append(i)
                        in_degree[i] += 1
                elif ax1 <= bx0 + 2.0 and max(0.0, min(ay1, by1) - max(ay0, by0)) > 4.0:
                    # Column A is strictly left of Column B with vertical overlap
                    graph[i].append(j)
                    in_degree[j] += 1
                elif bx1 <= ax0 + 2.0 and max(0.0, min(ay1, by1) - max(ay0, by0)) > 4.0:
                    # Column B is strictly left of Column A with vertical overlap
                    graph[j].append(i)
                    in_degree[i] += 1

    # Kahn's algorithm with priority tie-breaker (highest Y top-down, then left-to-right)
    ready = [i for i in range(n) if in_degree[i] == 0]
    result: list[int] = []

    while ready:
        ready.sort(key=lambda idx: (-blocks[idx].bbox[3], blocks[idx].bbox[0]))
        curr = ready.pop(0)
        result.append(curr)
        for nxt in graph[curr]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                ready.append(nxt)

    if len(result) == n:
        return [blocks[i] for i in result]
    return blocks


def internal_interleave_columnar_blocks(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    """Interleave line fragments when a scanned table was split into columns."""
    candidates = [block for block in blocks if len(block.lines) >= 20]
    if len(candidates) < 3:
        return blocks
    x0s = [block.bbox[0] for block in candidates]
    x1s = [block.bbox[2] for block in candidates]
    if max(x0s) - min(x0s) > 20.0 or max(x1s) - min(x1s) > 30.0:
        return blocks
    merged_lines = tuple(line for block in candidates for line in block.lines)
    boxes = numpy.asarray(tuple(line.bbox for line in merged_lines), dtype=numpy.float32)
    ordered = internal_row_order_indexes(numpy.arange(len(merged_lines)), boxes)
    merged = ParsedBlock(
        lines=tuple(merged_lines[int(index)] for index in ordered),
        bbox=internal_block_bbox(merged_lines),
    )
    candidate_ids = {id(block) for block in candidates}
    return [merged, *(block for block in blocks if id(block) not in candidate_ids)]


def internal_column_major_prose(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    """Use column-major order for wide prose blocks from magazine-style pages."""
    output: list[ParsedBlock] = []
    for block in blocks:
        if len(block.lines) < 80:
            output.append(block)
            continue
        alphabetic = sum(character.isalpha() for line in block.lines for character in line.text)
        total = sum(character.isalnum() for line in block.lines for character in line.text)
        starts = sorted(line.bbox[0] for line in block.lines)
        clusters: list[float] = []
        for start in starts:
            if not clusters or start - clusters[-1] > 40.0:
                clusters.append(start)
        if len(clusters) < 3 or alphabetic / max(1, total) < 0.45:
            output.append(block)
            continue
        line_clusters = [
            min(range(len(clusters)), key=lambda index: abs(line.bbox[0] - clusters[index]))
            for line in block.lines
        ]
        transitions = sum(
            left != right for left, right in zip(line_clusters, line_clusters[1:], strict=False)
        )
        if transitions / max(1, len(line_clusters) - 1) < 0.25:
            output.append(block)
            continue
        # Stable column-major ordering from the nearest-column assignment.
        # Re-deriving membership with a fixed window emitted lines close to
        # two columns twice and silently dropped lines close to none.
        ordered = tuple(
            line
            for column_index in range(len(clusters))
            for line in sorted(
                (
                    line
                    for line, assigned in zip(block.lines, line_clusters, strict=True)
                    if assigned == column_index
                ),
                key=lambda line: -line.bbox[1],
            )
        )
        output.append(replace(block, lines=ordered))
    return output


def internal_transpose_numeric_table_blocks(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    """Transpose vectorized table columns into row-wise reading order."""
    output: list[ParsedBlock] = []
    for block in blocks:
        if len(block.lines) < 300:
            output.append(block)
            continue
        text = " ".join(line.text for line in block.lines)
        numeric = sum(character.isdigit() for character in text)
        alphanumeric = sum(character.isalnum() for character in text)
        starts = sorted(line.bbox[0] for line in block.lines)
        columns: list[float] = []
        for start in starts:
            if not columns or start - columns[-1] > 8.0:
                columns.append(start)
        if numeric / max(1, alphanumeric) < 0.25 or len(columns) < 20:
            output.append(block)
            continue
        ordered = tuple(sorted(block.lines, key=lambda line: (line.bbox[0], line.bbox[1])))
        output.append(replace(block, lines=ordered))
    return output


def order_lines(lines: tuple[ParsedLine, ...]) -> tuple[ParsedLine, ...]:
    if len(lines) < 2:
        return lines
    boxes = numpy.asarray(tuple(line.bbox for line in lines), dtype=numpy.float32)
    heights = numpy.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    regions = internal_xy_cut_regions(
        numpy.arange(len(lines), dtype=numpy.int64),
        boxes,
        (),
        max(1.0, finite_median(heights)),
    )
    return tuple(lines[int(index)] for region in regions for index in region)


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
        (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for box in values
        if (box[2] - box[0]) / span >= 0.70
    )
    regions = internal_xy_cut_regions(
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


def internal_has_repeated_block_columns(blocks: tuple[ParsedBlock, ...]) -> bool:
    """Identify pages whose blocks form a repeated multi-column grid."""
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
