# SPDX-License-Identifier: AGPL-3.0-only
"""Tesseract integration: rasterization, recognition, and rescue passes."""

from __future__ import annotations

import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any

import numpy

from core_pdf.impl.engine.layout.spatial import (
    maximum_candidate_coverage,
)
from core_pdf.impl.engine.model.geometry import (
    overlap_ratio_min,
)
from core_pdf.impl.engine.parse.capture import internal_promoted_hidden_observations
from core_pdf.impl.engine.parse.model import (
    HIDDEN_TEXT_VERIFY_MIN_CONFIDENCE,
    HIDDEN_TEXT_VERIFY_MIN_MATCHED_TOKENS,
    HIDDEN_TEXT_VERIFY_MIN_SPATIAL_OVERLAP,
    HIDDEN_TEXT_VERIFY_MIN_TOKEN_OVERLAP,
    HIDDEN_TEXT_VERIFY_PIXELS,
    MAX_OCR_PIXELS,
    MAX_OCR_RASTER_BYTES,
    OCR_PREFLIGHT_PIXELS,
    PRIMARY_OCR_PIXELS,
    PSM_SPARSE_TEXT,
    CapturedPage,
    ObservationBatch,
    ObservationSource,
    OcrPass,
    OcrPassScope,
    RecognitionResult,
    WorkPlan,
    internal_bbox_tuple,
    internal_Candidate,
    internal_candidate,
    internal_text_utility_stats,
)
from core_pdf.impl.engine.parse.ocr_model import (
    internal_OcrRegion,
    internal_OcrTask,
    internal_PackedStrokedTextRaster,
    internal_pixel_box_to_page_box,
    internal_Raster,
    internal_RasterRegion,
    internal_RecognitionTrace,
)
from core_pdf.impl.engine.parse.ocr_raster import (
    internal_adaptive_ocr_raster,
    internal_raster_text_signal,
    internal_rendered_page_raster,
    internal_safe_image_crop,
)
from core_pdf.impl.engine.parse.ocr_regions import (
    internal_adaptive_rescue_decision,
    internal_candidate_ocr_regions,
    internal_candidate_region_tasks,
    internal_direct_scan_allowed,
    internal_dominant_image_region,
    internal_estimated_text_height,
    internal_has_distributed_outline_text,
    internal_high_resolution_weak_region_tasks,
    internal_ocr_region_batch,
    internal_ocr_task_groups,
    internal_page_image_regions,
    internal_primary_text_is_sufficient,
    internal_tile_tasks,
    internal_weak_region_tasks,
)
from core_pdf.impl.engine.parse.ocr_stroked_vector import (
    internal_decode_stroked_vector_text,
    internal_full_stroked_vector_text_raster,
    internal_packed_stroked_vector_decode_gate,
    internal_recover_stroked_vector_text,
    internal_remap_stroked_vector_candidate,
    internal_stroked_vector_text_raster,
)
from core_pdf.impl.engine.parse.ocr_tesseract import (
    internal_normalized_ocr_token_key,
    internal_recognize_group,
    internal_recover_timed_out_tasks,
)
from core_pdf.impl.engine.render.display import (
    RenderOptions,
)
from core_pdf.impl.engine.render.page import compose_page
from core_pdf.impl.engine.render.raster_image import RasterImage
from core_pdf.impl.runtime.execution import TaskScope, WorkStage
from core_pdf.impl.text import search_key, text_tokens

internal_OCR_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class internal_HiddenTextVerification:
    hidden_tokens: int
    preview_tokens: int
    matched_tokens: int
    spatially_matched_tokens: int
    token_overlap: float
    spatial_overlap: float
    accepted: bool
    reason: str

    def as_record(self) -> dict[str, int | float | bool | str]:
        return {
            "hidden_tokens": self.hidden_tokens,
            "preview_tokens": self.preview_tokens,
            "matched_tokens": self.matched_tokens,
            "spatially_matched_tokens": self.spatially_matched_tokens,
            "token_overlap": self.token_overlap,
            "spatial_overlap": self.spatial_overlap,
            "accepted": self.accepted,
            "reason": self.reason,
        }


def internal_hidden_text_verification(
    hidden: ObservationBatch,
    preview: ObservationBatch,
) -> internal_HiddenTextVerification:
    """Compare a word-level raster preview with hidden text and its page geometry."""
    hidden_by_token: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for text, raw_box in zip(hidden.text, hidden.bbox, strict=True):
        box = internal_bbox_tuple(raw_box)
        for token in text_tokens(text):
            hidden_by_token[token].append(box)

    preview_entries = tuple(
        (token, internal_bbox_tuple(raw_box))
        for text, raw_box in zip(preview.text, preview.bbox, strict=True)
        for token in text_tokens(text)
    )
    used: dict[str, set[int]] = defaultdict(set)
    matched = 0
    spatially_matched = 0
    for token, preview_box in preview_entries:
        candidates = hidden_by_token.get(token, ())
        available = (
            (index, box) for index, box in enumerate(candidates) if index not in used[token]
        )
        preview_center_x = (preview_box[0] + preview_box[2]) * 0.5
        preview_center_y = (preview_box[1] + preview_box[3]) * 0.5
        closest = min(
            available,
            key=lambda item: (
                ((item[1][0] + item[1][2]) * 0.5 - preview_center_x) ** 2
                + ((item[1][1] + item[1][3]) * 0.5 - preview_center_y) ** 2
            ),
            default=None,
        )
        if closest is None:
            continue
        index, hidden_box = closest
        used[token].add(index)
        matched += 1
        hidden_center_x = (hidden_box[0] + hidden_box[2]) * 0.5
        hidden_center_y = (hidden_box[1] + hidden_box[3]) * 0.5
        x_tolerance = max(
            12.0,
            (preview_box[2] - preview_box[0]) * 1.5,
            (hidden_box[2] - hidden_box[0]) * 1.5,
        )
        y_tolerance = max(
            6.0,
            (preview_box[3] - preview_box[1]) * 1.5,
            (hidden_box[3] - hidden_box[1]) * 1.5,
        )
        spatially_matched += int(
            abs(preview_center_x - hidden_center_x) <= x_tolerance
            and abs(preview_center_y - hidden_center_y) <= y_tolerance
        )

    preview_tokens = len(preview_entries)
    hidden_tokens = sum(len(boxes) for boxes in hidden_by_token.values())
    token_overlap = matched / max(1, preview_tokens)
    spatial_overlap = spatially_matched / max(1, preview_tokens)
    if matched < HIDDEN_TEXT_VERIFY_MIN_MATCHED_TOKENS:
        accepted = False
        reason = "insufficient-matched-tokens"
    elif token_overlap < HIDDEN_TEXT_VERIFY_MIN_TOKEN_OVERLAP:
        accepted = False
        reason = "low-token-overlap"
    elif spatial_overlap < HIDDEN_TEXT_VERIFY_MIN_SPATIAL_OVERLAP:
        accepted = False
        reason = "low-spatial-overlap"
    else:
        accepted = True
        reason = "semantic-and-spatial-match"
    return internal_HiddenTextVerification(
        hidden_tokens=hidden_tokens,
        preview_tokens=preview_tokens,
        matched_tokens=matched,
        spatially_matched_tokens=spatially_matched,
        token_overlap=token_overlap,
        spatial_overlap=spatial_overlap,
        accepted=accepted,
        reason=reason,
    )


internal_GRID_DARK_THRESHOLD = 160
internal_GRID_LINE_MIN_FRACTION = 0.30
internal_GRID_LINE_CLUSTER_PX = 5
internal_GRID_MIN_LINES = 4
internal_GRID_MIN_SPAN_FRACTION = 0.25
internal_GRID_CELL_MIN_PX = 6
internal_GRID_CELL_INSET_PX = 2
internal_GRID_CELL_MIN_INK = 0.003
internal_GRID_MAX_CELLS = 1200
internal_GRID_MIN_CELLS = 12
internal_GRID_CELL_MIN_CONFIDENCE = 50.0
internal_PSM_SINGLE_LINE = 7


def internal_close_row_gaps(mask: numpy.ndarray, gap: int) -> numpy.ndarray:
    """Bridge horizontal gaps up to ``gap`` pixels inside each row.

    Scanned rulings drop out along their length, so a rule reads as many
    short runs. Closing (dilate then erode along the row) reconnects them
    without thickening genuine text into false rules.
    """
    if gap <= 0:
        return mask
    window = gap + 1
    padded = numpy.zeros((mask.shape[0], mask.shape[1] + 2 * window), dtype=numpy.int32)
    padded[:, window:-window] = mask
    sums = numpy.cumsum(padded, axis=1)
    dilated = (sums[:, 2 * window :] - sums[:, : -2 * window]) > 0
    padded2 = numpy.zeros_like(padded)
    padded2[:, window:-window] = dilated
    sums2 = numpy.cumsum(padded2, axis=1)
    return (sums2[:, 2 * window :] - sums2[:, : -2 * window]) >= (2 * window)


def internal_longest_true_runs(mask: numpy.ndarray) -> numpy.ndarray:
    """Return the longest consecutive True run per row of a boolean matrix."""
    height, width = mask.shape
    separated = numpy.zeros((height, width + 2), dtype=numpy.int8)
    separated[:, 1:-1] = mask
    flat = separated.ravel()
    deltas = numpy.diff(flat)
    starts = numpy.flatnonzero(deltas == 1)
    ends = numpy.flatnonzero(deltas == -1)
    longest = numpy.zeros(height, dtype=numpy.int64)
    if len(starts):
        numpy.maximum.at(longest, starts // (width + 2), ends - starts)
    return longest


def internal_cluster_line_positions(
    positions: numpy.ndarray,
    tolerance: int = internal_GRID_LINE_CLUSTER_PX,
) -> list[int]:
    clusters: list[list[int]] = []
    for position in positions.tolist():
        if clusters and position - clusters[-1][-1] <= tolerance:
            clusters[-1].append(position)
        else:
            clusters.append([position])
    return [int(round(sum(cluster) / len(cluster))) for cluster in clusters]


internal_GRID_STRIP_MIN_FRACTION = 0.5
internal_GRID_STRIP_GAP_PX = 12
internal_GRID_MAX_SKEW = 0.02
internal_GRID_DETECT_POOL = 3


def internal_estimate_ruling_skew(dark: numpy.ndarray) -> float:
    """Estimate page skew from horizontal rule offsets between page thirds.

    A fraction of a degree of scanner skew walks a page-wide rule across a
    dozen pixel rows, which breaks projection-based detection outright. The
    rules themselves measure the skew: detect them separately in the left
    and right thirds, pair them up, and read the slope off the median
    vertical offset.
    """
    height, width = dark.shape
    strip = width // 3
    if strip < 50:
        return 0.0
    gap = internal_GRID_STRIP_GAP_PX
    left_runs = internal_longest_true_runs(internal_close_row_gaps(dark[:, :strip], gap))
    right_runs = internal_longest_true_runs(internal_close_row_gaps(dark[:, -strip:], gap))
    minimum = strip * internal_GRID_STRIP_MIN_FRACTION
    left_lines = internal_cluster_line_positions(numpy.flatnonzero(left_runs >= minimum))
    right_lines = internal_cluster_line_positions(numpy.flatnonzero(right_runs >= minimum))
    if len(left_lines) < 3 or len(right_lines) < 3:
        return 0.0
    baseline = width - strip
    offsets = []
    for line in left_lines:
        nearest = min(right_lines, key=lambda candidate: abs(candidate - line))
        if abs(nearest - line) <= baseline * internal_GRID_MAX_SKEW:
            offsets.append(nearest - line)
    if len(offsets) < 3:
        return 0.0
    return float(numpy.median(offsets)) / baseline


def internal_vertical_shear(dark: numpy.ndarray, slope: float) -> numpy.ndarray:
    """Shift each column vertically by ``-slope * x`` to straighten h-rules."""
    height, width = dark.shape
    shifts = numpy.round(slope * numpy.arange(width)).astype(numpy.int64)
    rows = numpy.arange(height)[:, None] + shifts[None, :]
    numpy.clip(rows, 0, height - 1, out=rows)
    return numpy.take_along_axis(dark, rows, axis=0)


def internal_detect_ruling_grid(
    image: RasterImage,
) -> tuple[list[int], list[int], numpy.ndarray, float] | None:
    """Find a ruled table grid; return (x_edges, y_edges, source samples, skew).

    A ruling is a near-full-span dark run: project the longest dark run per
    row (and per column) after closing scan dropouts and straightening the
    measured skew, and keep positions exceeding a fraction of the page span,
    clustered so a thick or doubled rule counts once. Ruled ledgers and data
    books produce a dozen or more of each; ordinary prose and underline
    styled forms do not reach the minimum in both axes. Edges are reported
    in the sheared (straightened) frame; callers map cells back through the
    returned skew.
    """
    array = numpy.asarray(image.array())
    if array.ndim == 3 and array.shape[2] >= 3:
        color = array[:, :, :3]
    elif array.ndim == 3 and array.shape[2] == 1:
        color = array[:, :, 0]
    elif array.ndim == 2:
        color = array
    else:
        return None
    height, width = color.shape[:2]
    if height < 100 or width < 100:
        return None
    # Detection runs on every recognized page, so work on a 3x max-pooled
    # mask: rules survive any-pooling and the +-3px edge error disappears
    # into the cell insets, while the shears and run scans cost a ninth.
    pool = internal_GRID_DETECT_POOL
    pooled_height = height // pool
    pooled_width = width // pool
    cropped = color[: pooled_height * pool, : pooled_width * pool]
    if cropped.ndim == 3:
        pooled = (
            cropped.reshape(pooled_height, pool, pooled_width, pool, 3).min(axis=(1, 3, 4))
            < internal_GRID_DARK_THRESHOLD
        )
    else:
        pooled = (
            cropped.reshape(pooled_height, pool, pooled_width, pool).min(axis=(1, 3))
            < internal_GRID_DARK_THRESHOLD
        )
    pooled_height, pooled_width = pooled.shape
    slope = internal_estimate_ruling_skew(pooled)
    straight = internal_vertical_shear(pooled, slope) if slope else pooled
    straight_columns = internal_vertical_shear(pooled.T, -slope) if slope else pooled.T
    gap = max(1, internal_GRID_STRIP_GAP_PX // pool)
    row_runs = internal_longest_true_runs(internal_close_row_gaps(straight, gap))
    column_runs = internal_longest_true_runs(internal_close_row_gaps(straight_columns, gap))
    y_lines = internal_cluster_line_positions(
        numpy.flatnonzero(row_runs >= pooled_width * internal_GRID_LINE_MIN_FRACTION),
        tolerance=max(1, internal_GRID_LINE_CLUSTER_PX // pool),
    )
    x_lines = internal_cluster_line_positions(
        numpy.flatnonzero(column_runs >= pooled_height * internal_GRID_LINE_MIN_FRACTION),
        tolerance=max(1, internal_GRID_LINE_CLUSTER_PX // pool),
    )
    if len(y_lines) < internal_GRID_MIN_LINES or len(x_lines) < internal_GRID_MIN_LINES:
        return None
    if (
        y_lines[-1] - y_lines[0] < pooled_height * internal_GRID_MIN_SPAN_FRACTION
        or x_lines[-1] - x_lines[0] < pooled_width * internal_GRID_MIN_SPAN_FRACTION
    ):
        return None
    scaled_x = [line * pool + pool // 2 for line in x_lines]
    scaled_y = [line * pool + pool // 2 for line in y_lines]
    return scaled_x, scaled_y, color, slope


def internal_grid_cell_tasks(
    task: internal_OcrTask,
    x_lines: list[int],
    y_lines: list[int],
    source_samples: numpy.ndarray,
    slope: float,
) -> tuple[internal_OcrTask, ...]:
    """Build one single-line OCR task per populated ruled cell.

    Tesseract's own segmentation merges and splits ruled cells; giving it one
    cell at a time replaces its layout analysis with the grid geometry. Grid
    edges arrive in the straightened frame, so each cell shifts back through
    the measured skew at its own centre -- the residual rotation across one
    cell is under a pixel. Cell pixel boxes stay in full-image coordinates so
    the shared page-box mapping applies unchanged, and near-blank cells are
    skipped by ink density instead of being recognized into noise.
    """
    if (len(x_lines) - 1) * (len(y_lines) - 1) > internal_GRID_MAX_CELLS:
        return ()
    # Line positions mark rule centres, and a scanned rule is several pixels
    # thick: an inset smaller than the rule leaves box fragments inside every
    # crop, which single-line recognition reads as bars or rejects outright.
    inset = max(internal_GRID_CELL_INSET_PX, int(round(task.resolution / 40)))
    height, width = source_samples.shape[:2]
    tasks: list[internal_OcrTask] = []
    for row_start, row_end in zip(y_lines, y_lines[1:]):
        if row_end - row_start < internal_GRID_CELL_MIN_PX + 2 * inset:
            continue
        for column_start, column_end in zip(x_lines, x_lines[1:]):
            if column_end - column_start < internal_GRID_CELL_MIN_PX + 2 * inset:
                continue
            center_x = (column_start + column_end) * 0.5
            center_y = (row_start + row_end) * 0.5
            row_shift = int(round(slope * center_x))
            column_shift = -int(round(slope * center_y))
            top = max(0, row_start + inset + row_shift)
            bottom = min(height, row_end - inset + row_shift)
            left = max(0, column_start + inset + column_shift)
            right = min(width, column_end - inset + column_shift)
            if bottom - top < internal_GRID_CELL_MIN_PX or right - left < (
                internal_GRID_CELL_MIN_PX
            ):
                continue
            cell = source_samples[top:bottom, left:right]
            if cell.ndim == 3:
                ink_ratio = float(
                    numpy.count_nonzero(cell.min(axis=2) < internal_GRID_DARK_THRESHOLD)
                ) / (cell.shape[0] * cell.shape[1])
            else:
                ink_ratio = float(numpy.count_nonzero(cell < internal_GRID_DARK_THRESHOLD)) / (
                    cell.shape[0] * cell.shape[1]
                )
            if ink_ratio < internal_GRID_CELL_MIN_INK:
                continue
            tasks.append(
                internal_OcrTask(
                    mode=internal_PSM_SINGLE_LINE,
                    image=task.image,
                    rectangle=(left, top, right - left, bottom - top),
                    page_box=task.page_box,
                    resolution=task.resolution,
                    minimum_confidence=internal_GRID_CELL_MIN_CONFIDENCE,
                )
            )
    return tuple(tasks)


def internal_grid_region_page_box(
    task: internal_OcrTask,
    x_lines: list[int],
    y_lines: list[int],
) -> tuple[float, float, float, float]:
    return internal_pixel_box_to_page_box(
        (x_lines[0], y_lines[0], x_lines[-1], y_lines[-1]),
        task.image.width,
        task.image.height,
        task.page_box,
    )


internal_GRID_MIN_ROWS = 8
internal_GRID_MIN_COLUMNS = 5
internal_GRID_MAX_ROW_HEIGHT_DEVIATION = 0.4
internal_GRID_MAX_STRADDLE_RATIO = 0.25


def internal_grid_is_regular_table(
    grid: tuple[list[int], list[int], numpy.ndarray, float],
    prior: ObservationBatch,
    task: internal_OcrTask,
) -> bool:
    """Distinguish a data table's grid from a form's boxed fields.

    Boxed forms clear the line-count bar as easily as data tables, but
    slicing a form into cells chops label text that spans boxes. A data
    table has many rows of near-uniform height and its words respect the
    column rules; a form has neither, and the words the page pass already
    read betray it by straddling the vertical lines.
    """
    x_lines, y_lines, _source_samples, slope = grid
    if len(y_lines) - 1 < internal_GRID_MIN_ROWS or len(x_lines) - 1 < internal_GRID_MIN_COLUMNS:
        return False
    heights = numpy.diff(numpy.asarray(y_lines, dtype=numpy.float64))
    median_height = float(numpy.median(heights))
    if median_height <= 0.0:
        return False
    deviation = float(numpy.median(numpy.abs(heights - median_height))) / median_height
    if deviation > internal_GRID_MAX_ROW_HEIGHT_DEVIATION:
        return False
    if not len(prior):
        return True
    page_x0, _page_y0, page_x1, page_y1 = task.page_box
    page_width = page_x1 - page_x0
    scale = task.image.width / max(1e-6, page_width)
    interior = x_lines[1:-1]
    if not interior:
        return True
    grid_box = internal_grid_region_page_box(task, x_lines, y_lines)
    inside = 0
    straddling = 0
    slack = max(2.0, task.image.width * 0.002)
    for box in prior.bbox:
        center_x = float(box[0] + box[2]) * 0.5
        center_y = float(box[1] + box[3]) * 0.5
        if not (grid_box[0] <= center_x <= grid_box[2] and grid_box[1] <= center_y <= grid_box[3]):
            continue
        inside += 1
        # Straighten the word into the sheared frame the lines live in.
        pixel_y = (page_y1 - center_y) * scale
        pixel_x0 = (float(box[0]) - page_x0) * scale + slope * pixel_y
        pixel_x1 = (float(box[2]) - page_x0) * scale + slope * pixel_y
        if any(pixel_x0 < line - slack and pixel_x1 > line + slack for line in interior):
            straddling += 1
    if inside < 8:
        return True
    return straddling <= inside * internal_GRID_MAX_STRADDLE_RATIO


def internal_grid_row_observations(
    observations: ObservationBatch,
) -> ObservationBatch:
    """Merge cell reads into one observation per grid row, left to right.

    Cell observations are isolated boxes, and the column gutters between them
    invite the layout cut to emit the table column-major. The reference
    reading order for a data table is row-major, so pre-assemble each row
    into a single line observation the layout keeps whole.
    """
    if not len(observations):
        return observations
    heights = observations.bbox[:, 3] - observations.bbox[:, 1]
    tolerance = max(2.0, float(numpy.median(heights)) * 0.6)
    order = numpy.argsort(-(observations.bbox[:, 1] + observations.bbox[:, 3]) * 0.5)
    rows: list[list[int]] = []
    row_center = 0.0
    for index in order.tolist():
        center = float(observations.bbox[index, 1] + observations.bbox[index, 3]) * 0.5
        if rows and abs(row_center - center) <= tolerance:
            rows[-1].append(index)
        else:
            rows.append([index])
            row_center = center
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    confidences: list[float] = []
    for row in rows:
        ordered = sorted(row, key=lambda index: float(observations.bbox[index, 0]))
        texts.append(" ".join(observations.text[index].strip() for index in ordered))
        row_boxes = observations.bbox[ordered]
        boxes.append(
            (
                float(row_boxes[:, 0].min()),
                float(row_boxes[:, 1].min()),
                float(row_boxes[:, 2].max()),
                float(row_boxes[:, 3].max()),
            )
        )
        confidences.append(
            float(numpy.mean([float(observations.confidence[index]) for index in ordered]))
        )
    return ObservationBatch.from_columns(
        texts,
        boxes,
        source=ObservationSource.OCR,
        confidence=confidences,
        sequence=range(len(texts)),
        rotation=(0 for _ in texts),
        font_size=(box[3] - box[1] for box in boxes),
        line_break_before=(True for _ in texts),
    )


def internal_candidate_text_containment(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> bool:
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if not shorter or sum(len(token) for token in shorter) < 4:
        return False
    width = len(shorter)
    return any(longer[index : index + width] == shorter for index in range(len(longer) - width + 1))


def internal_merge_candidate_batches(
    candidates: tuple[internal_Candidate, ...],
) -> internal_Candidate:
    if not candidates:
        return internal_candidate(-1, ObservationBatch.empty())
    if len(candidates) == 1:
        return candidates[0]
    modes = {candidate.mode for candidate in candidates}
    merged_by_mode: list[internal_Candidate] = []
    for mode in sorted(modes):
        mode_candidates = tuple(candidate for candidate in candidates if candidate.mode == mode)
        combined = ObservationBatch.concatenate(
            *(candidate.observations for candidate in mode_candidates)
        )
        combined_symbols = ObservationBatch.concatenate(
            *(candidate.symbols for candidate in mode_candidates)
        )
        fuzzy_tile_deduplication = len(mode_candidates) > 1
        order = numpy.lexsort((combined.bbox[:, 0], -combined.bbox[:, 1]))
        normalized_text = tuple(search_key(text) for text in combined.text)
        normalized_tokens = tuple(
            tuple(
                key
                for match in internal_OCR_TOKEN.finditer(text)
                if (key := internal_normalized_ocr_token_key(match.group(0)))
            )
            for text in combined.text
        )
        observation_utility = numpy.fromiter(
            (
                internal_text_utility_stats(text, float(confidence)).utility
                for text, confidence in zip(
                    combined.text,
                    combined.confidence,
                    strict=True,
                )
            ),
            dtype=numpy.float32,
            count=len(combined),
        )
        deduplicated: list[int] = []
        for raw_index in order:
            index = int(raw_index)
            duplicate_index = next(
                (
                    accepted_position
                    for accepted_position in range(
                        max(0, len(deduplicated) - 24), len(deduplicated)
                    )
                    if (
                        overlap_ratio_min(
                            combined.bbox[index],
                            combined.bbox[deduplicated[accepted_position]],
                        )
                        >= (
                            0.35
                            if internal_candidate_text_containment(
                                normalized_tokens[deduplicated[accepted_position]],
                                normalized_tokens[index],
                            )
                            else (
                                0.45
                                if normalized_text[deduplicated[accepted_position]]
                                == normalized_text[index]
                                else (0.70 if fuzzy_tile_deduplication else math.inf)
                            )
                        )
                    )
                ),
                None,
            )
            if duplicate_index is None:
                deduplicated.append(index)
                continue
            accepted_index = deduplicated[duplicate_index]
            containment = internal_candidate_text_containment(
                normalized_tokens[accepted_index],
                normalized_tokens[index],
            )
            if containment and len(normalized_text[index]) != len(normalized_text[accepted_index]):
                if len(normalized_text[index]) > len(normalized_text[accepted_index]):
                    deduplicated[duplicate_index] = index
            elif observation_utility[index] > observation_utility[accepted_index]:
                deduplicated[duplicate_index] = index
        heights = tuple(
            candidate.metrics.median_text_height
            for candidate in mode_candidates
            if candidate.metrics.median_text_height > 0.0
        )
        merged_by_mode.append(
            internal_candidate(
                mode,
                combined.take(deduplicated),
                symbols=combined_symbols,
                median_text_height=float(numpy.median(heights)) if heights else 0.0,
            )
        )
    return max(merged_by_mode, key=lambda candidate: candidate.metrics.utility)


def internal_augment_candidate(
    primary: internal_Candidate,
    supplement: internal_Candidate,
    *,
    minimum_confidence: float,
) -> tuple[internal_Candidate, int]:
    """Add only high-quality supplement observations absent from the primary pass."""
    if not len(supplement.observations):
        return primary, 0
    observations = supplement.observations
    confidence = observations.confidence
    informative = numpy.fromiter(
        (sum(character.isalnum() for character in text) >= 1 for text in observations.text),
        dtype=numpy.bool_,
        count=len(observations),
    )
    useful = numpy.fromiter(
        (
            (
                internal_text_utility_stats(text, float(value)).utility >= 2.0
                or (len(text.strip()) == 1 and text.strip().isalnum() and float(value) >= 85.0)
            )
            for text, value in zip(observations.text, confidence, strict=True)
        ),
        dtype=numpy.bool_,
        count=len(observations),
    )
    coverage = maximum_candidate_coverage(
        observations.bbox,
        primary.observations.bbox,
    )
    additions = (
        (confidence >= max(70.0, minimum_confidence)) & informative & useful & (coverage < 0.30)
    )
    added = int(numpy.count_nonzero(additions))
    if not added:
        return primary, 0
    combined = ObservationBatch.concatenate_selected(
        primary.observations,
        observations,
        additions,
    )
    return internal_candidate(primary.mode, combined, symbols=primary.symbols), added


def internal_candidate_timing_record(
    candidates: tuple[internal_Candidate, ...],
) -> dict[str, object]:
    """Aggregate the per-candidate recognition timings shared by pass diagnostics."""
    return {
        "recognition_seconds": sum(candidate.recognition_seconds for candidate in candidates),
        "setup_seconds": sum(candidate.setup_seconds for candidate in candidates),
        "api_seconds": sum(candidate.api_seconds for candidate in candidates),
        "iterator_seconds": sum(candidate.iterator_seconds for candidate in candidates),
        "cleanup_seconds": sum(candidate.cleanup_seconds for candidate in candidates),
        "candidate_seconds": sum(candidate.candidate_seconds for candidate in candidates),
        "recognition_statuses": tuple(candidate.recognition_status for candidate in candidates),
    }


def internal_record_candidates(
    candidates: tuple[tuple[str, internal_Candidate], ...],
    selected_name: str,
    trace: internal_RecognitionTrace,
) -> None:
    trace.candidates = tuple(
        {
            "name": name,
            "mode": candidate.mode,
            "selected": name == selected_name,
            **candidate.metrics.as_record(),
        }
        for name, candidate in candidates
    )
    if os.environ.get("CORE_PDF_CANDIDATE_ANALYSIS", "").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        trace.candidate_analysis = tuple(
            {
                "name": name,
                "mode": candidate.mode,
                "selected": name == selected_name,
                "text": "\n".join(candidate.observations.text),
                **candidate.metrics.as_record(),
            }
            for name, candidate in candidates
        )


# Small affine placement noise is cheaper to absorb in OCR coordinates than to
# recompose and rasterize the entire page around an otherwise usable source image.
OCR_IMAGE_REGIONS_MAX_AXIS_DEVIATION = 0.01


def recognize_page(
    capture: CapturedPage,
    plan: WorkPlan,
    context: TaskScope,
) -> RecognitionResult:
    trace = internal_RecognitionTrace.create()
    if not plan.ocr_passes:
        return RecognitionResult(ObservationBatch.empty(), trace.report())
    with context.reserve_raster(MAX_OCR_RASTER_BYTES):
        context.raise_if_cancelled()
        observations = internal_recognize_page_with_reserved_raster(
            capture,
            plan,
            context,
            trace=trace,
        )
    observations = internal_recover_stroked_vector_text(capture, observations, trace)
    return RecognitionResult(observations, trace.report())


def internal_recognize_page_with_reserved_raster(
    capture: CapturedPage,
    plan: WorkPlan,
    context: TaskScope,
    *,
    trace: internal_RecognitionTrace | None = None,
) -> ObservationBatch:
    trace = trace or internal_RecognitionTrace.create()
    page = capture.page
    page_box = (0.0, 0.0, float(page.width), float(page.height))
    compact_image: bool | str = True
    if capture.evidence.full_page_image:
        image_filters = capture.evidence.image_filters
        if any("JPX" in str(filter_name).upper() for filter_name in image_filters):
            compact_image = "grayscale"
    dominant_regions: dict[int, internal_RasterRegion | None] = {}
    rendered_rasters: dict[tuple[float, int, bool], internal_Raster | None] = {}
    rendered_page: Any | None = None
    candidate_regions: tuple[internal_OcrRegion, ...] | None = None
    candidates: list[tuple[str, internal_Candidate]] = []
    pass_diagnostics = trace.passes
    selected_name = ""
    selected: internal_Candidate | None = None
    selected_tasks: tuple[internal_OcrTask, ...] = ()
    previous_region_additions = 0
    seeded_region_selected = False
    adaptive_rescue_used = False

    def recognize_batch(tasks: tuple[internal_OcrTask, ...]) -> tuple[internal_Candidate, ...]:
        groups = internal_ocr_task_groups(tasks)
        results = context.map_ordered(internal_recognize_group, groups, stage=WorkStage.OCR)
        return tuple(candidate for group in results for candidate in group)

    def recognize_tasks(tasks: tuple[internal_OcrTask, ...]) -> tuple[internal_Candidate, ...]:
        candidates = recognize_batch(tasks)
        if any(candidate.recognition_status == "timeout" for candidate in candidates):
            context.raise_if_cancelled()
            candidates = internal_recover_timed_out_tasks(tasks, candidates, recognize_batch)
        return candidates

    def dominant_image_region_cached(pixel_budget: int) -> internal_RasterRegion | None:
        if pixel_budget not in dominant_regions:
            dominant_regions[pixel_budget] = internal_dominant_image_region(
                capture,
                max_pixels=pixel_budget,
            )
        return dominant_regions[pixel_budget]

    def rendered_raster_cached(ocr_pass: OcrPass) -> internal_Raster | None:
        raster_key = (ocr_pass.scale, ocr_pass.pixel_budget, ocr_pass.include_native_text)
        if raster_key not in rendered_rasters:
            rendered_rasters[raster_key] = internal_rendered_page_raster(
                capture,
                ocr_pass.scale,
                max_pixels=ocr_pass.pixel_budget,
                include_native_text=ocr_pass.include_native_text,
                trace=trace,
            )
        return rendered_rasters[raster_key]

    if plan.verify_hidden_text:
        context.raise_if_cancelled()
        started = time.perf_counter()
        verification_pass = OcrPass(
            "hidden-text-verification",
            OcrPassScope.PAGE,
            1.0,
            (PSM_SPARSE_TEXT,),
            minimum_confidence=HIDDEN_TEXT_VERIFY_MIN_CONFIDENCE,
            pixel_budget=HIDDEN_TEXT_VERIFY_PIXELS,
            recognize_words=True,
            region_first=False,
        )
        verification_region = internal_dominant_image_region(
            capture,
            max_pixels=HIDDEN_TEXT_VERIFY_PIXELS,
        )
        verification_tasks = (
            internal_tile_tasks(
                verification_region.raster,
                verification_region.page_box,
                verification_pass,
                compact_image=compact_image,
            )
            if verification_region is not None
            else ()
        )
        verification_candidates = recognize_tasks(verification_tasks)
        verification_candidate = internal_merge_candidate_batches(verification_candidates)
        verification = internal_hidden_text_verification(
            capture.observations,
            verification_candidate.observations,
        )
        raster_pixels = (
            verification_region.raster.width * verification_region.raster.height
            if verification_region is not None
            else 0
        )
        verification_record: dict[str, object] = {
            "name": verification_pass.name,
            "scope": verification_pass.scope.value,
            "scale": verification_pass.scale,
            "modes": verification_pass.modes,
            "recognize_words": verification_pass.recognize_words,
            "character_confidence_threshold": None,
            "task_count": len(verification_tasks),
            "raster_pixels": raster_pixels,
            "region_stage": "dominant-image-preview",
            "region_boxes": (
                (verification_region.page_box,) if verification_region is not None else ()
            ),
            "full_page_fallback": False,
            "elapsed_seconds": time.perf_counter() - started,
            "render_timings": trace.render_timings or {},
            **internal_candidate_timing_record(verification_candidates),
            "accepted_additions": 0,
            "adaptive_retry_scale": None,
            "adaptive_preflight": None,
            "adaptive_rescue_decision": None,
            "adaptive_rescue": None,
            "pixel_budget": verification_pass.pixel_budget,
            "rectangles": tuple(task.rectangle for task in verification_tasks),
            "selected": verification.accepted,
            **verification_candidate.metrics.as_record(),
            **verification.as_record(),
        }
        pass_diagnostics.append(verification_record)
        trace.hidden_text_verification = {
            "raster_pixels": raster_pixels,
            **verification.as_record(),
        }
        if verification.accepted:
            return internal_promoted_hidden_observations(capture)

    for ocr_pass in plan.ocr_passes:
        if (
            selected is not None
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.run_if_characters_below is not None
            and internal_primary_text_is_sufficient(selected)
        ):
            continue
        if (
            selected is not None
            and ocr_pass.run_if_characters_below is not None
            and selected.metrics.characters >= ocr_pass.run_if_characters_below
        ):
            continue
        if (
            selected is not None
            and ocr_pass.scope is OcrPassScope.IMAGE_REGIONS
            and ocr_pass.run_if_characters_below is not None
            and selected.metrics.characters >= 28
            and selected.metrics.mean_confidence >= 97.0
        ):
            continue
        if (
            selected is not None
            and ocr_pass.scope is OcrPassScope.IMAGE_REGIONS
            and ocr_pass.run_if_characters_below is not None
            and selected.metrics.characters >= 1500
            and selected.metrics.mean_confidence >= 98.0
        ):
            continue
        if (
            ocr_pass.run_if_additions_below is not None
            and previous_region_additions >= ocr_pass.run_if_additions_below
        ):
            continue
        if (
            ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.run_if_additions_below is not None
            and previous_region_additions == 0
            and selected is None
            and capture.evidence.visible_native_characters >= 3_000
        ):
            continue
        if (
            ocr_pass.scope is OcrPassScope.WEAK_REGIONS
            and ocr_pass.run_if_additions_below is not None
            and previous_region_additions == 0
            and selected is not None
            and selected.metrics.characters >= 32
            and selected.metrics.mean_confidence >= 90.0
        ):
            continue
        if (
            ocr_pass.scope is OcrPassScope.PAGE
            and seeded_region_selected
            and ocr_pass.run_if_additions_below is not None
        ):
            selected = None
            selected_name = ""
            selected_tasks = ()
            seeded_region_selected = False
        context.raise_if_cancelled()
        started = time.perf_counter()
        adaptive_preflight: dict[str, object] | None = None
        vector_preview = bool(
            capture.evidence.image_count == 0
            and capture.evidence.vector_complexity >= 100_000
            and capture.evidence.text_coverage < 0.05
        )
        if (
            ocr_pass.adaptive_scale
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.pixel_budget == PRIMARY_OCR_PIXELS
            and (capture.evidence.full_page_image or vector_preview)
        ):
            preview_raster: internal_Raster | None = None
            if capture.evidence.full_page_image:
                if OCR_PREFLIGHT_PIXELS not in dominant_regions:
                    # The preview only measures text height, so enlarging it would
                    # cost time and shift the projection this decision depends on.
                    dominant_regions[OCR_PREFLIGHT_PIXELS] = internal_dominant_image_region(
                        capture,
                        max_pixels=OCR_PREFLIGHT_PIXELS,
                        upscale=False,
                    )
                preview_region = dominant_regions[OCR_PREFLIGHT_PIXELS]
                preview_raster = preview_region.raster if preview_region is not None else None
            else:
                if rendered_page is None:
                    rendered_page = compose_page(
                        capture.page,
                        RenderOptions(include_text=ocr_pass.include_native_text),
                        page_program=capture.program,
                    )
                preview_raster = internal_rendered_page_raster(
                    capture,
                    ocr_pass.scale,
                    rendered=rendered_page,
                    cache=True,
                    max_pixels=OCR_PREFLIGHT_PIXELS,
                    include_native_text=ocr_pass.include_native_text,
                    trace=trace,
                )
            if preview_raster is not None:
                preview_height = internal_estimated_text_height(preview_raster)
                projected_height = preview_height * math.sqrt(
                    ocr_pass.pixel_budget / max(1, preview_raster.width * preview_raster.height)
                )
                projected_limit = 22.0 if vector_preview else 20.0
                if 12.0 <= projected_height < projected_limit:
                    original_scale = ocr_pass.scale
                    ocr_pass = replace(
                        ocr_pass,
                        scale=min(
                            8.0,
                            max(
                                original_scale + 0.5,
                                original_scale * 32.0 / projected_height,
                            ),
                        ),
                        pixel_budget=MAX_OCR_PIXELS,
                    )
                    adaptive_preflight = {
                        "preview_pixels": preview_raster.width * preview_raster.height,
                        "preview_text_height": preview_height,
                        "projected_primary_text_height": projected_height,
                        "selected_scale": ocr_pass.scale,
                        "source": "vector-render" if vector_preview else "dominant-image",
                    }
        tasks: tuple[internal_OcrTask, ...]
        packed_stroked: internal_PackedStrokedTextRaster | None = None
        raster_pixels = 0
        skipped_raster_pixels = 0
        image_text_preflight: tuple[dict[str, object], ...] = ()
        skipped_region_boxes: tuple[tuple[float, float, float, float], ...] = ()
        region_stage = "page"
        region_boxes: tuple[tuple[float, float, float, float], ...] = ()
        if (
            ocr_pass.region_first
            and ocr_pass.scope in {OcrPassScope.PAGE, OcrPassScope.WEAK_REGIONS}
            and (
                ocr_pass.scope is not OcrPassScope.WEAK_REGIONS
                or selected is not None
                or ocr_pass.seed_with_native
            )
        ):
            if candidate_regions is None:
                candidate_regions = internal_candidate_ocr_regions(capture)
            distributed_outline_text = bool(
                ocr_pass.scope is OcrPassScope.PAGE
                and internal_has_distributed_outline_text(capture)
            )
            region_batch = (
                (
                    internal_OcrRegion(
                        page_box,
                        float("inf"),
                        ("distributed-outline-text",),
                    ),
                )
                if distributed_outline_text
                else internal_ocr_region_batch(
                    candidate_regions,
                    ocr_pass,
                    page_area=max(1.0, float(page.width) * float(page.height)),
                )
            )
            tasks, raster_pixels, rendered_page, region_boxes = internal_candidate_region_tasks(
                capture,
                region_batch,
                ocr_pass,
                rendered=rendered_page,
                compact_image=compact_image,
                trace=trace,
            )
            region_stage = (
                "distributed-outline-page" if distributed_outline_text else "initial-regions"
            )
            if len(region_batch) == 1 and "page-fallback" in region_batch[0].reasons:
                region_stage = "page"
        elif ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            if selected is None and not ocr_pass.seed_with_native:
                continue
            if selected is not None and selected_tasks:
                tasks, raster_pixels, rendered_page, region_boxes = (
                    internal_high_resolution_weak_region_tasks(
                        capture,
                        selected_tasks,
                        ocr_pass,
                        selected.observations,
                        rendered=rendered_page,
                        compact_image=compact_image,
                        trace=trace,
                    )
                )
                region_stage = "weak-region-crops"
            else:
                direct_region = dominant_image_region_cached(ocr_pass.pixel_budget)
                raster = direct_region.raster if direct_region is not None else None
                raster_page_box = direct_region.page_box if direct_region is not None else page_box
                if raster is None:
                    raster = rendered_raster_cached(ocr_pass)
                    raster_page_box = page_box
                tasks = (
                    internal_weak_region_tasks(
                        raster,
                        raster_page_box,
                        ocr_pass,
                        selected.observations if selected is not None else capture.observations,
                        compact_image=compact_image,
                    )
                    if raster is not None
                    else ()
                )
                raster_pixels = (
                    sum(task.rectangle[2] * task.rectangle[3] for task in tasks)
                    if raster is not None
                    else 0
                )
        elif ocr_pass.scope is OcrPassScope.STROKED_VECTOR_TEXT:
            packed_stroked = internal_stroked_vector_text_raster(
                capture,
                ocr_pass.scale,
                max_pixels=ocr_pass.pixel_budget,
                trace=trace,
            )
            if packed_stroked is not None:
                region_stage = "packed-stroked-vector-text"
                region_boxes = (
                    (capture.evidence.stroked_vector_text.bbox,)
                    if capture.evidence.stroked_vector_text.bbox is not None
                    else ()
                )
                tasks = internal_tile_tasks(
                    packed_stroked.raster,
                    packed_stroked.packed_box,
                    replace(ocr_pass, recognize_words=True, collect_symbols=True),
                    compact_image=compact_image,
                )
                raster_pixels = packed_stroked.raster.width * packed_stroked.raster.height
            else:
                fallback_region = internal_full_stroked_vector_text_raster(
                    capture,
                    ocr_pass.scale,
                    max_pixels=ocr_pass.pixel_budget,
                    trace=trace,
                )
                region_stage = "stroked-vector-text-fallback"
                region_boxes = (fallback_region.page_box,) if fallback_region is not None else ()
                tasks = (
                    internal_tile_tasks(
                        fallback_region.raster,
                        fallback_region.page_box,
                        ocr_pass,
                        compact_image=compact_image,
                    )
                    if fallback_region is not None
                    else ()
                )
                raster_pixels = (
                    fallback_region.raster.width * fallback_region.raster.height
                    if fallback_region is not None
                    else 0
                )
                trace.stroked_vector_packed = {
                    "accepted": False,
                    "cells": 0,
                    "raster_pixels": 0,
                    "unmapped_observations": 0,
                    "fallback_used": bool(tasks),
                }
        elif ocr_pass.scope is OcrPassScope.IMAGE_REGIONS:
            regions = internal_page_image_regions(
                capture,
                minimum_area_ratio=0.02,
                max_pixels=ocr_pass.pixel_budget,
                maximum_axis_deviation=OCR_IMAGE_REGIONS_MAX_AXIS_DEVIATION,
            )
            if regions:
                region_signals = tuple(
                    (region, internal_raster_text_signal(region.raster.image)) for region in regions
                )
                image_text_preflight = tuple(
                    {
                        "page_box": region.page_box,
                        "raster_pixels": region.raster.width * region.raster.height,
                        **signal.as_record(),
                    }
                    for region, signal in region_signals
                )
                eligible_regions = tuple(
                    region
                    for region, signal in region_signals
                    if signal.likely_text
                    or (
                        signal.horizontal_edge_ratio >= 0.035
                        and sum(len(t.strip()) for t in capture.observations.text) < 15
                    )
                )
                skipped_regions = tuple(
                    region for region, signal in region_signals if not signal.likely_text
                )
                skipped_raster_pixels = sum(
                    region.raster.width * region.raster.height for region in skipped_regions
                )
                skipped_region_boxes = tuple(region.page_box for region in skipped_regions)
                region_boxes = tuple(region.page_box for region in eligible_regions)
                region_stage = "direct-image-regions"
                tasks = tuple(
                    task
                    for region in eligible_regions
                    for task in internal_tile_tasks(
                        region.raster,
                        region.page_box,
                        ocr_pass,
                        compact_image=compact_image,
                    )
                )
                raster_pixels = sum(
                    region.raster.width * region.raster.height for region in eligible_regions
                )
            else:
                fallback_scale = max(2.0, ocr_pass.scale)
                image_crop = internal_safe_image_crop(capture)
                raster = internal_rendered_page_raster(
                    capture,
                    fallback_scale,
                    crop=image_crop,
                    max_pixels=ocr_pass.pixel_budget,
                    include_native_text=ocr_pass.include_native_text,
                    trace=trace,
                )
                raster_page_box = image_crop or page_box
                tasks = (
                    internal_tile_tasks(
                        raster,
                        raster_page_box,
                        ocr_pass,
                        compact_image=compact_image,
                    )
                    if raster is not None
                    else ()
                )
                raster_pixels = raster.width * raster.height if raster is not None else 0
        else:
            direct_region = (
                dominant_image_region_cached(ocr_pass.pixel_budget)
                if internal_direct_scan_allowed(capture, plan)
                else None
            )
            raster = direct_region.raster if direct_region is not None else None
            raster_page_box = direct_region.page_box if direct_region is not None else page_box
            if raster is None:
                raster = rendered_raster_cached(ocr_pass)
                raster_page_box = page_box
            task_raster = (
                internal_adaptive_ocr_raster(raster)
                if raster is not None and ocr_pass.name == "adaptive-page"
                else raster
            )
            tasks = (
                internal_tile_tasks(
                    task_raster,
                    raster_page_box,
                    ocr_pass,
                    compact_image=compact_image,
                )
                if task_raster is not None
                else ()
            )
            raster_pixels = raster.width * raster.height if raster is not None else 0
        if not tasks:
            if not image_text_preflight:
                continue
            region_stage = "image-text-preflight"

        candidate_source_tasks = tasks
        task_candidates = recognize_tasks(tasks)
        if packed_stroked is not None:
            remapped_with_counts = tuple(
                internal_remap_stroked_vector_candidate(candidate, packed_stroked)
                for candidate in task_candidates
            )
            task_candidates = tuple(item[0] for item in remapped_with_counts)
            unmapped_observations = sum(item[1] for item in remapped_with_counts)
            packed_candidate = internal_merge_candidate_batches(task_candidates)
            decode_started = time.perf_counter()
            packed_decode = internal_decode_stroked_vector_text(
                capture,
                packed_candidate.observations,
                packed_candidate.symbols,
            )
            decode_seconds = time.perf_counter() - decode_started
            packed_accepted, packed_gate = internal_packed_stroked_vector_decode_gate(
                packed_decode,
                len(packed_stroked.cells),
            )
            packed_pixels = raster_pixels
            fallback_used = False
            if packed_accepted:
                # Seed packing only rasterizes multi-glyph runs, so isolated
                # glyphs (pin numbers, lone digits) are never shown to OCR when
                # the packed decode gate passes. Recognize them from their own
                # high-scale montage as a supplement.
                isolated_packed = internal_stroked_vector_text_raster(
                    capture,
                    ocr_pass.scale,
                    max_pixels=ocr_pass.pixel_budget,
                    variant="isolated",
                    trace=trace,
                )
                isolated_tasks = (
                    internal_tile_tasks(
                        isolated_packed.raster,
                        isolated_packed.packed_box,
                        replace(
                            ocr_pass,
                            recognize_words=True,
                            collect_symbols=True,
                            minimum_confidence=50.0,
                        ),
                        compact_image=compact_image,
                    )
                    if isolated_packed is not None
                    else ()
                )
                if isolated_tasks and isolated_packed is not None:
                    isolated_remapped = tuple(
                        internal_remap_stroked_vector_candidate(
                            candidate,
                            isolated_packed,
                            digit_bearing_only=True,
                        )
                        for candidate in recognize_tasks(isolated_tasks)
                    )
                    isolated_candidates = tuple(item[0] for item in isolated_remapped)
                    packed_gate["isolated_cells"] = len(isolated_packed.cells)
                    packed_gate["isolated_observations"] = sum(
                        len(item[0].observations) for item in isolated_remapped
                    )
                    task_candidates = (*task_candidates, *isolated_candidates)
                    candidate_source_tasks = (*candidate_source_tasks, *isolated_tasks)
                    tasks = (*tasks, *isolated_tasks)
                    packed_candidate = internal_merge_candidate_batches(task_candidates)
                    raster_pixels += isolated_packed.raster.width * isolated_packed.raster.height
                trace.pending_stroked_decode = (
                    id(packed_candidate.observations),
                    packed_decode,
                    decode_seconds,
                )
            else:
                fallback_region = internal_full_stroked_vector_text_raster(
                    capture,
                    ocr_pass.scale,
                    max_pixels=ocr_pass.pixel_budget,
                    trace=trace,
                )
                fallback_tasks = (
                    internal_tile_tasks(
                        fallback_region.raster,
                        fallback_region.page_box,
                        replace(ocr_pass, recognize_words=False),
                        compact_image=compact_image,
                    )
                    if fallback_region is not None
                    else ()
                )
                if fallback_tasks:
                    fallback_used = True
                    fallback_candidates = recognize_tasks(fallback_tasks)
                    task_candidates = (*task_candidates, *fallback_candidates)
                    candidate_source_tasks = (*candidate_source_tasks, *fallback_tasks)
                    tasks = (*tasks, *fallback_tasks)
                    packed_candidate = internal_merge_candidate_batches(fallback_candidates)
                    raster_pixels += (
                        fallback_region.raster.width * fallback_region.raster.height
                        if fallback_region is not None
                        else 0
                    )
                    region_stage = "stroked-vector-text-fallback"
                    region_boxes = (
                        (fallback_region.page_box,) if fallback_region is not None else region_boxes
                    )
            trace.stroked_vector_packed = {
                **packed_gate,
                "raster_pixels": packed_pixels,
                "unmapped_observations": unmapped_observations,
                "symbol_observations": len(packed_candidate.symbols),
                "fallback_used": fallback_used,
            }
            candidate = packed_candidate
        else:
            candidate = internal_merge_candidate_batches(task_candidates)
        if (
            selected is not None
            and plan.augment_page_candidates
            and ocr_pass.scope is OcrPassScope.PAGE
            and not capture.evidence.vector_complexity >= 180
        ):
            candidate, _ = internal_augment_candidate(
                selected,
                candidate,
                minimum_confidence=70.0,
            )
        adaptive_retry_scale: float | None = None
        adaptive_rescue: dict[str, object] | None = None
        adaptive_rescue_decision: dict[str, object] | None = None
        median_height = candidate.metrics.median_text_height
        rescue_eligible = bool(
            ocr_pass.adaptive_scale
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.pixel_budget < MAX_OCR_PIXELS
            and not adaptive_rescue_used
            and candidate.metrics.characters >= ocr_pass.minimum_characters_for_rescue
            and (candidate.metrics.characters < 32 or 0.0 < median_height < 24.0)
        )
        run_rescue = False
        if rescue_eligible:
            adaptive_rescue_used = True
            run_rescue, adaptive_rescue_decision = internal_adaptive_rescue_decision(
                candidate,
                candidate_source_tasks,
                ocr_pass,
            )
        if run_rescue:
            factor = 1.5 if median_height <= 0.0 else min(2.5, max(1.25, 32.0 / median_height))
            adaptive_retry_scale = min(8.0, max(ocr_pass.scale + 0.5, ocr_pass.scale * factor))
            retry_pass = replace(
                ocr_pass,
                name="adaptive-rescue",
                scale=adaptive_retry_scale,
                pixel_budget=MAX_OCR_PIXELS,
                region_first=False,
            )
            retry_scope = (
                "page"
                if candidate.metrics.characters < 32 or median_height < 18.0
                else "weak-regions"
            )
            retry_boxes: tuple[tuple[float, float, float, float], ...] = ()
            if retry_scope == "page":
                retry_raster = internal_rendered_page_raster(
                    capture,
                    adaptive_retry_scale,
                    max_pixels=MAX_OCR_PIXELS,
                    include_native_text=ocr_pass.include_native_text,
                    trace=trace,
                )
                retry_tasks = (
                    internal_tile_tasks(
                        retry_raster,
                        page_box,
                        retry_pass,
                        compact_image=compact_image,
                    )
                    if retry_raster is not None
                    else ()
                )
                rescue_pixels = (
                    retry_raster.width * retry_raster.height if retry_raster is not None else 0
                )
            else:
                retry_pass = replace(
                    retry_pass,
                    scope=OcrPassScope.WEAK_REGIONS,
                    tiles=max(6, retry_pass.tiles),
                    region_columns=max(3, retry_pass.region_columns),
                    max_regions=max(8, retry_pass.max_regions),
                )
                retry_tasks, rescue_pixels, rendered_page, retry_boxes = (
                    internal_high_resolution_weak_region_tasks(
                        capture,
                        tasks,
                        retry_pass,
                        candidate.observations,
                        rendered=rendered_page,
                        compact_image=compact_image,
                        trace=trace,
                    )
                )
            if retry_tasks:
                candidate_source_tasks = (*candidate_source_tasks, *retry_tasks)
                retry_candidates = recognize_tasks(retry_tasks)
                retry_candidate = internal_merge_candidate_batches(retry_candidates)
                augmented_candidate, rescue_additions = internal_augment_candidate(
                    candidate,
                    retry_candidate,
                    minimum_confidence=ocr_pass.minimum_confidence,
                )
                if retry_candidate.metrics.utility > augmented_candidate.metrics.utility * 1.05:
                    candidate = retry_candidate
                elif augmented_candidate.metrics.utility > candidate.metrics.utility:
                    candidate = augmented_candidate
                task_candidates = (*task_candidates, *retry_candidates)
                raster_pixels += rescue_pixels
                adaptive_rescue = {
                    "scope": retry_scope,
                    "scale": adaptive_retry_scale,
                    "raster_pixels": rescue_pixels,
                    "task_count": len(retry_tasks),
                    "accepted_additions": rescue_additions,
                    "region_boxes": retry_boxes,
                }
        additions = 0
        if ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            used_native_seed = selected is None
            if selected is not None:
                candidate, additions = internal_augment_candidate(
                    selected,
                    candidate,
                    minimum_confidence=ocr_pass.minimum_confidence,
                )
            else:
                additions = len(candidate.observations)
        candidates.append((ocr_pass.name, candidate))
        elapsed = time.perf_counter() - started
        pass_diagnostics.append(
            {
                "name": ocr_pass.name,
                "scope": ocr_pass.scope.value,
                "scale": ocr_pass.scale,
                "modes": ocr_pass.modes,
                "recognize_words": any(task.recognize_words for task in tasks),
                "character_confidence_threshold": ocr_pass.character_confidence_threshold,
                "task_count": len(tasks),
                "raster_pixels": raster_pixels,
                "skipped_raster_pixels": skipped_raster_pixels,
                "image_text_preflight": image_text_preflight,
                "region_stage": region_stage,
                "region_boxes": region_boxes,
                "skipped_region_boxes": skipped_region_boxes,
                "full_page_fallback": (
                    region_stage == "page" and ocr_pass.scope is OcrPassScope.PAGE
                ),
                "elapsed_seconds": elapsed,
                "render_timings": trace.render_timings or {},
                **internal_candidate_timing_record(task_candidates),
                "accepted_additions": additions,
                "adaptive_retry_scale": adaptive_retry_scale,
                "adaptive_preflight": adaptive_preflight,
                "adaptive_rescue_decision": adaptive_rescue_decision,
                "adaptive_rescue": adaptive_rescue,
                "pixel_budget": ocr_pass.pixel_budget,
                "rectangles": tuple(task.rectangle for task in tasks),
                "selected": False,
                **candidate.metrics.as_record(),
            }
        )
        if not tasks:
            continue
        if ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            previous_region_additions = additions
            if additions:
                selected_name = ocr_pass.name
                selected = candidate
                selected_tasks = (*selected_tasks, *candidate_source_tasks)
                seeded_region_selected = used_native_seed and ocr_pass.seed_with_native
            continue
        if selected is None or candidate.metrics.utility > (
            selected.metrics.utility * ocr_pass.minimum_utility_gain
        ):
            selected_name = ocr_pass.name
            selected = candidate
            selected_tasks = candidate_source_tasks

    if selected is None:
        internal_record_candidates(tuple(candidates), selected_name, trace)
        return ObservationBatch.empty()
    for diagnostic in pass_diagnostics:
        diagnostic["selected"] = diagnostic["name"] == selected_name
    internal_record_candidates(tuple(candidates), selected_name, trace)
    if selected_tasks:
        # Ruled scanned tables defeat Tesseract's page segmentation; when the
        # page raster shows a full ruling grid, re-recognize cell by cell and
        # let the grid text replace the page-segmented text inside the grid.
        source_task = max(
            selected_tasks,
            key=lambda task: task.rectangle[2] * task.rectangle[3],
        )
        grid = internal_detect_ruling_grid(source_task.image)
        if grid is not None and internal_grid_is_regular_table(
            grid, selected.observations, source_task
        ):
            x_lines, y_lines, source_samples, slope = grid
            cell_tasks = internal_grid_cell_tasks(
                source_task, x_lines, y_lines, source_samples, slope
            )
            if len(cell_tasks) >= internal_GRID_MIN_CELLS:
                cell_candidate = internal_merge_candidate_batches(recognize_tasks(cell_tasks))
                cell_observations = internal_grid_row_observations(cell_candidate.observations)
                if len(cell_observations):
                    grid_box = internal_grid_region_page_box(source_task, x_lines, y_lines)
                    prior = selected.observations
                    centers_x = (prior.bbox[:, 0] + prior.bbox[:, 2]) * 0.5
                    centers_y = (prior.bbox[:, 1] + prior.bbox[:, 3]) * 0.5
                    outside = ~(
                        (centers_x >= grid_box[0])
                        & (centers_x <= grid_box[2])
                        & (centers_y >= grid_box[1])
                        & (centers_y <= grid_box[3])
                    )
                    replaced_alnum = sum(
                        sum(character.isalnum() for character in prior.text[index])
                        for index in numpy.flatnonzero(~outside)
                    )
                    cell_alnum = sum(
                        sum(character.isalnum() for character in text)
                        for text in cell_observations.text
                    )
                    if cell_alnum < replaced_alnum * 0.8:
                        # The page-segmented reads carried more content than
                        # the cell reads; this grid's cells recognize worse
                        # than whole-page OCR, so keep the original.
                        return selected.observations
                    retained = prior.take(numpy.flatnonzero(outside))
                    trace.grid_cell_ocr = {
                        "cells": len(cell_tasks),
                        "cell_observations": len(cell_observations),
                        "replaced_observations": int(numpy.count_nonzero(~outside)),
                        "grid_box": grid_box,
                        "columns": len(x_lines) - 1,
                        "rows": len(y_lines) - 1,
                    }
                    return ObservationBatch.concatenate(retained, cell_observations)
    return selected.observations
