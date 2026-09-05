# SPDX-License-Identifier: AGPL-3.0-only
"""Raster ruling detection and cell-level OCR task construction."""

from __future__ import annotations

from statistics import fmean

import numpy

from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.render.model import RasterImage
from core_pdf.impl.runtime.array_views import finite_median
from core_pdf_ocr.impl.extract.contracts import ObservationSource
from core_pdf_ocr.impl.extract.ocr.types import internal_OcrTask, internal_pixel_box_to_page_box

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
    return [int(round(fmean(cluster))) for cluster in clusters]


internal_GRID_STRIP_MIN_FRACTION = 0.5
internal_GRID_STRIP_GAP_PX = 12
internal_GRID_MAX_SKEW = 0.02
internal_GRID_DETECT_POOL = 3


def internal_estimate_ruling_skew(dark: numpy.ndarray) -> float:
    """Estimate page skew from horizontal rule offsets between page thirds."""
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
    return finite_median(numpy.asarray(offsets, dtype=numpy.float64)) / baseline


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
    """Find a ruled table grid; return edges, source samples, and measured skew."""
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
    """Build one single-line OCR task per populated ruled cell."""
    if (len(x_lines) - 1) * (len(y_lines) - 1) > internal_GRID_MAX_CELLS:
        return ()
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
    """Distinguish a data table's grid from a form's boxed fields."""
    x_lines, y_lines, _source_samples, slope = grid
    if len(y_lines) - 1 < internal_GRID_MIN_ROWS or len(x_lines) - 1 < internal_GRID_MIN_COLUMNS:
        return False
    heights = numpy.diff(numpy.asarray(y_lines, dtype=numpy.float64))
    median_height = finite_median(heights)
    if median_height <= 0.0:
        return False
    deviation = finite_median(numpy.abs(heights - median_height)) / median_height
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
    """Merge cell reads into one observation per grid row, left to right."""
    if not len(observations):
        return observations
    heights = observations.bbox[:, 3] - observations.bbox[:, 1]
    tolerance = max(2.0, finite_median(heights) * 0.6)
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
        confidences.append(float(numpy.mean(observations.confidence[ordered])))
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
