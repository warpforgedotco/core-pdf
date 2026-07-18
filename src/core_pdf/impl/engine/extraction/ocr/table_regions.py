# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol

from core_layout.impl.layout.models import TableGrid
from core_ocr.impl.candidates import OcrCandidate
from core_ocr.impl.text_analysis import (
    extracted_text_token_count,
    normalized_text_tokens,
    numeric_token_ratio,
    text_ocr_quality_score,
)
from core_ocr.impl.types import (
    TESSERACT_RIL_TEXTLINE,
    TESSERACT_RIL_WORD,
    OcrComponentBox,
    OcrImage,
    OcrTextResult,
)

from core_pdf.impl.engine.extraction.cache import ExtractionCache
from core_pdf.impl.engine.extraction.common import page_geometry
from core_pdf.impl.engine.extraction.ocr import (
    candidates as ocr_candidates,
)
from core_pdf.impl.engine.extraction.ocr import (
    execution as ocr_execution,
)
from core_pdf.impl.engine.extraction.ocr import (
    tiling as ocr_tiling,
)
from core_pdf.impl.engine.extraction.tables.grid import detect_grid, merge_grids
from core_pdf.impl.engine.spec.s_07_content.capture import CapturedLine
from core_pdf.impl.engine.spec.s_07_document.page_boxes import rotate_page_lines


class TableOcrPage(Protocol):
    extraction_cache: ExtractionCache | None
    media_box: tuple[float, float, float, float] | None
    rotation: int
    chars: list[Any]

    def get_grid_lines(self) -> list[Any]: ...


OCR_TABLE_DEFAULT_PAGE_SEGMENTATION_MODE = 6
OCR_TABLE_CELL_TRACTABLE_REGION_COUNT = 48
OCR_TABLE_CELL_TRACTABLE_EXTENDED_REGION_COUNT = 96
OCR_TABLE_CELL_TRACTABLE_MEDIAN_AREA = 2_500
OCR_TABLE_VERTICAL_TRACTABLE_REGION_COUNT = 64
OCR_TABLE_TEXTLINE_REFINEMENT_MIN_HEIGHT_RATIO = 0.16
OCR_TABLE_TEXTLINE_REFINEMENT_MIN_TABLE_HEIGHT_RATIO = 0.18
OCR_TABLE_TEXTLINE_REFINEMENT_COARSE_ROW_COUNT = 12
OCR_TABLE_TEXTLINE_REFINEMENT_COARSE_ROW_HEIGHT_RATIO = 0.09
OCR_TABLE_TEXTLINE_REFINEMENT_MAX_ROWS = 96
OCR_TABLE_COLUMN_INFERENCE_MIN_ROWS = 3
OCR_TABLE_COLUMN_INFERENCE_MIN_WORD_BOXES = 12
OCR_TABLE_COLUMN_INFERENCE_MAX_EXISTING_SPANS = 4
OCR_TABLE_COLUMN_INFERENCE_MAX_SPANS = 16
OCR_TABLE_ROW_PROFILE_RETRY_CONFIDENCE = 55
OCR_TABLE_ROW_PROFILE_MAX_RETRY_ROWS = 12
OCR_TABLE_ROW_PROFILE_BROAD_WEAK_MIN_ROWS = 8
OCR_TABLE_ROW_PROFILE_BROAD_WEAK_FRACTION = 0.65
OCR_RASTER_TABLE_DARK_THRESHOLD = 170
OCR_RASTER_TABLE_MIN_LINES_PER_AXIS = 2
OCR_RASTER_TABLE_MIN_LINE_LENGTH_RATIO = 0.08
OCR_RASTER_TABLE_LINE_DENSITY_RATIO = 0.22
OCR_RASTER_TABLE_LINE_RUN_RATIO = 0.06
OCR_RASTER_TABLE_MAX_SCAN_STRIDE = 3
OCR_RASTER_TABLE_MAX_LINE_GAP_PIXELS = 3
OCR_TABLE_CONSENSUS_TRACTABLE_REGION_COUNT = 128
OCR_TESSERACT_TABLE_VARIABLES = {
    "preserve_interword_spaces": "1",
    "textord_tablefind_recognize_tables": "1",
    "textord_tabfind_find_tables": "1",
}
OCR_TESSERACT_TABLE_ROW_PROFILES = (
    (
        "line_no_dawg",
        7,
        {
            **OCR_TESSERACT_TABLE_VARIABLES,
            "load_freq_dawg": "0",
            "load_system_dawg": "0",
        },
    ),
    (
        "adaptive_otsu",
        OCR_TABLE_DEFAULT_PAGE_SEGMENTATION_MODE,
        {**OCR_TESSERACT_TABLE_VARIABLES, "thresholding_method": "1"},
    ),
    (
        "sauvola",
        OCR_TABLE_DEFAULT_PAGE_SEGMENTATION_MODE,
        {
            **OCR_TESSERACT_TABLE_VARIABLES,
            "thresholding_method": "2",
            "thresholding_window_size": "0.33",
            "thresholding_kfactor": "0.34",
        },
    ),
)
OCR_TABLE_CANDIDATE_NAMES = frozenset(
    ("table_rows", "table_cells", "table_cells_rotated", "table_cell_consensus")
)


@dataclass(frozen=True)
class RasterTableLine:
    orientation: str
    coord: float
    start: float
    end: float
    thickness: float
    slope: float = 0.0


@dataclass(frozen=True)
class TableOcrRegion:
    x0: int
    y0: int
    x1: int
    y1: int
    row_index: int
    col_index: int | None
    kind: str
    rotate_vertical: bool = False

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def rectangle(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


def table_ocr_line_anchor_tokens(tokens: list[str]) -> set[str]:
    return {token for token in tokens if any(ch.isdigit() for ch in token) or len(token) >= 4}


def table_ocr_line_has_table_signal(line: str, tokens: list[str]) -> bool:
    if "$" in line:
        return True
    digit_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    if digit_tokens >= 2:
        return True
    return len(tokens) >= 7 and digit_tokens >= 1


def collect_table_rectangle_ocr_candidates(
    page: TableOcrPage,
    image: OcrImage,
    timeout: float | None,
) -> list[OcrCandidate]:
    if not image.source.startswith("rendered_page_") or "_tile_" in image.source:
        return []
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return []
    grid = table_grid_for_ocr(page, image=image)
    if grid is None:
        return []
    page_width, page_height = page_display_dimensions_for_ocr(page)
    if page_width <= 0.0 or page_height <= 0.0:
        return []
    row_regions, cell_regions = table_ocr_regions_from_grid(
        grid,
        image,
        page_width=page_width,
        page_height=page_height,
    )
    row_regions, cell_regions = refine_table_ocr_regions_with_textlines(
        image,
        row_regions,
        cell_regions,
        timeout,
    )
    candidates: list[OcrCandidate] = []
    row_candidate = table_row_ocr_candidate(image, row_regions, timeout)
    if row_candidate is not None:
        candidates.append(row_candidate)
    cell_candidate = table_cell_ocr_candidate(
        image,
        cell_regions,
        timeout,
        use_rotated_vertical=False,
    )
    if cell_candidate is not None:
        candidates.append(cell_candidate)
    rotated_cell_candidate = table_cell_ocr_candidate(
        image,
        cell_regions,
        timeout,
        use_rotated_vertical=True,
    )
    if rotated_cell_candidate is not None:
        candidates.append(rotated_cell_candidate)
    consensus_candidate = table_cell_consensus_candidate(
        page,
        image,
        cell_regions,
        page_width=page_width,
        page_height=page_height,
        timeout=timeout,
    )
    if consensus_candidate is not None:
        candidates.append(consensus_candidate)
    return candidates


def table_grid_for_ocr(page: TableOcrPage, image: OcrImage | None = None) -> TableGrid | None:
    cache = page.extraction_cache
    vector_cache_key = "ocr_table_vector_grid"
    if cache is not None and vector_cache_key in cache:
        vector_grid = cache.get_as(vector_cache_key, TableGrid)
    else:
        vector_grid = vector_table_grid_for_ocr(page)
        if cache is not None:
            cache[vector_cache_key] = vector_grid
    if image is None:
        return vector_grid
    page_width, page_height = page_display_dimensions_for_ocr(page)
    if page_width <= 0.0 or page_height <= 0.0:
        return vector_grid
    raster_cache_key = (
        "ocr_table_raster_grid",
        image.width,
        image.height,
        image.bytes_per_pixel,
        image.bytes_per_line,
        image.resolution,
    )
    if cache is not None and raster_cache_key in cache:
        raster_grid = cache.get_as(raster_cache_key, TableGrid)
    else:
        raster_grid = raster_table_grid_for_ocr(
            image,
            page_width=page_width,
            page_height=page_height,
        )
        if cache is not None:
            cache[raster_cache_key] = raster_grid
    return select_table_grid_for_ocr(vector_grid, raster_grid)


def vector_table_grid_for_ocr(page: TableOcrPage) -> TableGrid | None:
    try:
        lines = page.get_grid_lines()
    except Exception:
        lines = []
    if not lines:
        grid = None
    else:
        page_space = page_geometry.PageSpace.from_page(page)
        if page_space is None:
            page_width = page_height = 0.0
        else:
            page_width = page_space.width
            page_height = page_space.height
        rotate = getattr(page, "rotation", 0) % 360
        if rotate and page_width > 0.0 and page_height > 0.0:
            lines = rotate_page_lines(
                lines,
                rotate=rotate,
                page_width=page_width,
                page_height=page_height,
            )
        grid = detect_grid(lines, line_tolerance=8.0)
    return grid


def select_table_grid_for_ocr(
    vector_grid: TableGrid | None,
    raster_grid: TableGrid | None,
) -> TableGrid | None:
    if vector_grid is None:
        return raster_grid
    if raster_grid is None:
        return vector_grid
    try:
        merged = merge_grids(vector_grid, raster_grid)
    except Exception:
        merged = vector_grid
    vector_cells = max(0, len(vector_grid.rows) - 1) * max(0, len(vector_grid.cols) - 1)
    raster_cells = max(0, len(raster_grid.rows) - 1) * max(0, len(raster_grid.cols) - 1)
    merged_cells = max(0, len(merged.rows) - 1) * max(0, len(merged.cols) - 1)
    if raster_cells >= max(vector_cells + 2, int(vector_cells * 1.25)):
        return merged if merged_cells >= raster_cells else raster_grid
    return merged if merged_cells <= max(vector_cells + raster_cells, 1) else vector_grid


def raster_table_grid_for_ocr(
    image: OcrImage,
    *,
    page_width: float,
    page_height: float,
) -> TableGrid | None:
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return None
    if image.width <= 0 or image.height <= 0:
        return None
    lines = raster_table_lines_from_image(
        image,
        page_width=page_width,
        page_height=page_height,
    )
    if len(lines) < 4:
        return None
    tolerance = max(1.5, min(page_width, page_height) * 0.004)
    grid = detect_grid(lines, line_tolerance=tolerance)
    if grid is None or not raster_grid_has_table_shape(grid):
        return None
    return grid


def raster_table_lines_from_image(
    image: OcrImage,
    *,
    page_width: float,
    page_height: float,
) -> list[CapturedLine]:
    horizontal_slope = raster_estimated_ruling_slope(image, "h")
    vertical_slope = raster_estimated_ruling_slope(image, "v")
    raster_lines = [
        *raster_projection_line_peaks(image, "h", horizontal_slope),
        *raster_projection_line_peaks(image, "v", vertical_slope),
    ]
    lines = [
        captured_line_from_raster_line(
            raster_line,
            image,
            page_width=page_width,
            page_height=page_height,
        )
        for raster_line in raster_lines
    ]
    return [line for line in lines if line is not None]


def raster_estimated_ruling_slope(image: OcrImage, orientation: str) -> float:
    if image.width <= 0 or image.height <= 0:
        return 0.0
    sample_step = max(4, min(12, max(image.width, image.height) // 300))
    points: list[tuple[int, int]] = []
    for y in range(0, image.height, sample_step):
        for x in range(0, image.width, sample_step):
            if raster_pixel_is_dark(image, x, y):
                points.append((x, y))
    if len(points) < 100:
        return 0.0
    center_x = image.width * 0.5
    center_y = image.height * 0.5
    best_slope = 0.0
    best_score = 0
    baseline_score = 0
    for slope in (0.0, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04):
        buckets: defaultdict[int, int] = defaultdict(int)
        for x, y in points:
            if orientation == "h":
                coord = int(round(y - slope * (x - center_x)))
            else:
                coord = int(round(x - slope * (y - center_y)))
            buckets[coord] += 1
        score = max(buckets.values()) if buckets else 0
        if slope == 0.0:
            baseline_score = score
        if score > best_score:
            best_slope = slope
            best_score = score
    if baseline_score <= 0 or best_score < baseline_score * 1.08:
        return 0.0
    return best_slope


def raster_projection_line_peaks(
    image: OcrImage,
    orientation: str,
    slope: float,
) -> list[RasterTableLine]:
    axis_length = image.height if orientation == "h" else image.width
    cross_length = image.width if orientation == "h" else image.height
    if axis_length <= 0 or cross_length <= 0:
        return []
    stride = raster_scan_stride(cross_length)
    sample_count = max(1, (cross_length + stride - 1) // stride)
    min_dark_count = max(12, int(sample_count * OCR_RASTER_TABLE_LINE_DENSITY_RATIO))
    min_run = max(8, int(sample_count * OCR_RASTER_TABLE_LINE_RUN_RATIO))
    candidates: list[tuple[int, int]] = []
    for coord in range(axis_length):
        dark_count, max_run = raster_projection_stats(
            image,
            orientation,
            coord,
            stride,
            slope,
        )
        if dark_count >= min_dark_count and max_run >= min_run:
            candidates.append((coord, dark_count + max_run))
    groups = raster_group_projection_candidates(candidates)
    peaks: list[RasterTableLine] = []
    min_length = max(24, int(cross_length * OCR_RASTER_TABLE_MIN_LINE_LENGTH_RATIO))
    for start, end, center_coord, score in groups:
        extent = raster_line_extent(
            image,
            orientation,
            start,
            end,
            slope,
        )
        if extent is None:
            continue
        extent_start, extent_end = extent
        if extent_end - extent_start < min_length:
            continue
        peaks.append(
            RasterTableLine(
                orientation,
                coord=float(center_coord),
                start=float(extent_start),
                end=float(extent_end),
                thickness=float(end - start + 1),
                slope=slope,
            )
        )
        del score
    return peaks


def raster_scan_stride(length: int) -> int:
    if length <= 1200:
        return 1
    return min(OCR_RASTER_TABLE_MAX_SCAN_STRIDE, max(1, length // 1200))


def raster_projection_stats(
    image: OcrImage,
    orientation: str,
    coord: int,
    stride: int,
    slope: float,
) -> tuple[int, int]:
    dark_count = 0
    current_run = 0
    max_run = 0
    if orientation == "h":
        center_x = image.width * 0.5
        for x in range(0, image.width, stride):
            y = int(round(coord + slope * (x - center_x)))
            if 0 <= y < image.height and raster_pixel_is_dark(image, x, y):
                dark_count += 1
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
    else:
        center_y = image.height * 0.5
        for y in range(0, image.height, stride):
            x = int(round(coord + slope * (y - center_y)))
            if 0 <= x < image.width and raster_pixel_is_dark(image, x, y):
                dark_count += 1
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
    return (dark_count, max_run)


def raster_group_projection_candidates(
    candidates: list[tuple[int, int]],
) -> list[tuple[int, int, float, int]]:
    if not candidates:
        return []
    groups: list[list[tuple[int, int]]] = []
    for coord, score in candidates:
        if groups and coord - groups[-1][-1][0] <= OCR_RASTER_TABLE_MAX_LINE_GAP_PIXELS:
            groups[-1].append((coord, score))
        else:
            groups.append([(coord, score)])
    result: list[tuple[int, int, float, int]] = []
    for group in groups:
        total_score = sum(score for ignored_coord, score in group)
        if total_score <= 0:
            continue
        center_coord = sum(coord * score for coord, score in group) / total_score
        result.append((group[0][0], group[-1][0], center_coord, total_score))
    return result


def raster_line_extent(
    image: OcrImage,
    orientation: str,
    line_start: int,
    line_end: int,
    slope: float,
) -> tuple[int, int] | None:
    if orientation == "h":
        axis_length = image.width
        band_start = max(0, line_start - 1)
        band_end = min(image.height - 1, line_end + 1)
    else:
        axis_length = image.height
        band_start = max(0, line_start - 1)
        band_end = min(image.width - 1, line_end + 1)
    positions: list[int] = []
    min_band_support = max(1, int((band_end - band_start + 1) * 0.35))
    for pos in range(axis_length):
        support = 0
        if orientation == "h":
            center_x = image.width * 0.5
            shifted_center = slope * (pos - center_x)
            for y_coord in range(band_start, band_end + 1):
                y = int(round(y_coord + shifted_center))
                if 0 <= y < image.height and raster_pixel_is_dark(image, pos, y):
                    support += 1
        else:
            center_y = image.height * 0.5
            shifted_center = slope * (pos - center_y)
            for x_coord in range(band_start, band_end + 1):
                x = int(round(x_coord + shifted_center))
                if 0 <= x < image.width and raster_pixel_is_dark(image, x, pos):
                    support += 1
        if support >= min_band_support:
            positions.append(pos)
    runs = raster_position_runs(positions)
    if not runs:
        return None
    return max(runs, key=lambda run: run[1] - run[0])


def raster_position_runs(positions: list[int]) -> list[tuple[int, int]]:
    if not positions:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position - previous <= OCR_RASTER_TABLE_MAX_LINE_GAP_PIXELS:
            previous = position
            continue
        runs.append((start, previous))
        start = previous = position
    runs.append((start, previous))
    return runs


def raster_pixel_is_dark(image: OcrImage, x: int, y: int) -> bool:
    if x < 0 or y < 0 or x >= image.width or y >= image.height:
        return False
    offset = y * image.bytes_per_line + x * image.bytes_per_pixel
    if offset < 0 or offset >= len(image.data):
        return False
    if image.bytes_per_pixel == 1:
        return image.data[offset] <= OCR_RASTER_TABLE_DARK_THRESHOLD
    if offset + 2 >= len(image.data):
        return False
    if image.bytes_per_pixel == 4 and offset + 3 < len(image.data) and image.data[offset + 3] <= 32:
        return False
    red = image.data[offset]
    green = image.data[offset + 1]
    blue = image.data[offset + 2]
    luminance = (red * 77 + green * 150 + blue * 29) >> 8
    return luminance <= OCR_RASTER_TABLE_DARK_THRESHOLD


def captured_line_from_raster_line(
    line: RasterTableLine,
    image: OcrImage,
    *,
    page_width: float,
    page_height: float,
) -> CapturedLine | None:
    geometry = table_image_geometry(
        image,
        page_width=page_width,
        page_height=page_height,
    )
    if geometry is None:
        return None
    if line.orientation == "h":
        center_x = image.width * 0.5
        y0 = line.coord + line.slope * (line.start - center_x)
        y1 = line.coord + line.slope * (line.end - center_x)
        segment = page_geometry.image_segment_to_page_segment(
            (float(line.start), y0),
            (float(line.end), y1),
            geometry,
        )
        thickness = page_geometry.image_axis_length_to_page_length(
            line.thickness,
            geometry,
            axis="y",
        )
        if segment is None or thickness is None:
            return None
        return CapturedLine(
            segment[0],
            segment[1],
            segment[2],
            segment[3],
            max(0.25, thickness),
        )
    center_y = image.height * 0.5
    x0 = line.coord + line.slope * (line.start - center_y)
    x1 = line.coord + line.slope * (line.end - center_y)
    segment = page_geometry.image_segment_to_page_segment(
        (x0, float(line.start)),
        (x1, float(line.end)),
        geometry,
    )
    thickness = page_geometry.image_axis_length_to_page_length(
        line.thickness,
        geometry,
        axis="x",
    )
    if segment is None or thickness is None:
        return None
    return CapturedLine(
        segment[0],
        segment[1],
        segment[2],
        segment[3],
        max(0.25, thickness),
    )


def raster_grid_has_table_shape(grid: TableGrid) -> bool:
    if len(grid.rows) < OCR_RASTER_TABLE_MIN_LINES_PER_AXIS:
        return False
    if len(grid.cols) < OCR_RASTER_TABLE_MIN_LINES_PER_AXIS:
        return False
    return max(0, len(grid.rows) - 1) * max(0, len(grid.cols) - 1) >= 1


def page_display_dimensions_for_ocr(page: TableOcrPage) -> tuple[float, float]:
    page_space = page_geometry.PageSpace.from_page(page)
    if page_space is None:
        return (0.0, 0.0)
    return (page_space.display_width, page_space.display_height)


def table_ocr_regions_from_grid(
    grid: Any,
    image: OcrImage,
    *,
    page_width: float,
    page_height: float,
) -> tuple[list[TableOcrRegion], list[TableOcrRegion]]:
    rows = list(getattr(grid, "rows", ()))
    cols = list(getattr(grid, "cols", ()))
    if len(rows) < 2 or len(cols) < 2:
        return ([], [])
    row_regions: list[TableOcrRegion] = []
    cell_regions: list[TableOcrRegion] = []
    for row_index in range(len(rows) - 1):
        y_top = float(rows[row_index])
        y_bottom = float(rows[row_index + 1])
        row_region = table_page_rect_to_ocr_region(
            cols[0],
            y_top,
            cols[-1],
            y_bottom,
            image,
            page_width=page_width,
            page_height=page_height,
            row_index=row_index,
            col_index=None,
            kind="row",
            padding_points=1.0,
        )
        if row_region is not None:
            row_regions.append(row_region)
        for col_index in range(len(cols) - 1):
            region = table_page_rect_to_ocr_region(
                cols[col_index],
                y_top,
                cols[col_index + 1],
                y_bottom,
                image,
                page_width=page_width,
                page_height=page_height,
                row_index=row_index,
                col_index=col_index,
                kind="cell",
                padding_points=0.75,
            )
            if region is not None:
                cell_regions.append(region)
    return (row_regions, cell_regions)


def refine_table_ocr_regions_with_textlines(
    image: OcrImage,
    row_regions: list[TableOcrRegion],
    cell_regions: list[TableOcrRegion],
    timeout: float | None,
) -> tuple[list[TableOcrRegion], list[TableOcrRegion]]:
    if not table_ocr_regions_need_textline_refinement(row_regions, image):
        return (row_regions, cell_regions)
    boxes = ocr_execution.ocr_component_boxes_with_timeout(
        image,
        TESSERACT_RIL_TEXTLINE,
        timeout,
        variables=OCR_TESSERACT_TABLE_VARIABLES,
    )
    refined_rows = table_textline_row_regions_from_boxes(row_regions, boxes, image)
    if len(refined_rows) <= len(row_regions):
        return (row_regions, cell_regions)
    word_boxes: list[OcrComponentBox] = []
    if table_ocr_regions_need_column_inference(
        refined_rows,
        cell_regions,
        image,
    ):
        word_boxes = ocr_execution.ocr_component_boxes_with_timeout(
            image,
            TESSERACT_RIL_WORD,
            timeout,
            variables=OCR_TESSERACT_TABLE_VARIABLES,
        )
    refined_cells = table_cell_regions_from_textline_rows(
        refined_rows,
        cell_regions,
        image,
        word_boxes=word_boxes,
    )
    return (refined_rows, refined_cells or cell_regions)


def table_ocr_regions_need_textline_refinement(
    row_regions: list[TableOcrRegion],
    image: OcrImage,
) -> bool:
    if not row_regions or image.height <= 0:
        return False
    max_height = max(region.height for region in row_regions)
    if max_height >= image.height * OCR_TABLE_TEXTLINE_REFINEMENT_MIN_HEIGHT_RATIO:
        return True
    if len(row_regions) <= 3 and max_height >= image.height * 0.10:
        return True
    table_bounds = table_region_bounds(row_regions)
    if table_bounds is None:
        return False
    table_height = table_bounds[3] - table_bounds[1]
    return bool(
        len(row_regions) <= OCR_TABLE_TEXTLINE_REFINEMENT_COARSE_ROW_COUNT
        and table_height >= image.height * OCR_TABLE_TEXTLINE_REFINEMENT_MIN_TABLE_HEIGHT_RATIO
        and max_height >= table_height * OCR_TABLE_TEXTLINE_REFINEMENT_COARSE_ROW_HEIGHT_RATIO
    )


def table_textline_row_regions_from_boxes(
    row_regions: list[TableOcrRegion],
    boxes: list[OcrComponentBox],
    image: OcrImage,
) -> list[TableOcrRegion]:
    table_bounds = table_region_bounds(row_regions)
    if table_bounds is None:
        return []
    table_left, table_top, table_right, table_bottom = table_bounds
    rects: list[tuple[int, int, int, int]] = []
    for box in boxes:
        rect = table_textline_box_rect(box, image, table_bounds)
        if rect is not None:
            rects.append(rect)
    bands = table_textline_bands(rects)
    if not bands or len(bands) > OCR_TABLE_TEXTLINE_REFINEMENT_MAX_ROWS:
        return []
    refined: list[TableOcrRegion] = []
    for row_index, (left, top, right, bottom) in enumerate(bands):
        del left, right
        padded_top = max(table_top, top - 3)
        padded_bottom = min(table_bottom, bottom + 3)
        if padded_bottom - padded_top < 4:
            continue
        refined.append(
            TableOcrRegion(
                table_left,
                padded_top,
                table_right,
                padded_bottom,
                row_index=row_index,
                col_index=None,
                kind="row",
                rotate_vertical=False,
            )
        )
    return refined


def table_region_bounds(
    regions: list[TableOcrRegion],
) -> tuple[int, int, int, int] | None:
    if not regions:
        return None
    left = min(region.x0 for region in regions)
    top = min(region.y0 for region in regions)
    right = max(region.x1 for region in regions)
    bottom = max(region.y1 for region in regions)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def table_textline_box_rect(
    box: OcrComponentBox,
    image: OcrImage,
    table_bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    if box.width < 6 or box.height < 4:
        return None
    left = max(0, min(image.width, int(box.left)))
    top = max(0, min(image.height, int(box.top)))
    right = max(left, min(image.width, int(box.left) + int(box.width)))
    bottom = max(top, min(image.height, int(box.top) + int(box.height)))
    if right - left < 6 or bottom - top < 4:
        return None
    table_left, table_top, table_right, table_bottom = table_bounds
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    if not (table_left <= center_x <= table_right and table_top <= center_y <= table_bottom):
        return None
    if bottom - top > max(24, int((table_bottom - table_top) * 0.18)):
        return None
    return (left, top, right, bottom)


def table_textline_bands(
    rects: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    if not rects:
        return []
    bands: list[tuple[int, int, int, int]] = []
    current: tuple[int, int, int, int] | None = None
    for rect in sorted(rects, key=lambda item: ((item[1] + item[3]) * 0.5, item[0])):
        if current is None:
            current = rect
            continue
        if table_textline_rects_share_band(current, rect):
            current = (
                min(current[0], rect[0]),
                min(current[1], rect[1]),
                max(current[2], rect[2]),
                max(current[3], rect[3]),
            )
        else:
            bands.append(current)
            current = rect
    if current is not None:
        bands.append(current)
    return bands


def table_textline_rects_share_band(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    first_center = (first[1] + first[3]) * 0.5
    second_center = (second[1] + second[3]) * 0.5
    first_height = max(1, first[3] - first[1])
    second_height = max(1, second[3] - second[1])
    center_delta = abs(first_center - second_center)
    if center_delta <= max(5.0, min(first_height, second_height) * 0.55):
        return True
    overlap = min(first[3], second[3]) - max(first[1], second[1])
    return overlap > 0 and overlap / max(1, min(first_height, second_height)) >= 0.35


def table_cell_regions_from_textline_rows(
    row_regions: list[TableOcrRegion],
    cell_regions: list[TableOcrRegion],
    image: OcrImage,
    *,
    word_boxes: list[OcrComponentBox] | None = None,
) -> list[TableOcrRegion]:
    existing_column_spans = table_column_spans_from_cell_regions(cell_regions, image)
    inferred_column_spans = table_column_spans_from_word_boxes(
        row_regions,
        word_boxes or [],
        image,
    )
    column_spans = select_table_column_spans(
        existing_column_spans,
        inferred_column_spans,
    )
    if not column_spans:
        return []
    return table_cell_regions_from_column_spans(row_regions, column_spans, image)


def table_cell_regions_from_column_spans(
    row_regions: list[TableOcrRegion],
    column_spans: list[tuple[int, int, int]],
    image: OcrImage,
) -> list[TableOcrRegion]:
    refined: list[TableOcrRegion] = []
    for row in row_regions:
        for col_index, left, right in column_spans:
            rectangle = ocr_tiling.clamp_ocr_bbox(
                left, row.y0, right, row.y1, image.width, image.height
            )
            if rectangle is None:
                continue
            x0, y0, x1, y1 = rectangle
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            refined.append(
                TableOcrRegion(
                    x0,
                    y0,
                    x1,
                    y1,
                    row_index=row.row_index,
                    col_index=col_index,
                    kind="cell",
                    rotate_vertical=table_region_looks_vertical(x0, y0, x1, y1),
                )
            )
    return refined


def table_ocr_regions_need_column_inference(
    row_regions: list[TableOcrRegion],
    cell_regions: list[TableOcrRegion],
    image: OcrImage,
) -> bool:
    if len(row_regions) < OCR_TABLE_COLUMN_INFERENCE_MIN_ROWS:
        return False
    existing_column_spans = table_column_spans_from_cell_regions(cell_regions, image)
    if not existing_column_spans:
        return False
    if len(existing_column_spans) > OCR_TABLE_COLUMN_INFERENCE_MAX_EXISTING_SPANS:
        return False
    table_bounds = table_region_bounds(row_regions)
    if table_bounds is None:
        return False
    table_width = table_bounds[2] - table_bounds[0]
    return table_width >= 80


def table_column_spans_from_cell_regions(
    cell_regions: list[TableOcrRegion],
    image: OcrImage,
) -> list[tuple[int, int, int]]:
    by_col: dict[int, list[TableOcrRegion]] = defaultdict(list)
    for region in cell_regions:
        if region.col_index is not None:
            by_col[region.col_index].append(region)
    spans: list[tuple[int, int, int]] = []
    for col_index, regions in sorted(by_col.items()):
        left = max(0, min(region.x0 for region in regions))
        right = min(image.width, max(region.x1 for region in regions))
        if right - left >= 4:
            spans.append((col_index, left, right))
    return spans


def select_table_column_spans(
    existing_spans: list[tuple[int, int, int]],
    inferred_spans: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    if not existing_spans:
        return inferred_spans
    if len(inferred_spans) <= len(existing_spans):
        return existing_spans
    if len(inferred_spans) > OCR_TABLE_COLUMN_INFERENCE_MAX_SPANS:
        return existing_spans
    return inferred_spans


def table_column_spans_from_word_boxes(
    row_regions: list[TableOcrRegion],
    word_boxes: list[OcrComponentBox],
    image: OcrImage,
) -> list[tuple[int, int, int]]:
    if (
        len(row_regions) < OCR_TABLE_COLUMN_INFERENCE_MIN_ROWS
        or len(word_boxes) < OCR_TABLE_COLUMN_INFERENCE_MIN_WORD_BOXES
    ):
        return []
    table_bounds = table_region_bounds(row_regions)
    if table_bounds is None:
        return []
    table_left, ignored_top, table_right, ignored_bottom = table_bounds
    del ignored_top, ignored_bottom
    table_width = table_right - table_left
    if table_width < 80:
        return []
    boundaries = table_column_boundaries_from_word_boxes(
        row_regions,
        word_boxes,
        image,
        table_bounds,
    )
    if not boundaries:
        return []
    coords = [float(table_left), *boundaries, float(table_right)]
    spans: list[tuple[int, int, int]] = []
    min_span_width = max(4, int(round(table_width * 0.015)))
    for col_index, (left, right) in enumerate(zip(coords, coords[1:], strict=False)):
        left_int = max(0, min(image.width, int(round(left))))
        right_int = max(left_int, min(image.width, int(round(right))))
        if right_int - left_int < min_span_width:
            continue
        spans.append((col_index, left_int, right_int))
    if len(spans) < 2:
        return []
    return spans


def table_column_boundaries_from_word_boxes(
    row_regions: list[TableOcrRegion],
    word_boxes: list[OcrComponentBox],
    image: OcrImage,
    table_bounds: tuple[int, int, int, int],
) -> list[float]:
    table_left, table_top, table_right, table_bottom = table_bounds
    table_width = table_right - table_left
    row_rects: defaultdict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for box in word_boxes:
        rect = table_word_box_rect(box, image, table_bounds)
        if rect is None:
            continue
        row_index = table_row_region_index_for_rect(row_regions, rect)
        if row_index is None:
            continue
        row_rects[row_index].append(rect)
    evidence: list[tuple[float, int, float]] = []
    for row_index, rects in row_rects.items():
        rects = table_normalized_word_rects(rects)
        if len(rects) < 2:
            continue
        for left_rect, right_rect in zip(rects, rects[1:], strict=False):
            gap = right_rect[0] - left_rect[2]
            if gap < table_column_boundary_min_gap(table_width, left_rect, right_rect):
                continue
            boundary = left_rect[2] + gap * 0.5
            edge_margin = max(8.0, table_width * 0.015)
            if boundary <= table_left + edge_margin:
                continue
            if boundary >= table_right - edge_margin:
                continue
            evidence.append((boundary, row_index, gap))
    return table_column_boundaries_from_evidence(evidence, len(row_regions), table_width)


def table_word_box_rect(
    box: OcrComponentBox,
    image: OcrImage,
    table_bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    if box.width < 3 or box.height < 3:
        return None
    left = max(0, min(image.width, int(box.left)))
    top = max(0, min(image.height, int(box.top)))
    right = max(left, min(image.width, int(box.left) + int(box.width)))
    bottom = max(top, min(image.height, int(box.top) + int(box.height)))
    if right - left < 3 or bottom - top < 3:
        return None
    table_left, table_top, table_right, table_bottom = table_bounds
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    if not (table_left <= center_x <= table_right and table_top <= center_y <= table_bottom):
        return None
    table_height = max(1, table_bottom - table_top)
    if bottom - top > max(48, int(table_height * 0.20)):
        return None
    return (left, top, right, bottom)


def table_row_region_index_for_rect(
    row_regions: list[TableOcrRegion],
    rect: tuple[int, int, int, int],
) -> int | None:
    _, top, _, bottom = rect
    center_y = (top + bottom) * 0.5
    best_index: int | None = None
    best_overlap = 0
    rect_height = max(1, bottom - top)
    for index, row in enumerate(row_regions):
        if row.y0 - 2 <= center_y <= row.y1 + 2:
            return index
        overlap = max(0, min(bottom, row.y1) - max(top, row.y0))
        if overlap > best_overlap:
            best_index = index
            best_overlap = overlap
    if best_overlap / rect_height >= 0.35:
        return best_index
    return None


def table_normalized_word_rects(
    rects: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    normalized: list[tuple[int, int, int, int]] = []
    for rect in sorted(rects, key=lambda item: (item[0], item[1], item[2], item[3])):
        if not normalized:
            normalized.append(rect)
            continue
        current = normalized[-1]
        if rect[0] <= current[2]:
            normalized[-1] = (
                min(current[0], rect[0]),
                min(current[1], rect[1]),
                max(current[2], rect[2]),
                max(current[3], rect[3]),
            )
        else:
            normalized.append(rect)
    return normalized


def table_column_boundary_min_gap(
    table_width: int,
    left_rect: tuple[int, int, int, int],
    right_rect: tuple[int, int, int, int],
) -> float:
    left_width = max(1, left_rect[2] - left_rect[0])
    right_width = max(1, right_rect[2] - right_rect[0])
    return max(10.0, table_width * 0.012, min(left_width, right_width) * 0.50)


def table_column_boundaries_from_evidence(
    evidence: list[tuple[float, int, float]],
    row_count: int,
    table_width: int,
) -> list[float]:
    if not evidence:
        return []
    tolerance = max(6.0, table_width * 0.008)
    groups: list[list[tuple[float, int, float]]] = []
    for item in sorted(evidence, key=lambda value: value[0]):
        if groups and abs(item[0] - table_boundary_group_center(groups[-1])) <= tolerance:
            groups[-1].append(item)
        else:
            groups.append([item])
    min_support = table_column_boundary_min_support(row_count)
    accepted: list[tuple[float, int, float]] = []
    for group in groups:
        rows = {row_index for ignored_boundary, row_index, ignored_gap in group}
        if len(rows) < min_support:
            continue
        center = table_boundary_group_center(group)
        average_gap = sum(gap for ignored_boundary, ignored_row, gap in group) / len(group)
        accepted.append((center, len(rows), average_gap))
    if len(accepted) > OCR_TABLE_COLUMN_INFERENCE_MAX_SPANS - 1:
        accepted = sorted(
            accepted,
            key=lambda item: (item[1], item[2]),
            reverse=True,
        )[: OCR_TABLE_COLUMN_INFERENCE_MAX_SPANS - 1]
    return [
        center
        for center, ignored_support, ignored_gap in sorted(
            accepted,
            key=lambda item: item[0],
        )
    ]


def table_boundary_group_center(group: list[tuple[float, int, float]]) -> float:
    return sum(boundary for boundary, ignored_row, ignored_gap in group) / len(group)


def table_column_boundary_min_support(row_count: int) -> int:
    return max(2, min(5, row_count // 5))


def table_page_rect_to_ocr_region(
    x0: float,
    y_top: float,
    x1: float,
    y_bottom: float,
    image: OcrImage,
    *,
    page_width: float,
    page_height: float,
    row_index: int,
    col_index: int | None,
    kind: str,
    padding_points: float,
) -> TableOcrRegion | None:
    x0, x1 = sorted((float(x0), float(x1)))
    y_bottom, y_top = sorted((float(y_bottom), float(y_top)))
    if x1 <= x0 or y_top <= y_bottom:
        return None
    geometry = table_image_geometry(
        image,
        page_width=page_width,
        page_height=page_height,
    )
    if geometry is None:
        return None
    rectangle = page_geometry.page_bbox_to_image_pixel_bbox(
        (x0, y_bottom, x1, y_top),
        geometry,
        padding=padding_points,
        clamp=True,
    )
    if rectangle is None:
        return None
    left, top, right, bottom = rectangle
    if right - left < 4 or bottom - top < 4:
        return None
    return TableOcrRegion(
        left,
        top,
        right,
        bottom,
        row_index=row_index,
        col_index=col_index,
        kind=kind,
        rotate_vertical=table_region_looks_vertical(left, top, right, bottom),
    )


def table_region_looks_vertical(left: int, top: int, right: int, bottom: int) -> bool:
    width = max(1, right - left)
    height = max(1, bottom - top)
    return height >= width * 1.8 and width <= 240


def table_row_ocr_candidate(
    image: OcrImage,
    regions: list[TableOcrRegion],
    timeout: float | None,
) -> OcrCandidate | None:
    if not regions:
        return None
    requests = [
        ocr_execution.RectangleOcrRequest(
            region.rectangle,
            table_region_page_segmentation_mode(region),
            dict(OCR_TESSERACT_TABLE_VARIABLES),
        )
        for region in regions
    ]
    primary_results = ocr_execution.ocr_image_regions_to_text_results_with_timeout(
        image,
        requests,
        timeout,
    )
    results = improve_table_row_ocr_results(
        image,
        regions,
        primary_results,
        timeout,
    )
    texts: list[str] = []
    confidences: list[int] = []
    for result in results:
        text = normalize_table_region_ocr_text(result.text)
        if not text:
            continue
        texts.append(text)
        if result.confidence is not None:
            confidences.append(result.confidence)
    if not texts:
        return None
    confidence = int(round(sum(confidences) / len(confidences))) if confidences else None
    return ocr_candidates.OcrCandidate(
        "table_rows",
        OcrTextResult("\n".join(texts), confidence),
        region_count=len(texts),
        image_width=image.width,
        image_height=image.height,
        image_resolution=image.resolution,
    )


def improve_table_row_ocr_results(
    image: OcrImage,
    regions: list[TableOcrRegion],
    primary_results: list[OcrTextResult],
    timeout: float | None,
) -> list[OcrTextResult]:
    if not regions or not primary_results:
        return primary_results
    broad_retry_rejection = table_row_profile_broad_retry_rejection(
        regions,
        primary_results,
    )
    if broad_retry_rejection is not None:
        return primary_results
    improved = list(primary_results)
    for profile_name, psm, variables in OCR_TESSERACT_TABLE_ROW_PROFILES:
        weak_indexes = table_row_profile_retry_indexes(regions, improved)
        if not weak_indexes:
            break
        requests = [
            ocr_execution.RectangleOcrRequest(
                regions[index].rectangle,
                psm,
                dict(variables),
            )
            for index in weak_indexes
        ]
        profile_results = ocr_execution.ocr_image_regions_to_text_results_with_timeout(
            image,
            requests,
            timeout,
        )
        for index, profile_result in zip(weak_indexes, profile_results, strict=False):
            current = improved[index]
            selected = select_table_row_profile_result(
                current,
                profile_result,
            )
            improved[index] = selected
    return improved


def table_row_profile_broad_retry_rejection(
    regions: list[TableOcrRegion],
    results: list[OcrTextResult],
) -> dict[str, Any] | None:
    if len(regions) < OCR_TABLE_ROW_PROFILE_BROAD_WEAK_MIN_ROWS:
        return None
    weak = [
        index
        for index, (region, result) in enumerate(zip(regions, results, strict=False))
        if should_retry_table_row_ocr_result(result, region)
    ]
    if len(weak) <= OCR_TABLE_ROW_PROFILE_MAX_RETRY_ROWS:
        return None
    weak_fraction = len(weak) / max(1, min(len(regions), len(results)))
    if weak_fraction <= OCR_TABLE_ROW_PROFILE_BROAD_WEAK_FRACTION:
        return None
    confidences = [
        result.confidence for result in results if result.confidence is not None and result.text
    ]
    average_confidence = int(round(sum(confidences) / len(confidences))) if confidences else None
    return {
        "kind": "table_row_profile_retry",
        "reason": "too_many_weak_rows",
        "rows": len(regions),
        "weak_rows": len(weak),
        "weak_fraction": round(weak_fraction, 4),
        "average_confidence": average_confidence,
    }


def table_row_profile_retry_indexes(
    regions: list[TableOcrRegion],
    results: list[OcrTextResult],
) -> list[int]:
    weak = [
        index
        for index, (region, result) in enumerate(zip(regions, results, strict=False))
        if should_retry_table_row_ocr_result(result, region)
    ]
    if len(weak) <= OCR_TABLE_ROW_PROFILE_MAX_RETRY_ROWS:
        return weak
    return sorted(
        weak,
        key=lambda index: table_row_profile_retry_priority(
            regions[index],
            results[index],
        ),
        reverse=True,
    )[:OCR_TABLE_ROW_PROFILE_MAX_RETRY_ROWS]


def table_row_profile_retry_priority(
    region: TableOcrRegion,
    result: OcrTextResult,
) -> tuple[int, int, float, int]:
    text = normalize_table_region_ocr_text(result.text)
    tokens = extracted_text_token_count(text)
    confidence = result.confidence if result.confidence is not None else 50
    return (
        1 if not text else 0,
        max(0, OCR_TABLE_ROW_PROFILE_RETRY_CONFIDENCE - confidence),
        text_ocr_quality_score(text),
        1 if tokens < 3 and region.width >= region.height * 3 else 0,
    )


def should_retry_table_row_ocr_result(
    result: OcrTextResult,
    region: TableOcrRegion,
) -> bool:
    text = normalize_table_region_ocr_text(result.text)
    tokens = extracted_text_token_count(text)
    confidence = result.confidence if result.confidence is not None else 50
    if not text:
        return True
    if confidence < OCR_TABLE_ROW_PROFILE_RETRY_CONFIDENCE:
        return True
    if tokens < 3 and region.width >= region.height * 3:
        return True
    return bool(text_ocr_quality_score(text) > 0.36 and confidence < 70)


def select_table_row_profile_result(
    primary: OcrTextResult,
    profile: OcrTextResult,
) -> OcrTextResult:
    profile_text = normalize_table_region_ocr_text(profile.text)
    primary_text = normalize_table_region_ocr_text(primary.text)
    if not profile_text:
        return primary
    if not primary_text:
        if not table_row_profile_result_can_seed_empty_primary(profile):
            return primary
        return profile
    if not table_row_profile_replacement_is_safe(primary, profile):
        return primary
    primary_tokens = extracted_text_token_count(primary_text)
    profile_tokens = extracted_text_token_count(profile_text)
    if primary_tokens >= 5 and profile_tokens < max(2, int(primary_tokens * 0.62)):
        return primary
    if profile_tokens > max(primary_tokens + 48, int(max(1, primary_tokens) * 2.7)):
        profile_quality = text_ocr_quality_score(profile_text)
        primary_quality = text_ocr_quality_score(primary_text)
        if profile_quality > primary_quality + 0.02:
            return primary
    primary_score = table_row_ocr_result_score(primary)
    profile_score = table_row_ocr_result_score(profile)
    return profile if profile_score > primary_score + 4.0 else primary


def table_row_profile_result_can_seed_empty_primary(result: OcrTextResult) -> bool:
    text = normalize_table_region_ocr_text(result.text)
    tokens = normalized_text_tokens(text)
    if len(tokens) < 3:
        return False
    confidence = result.confidence if result.confidence is not None else 50
    if confidence < 45:
        return False
    return text_ocr_quality_score(text) <= 0.34


def table_row_profile_replacement_is_safe(
    primary: OcrTextResult,
    profile: OcrTextResult,
) -> bool:
    primary_text = normalize_table_region_ocr_text(primary.text)
    profile_text = normalize_table_region_ocr_text(profile.text)
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    profile_confidence = profile.confidence if profile.confidence is not None else 50
    if profile_confidence + 8 < primary_confidence:
        return False
    if (
        profile_confidence < OCR_TABLE_ROW_PROFILE_RETRY_CONFIDENCE
        and profile_confidence < primary_confidence + 12
    ):
        return False
    primary_quality = text_ocr_quality_score(primary_text)
    profile_quality = text_ocr_quality_score(profile_text)
    if profile_quality > 0.40 and profile_quality >= primary_quality - 0.04:
        return False
    return table_row_profile_preserves_anchor_tokens(primary_text, profile_text)


def table_row_profile_preserves_anchor_tokens(
    primary_text: str,
    profile_text: str,
) -> bool:
    primary_anchors = table_ocr_line_anchor_tokens(normalized_text_tokens(primary_text))
    profile_anchors = table_ocr_line_anchor_tokens(normalized_text_tokens(profile_text))
    if len(primary_anchors) < 2 or len(profile_anchors) < 2:
        return True
    overlap = len(primary_anchors.intersection(profile_anchors))
    return overlap / max(1, min(len(primary_anchors), len(profile_anchors))) >= 0.45


def table_row_ocr_result_score(result: OcrTextResult) -> float:
    text = normalize_table_region_ocr_text(result.text)
    if not text:
        return float("-inf")
    tokens = normalized_text_tokens(text)
    confidence = result.confidence if result.confidence is not None else 50
    quality = text_ocr_quality_score(text)
    score = float(confidence) + min(len(tokens), 48) * 1.15 - quality * 45.0
    if table_ocr_line_has_table_signal(text, tokens):
        score += 6.0
    one_char_tokens = sum(1 for token in tokens if len(token) == 1)
    if tokens:
        score -= min(20.0, one_char_tokens / len(tokens) * 24.0)
    return score


def table_cell_ocr_candidate(
    image: OcrImage,
    regions: list[TableOcrRegion],
    timeout: float | None,
    *,
    use_rotated_vertical: bool,
) -> OcrCandidate | None:
    if not regions:
        return None
    active_regions = (
        [region for region in regions if region.rotate_vertical]
        if use_rotated_vertical
        else regions
    )
    if not active_regions:
        return None
    if use_rotated_vertical:
        if not table_vertical_cell_ocr_is_tractable(active_regions):
            return None
    elif not table_cell_ocr_is_tractable(active_regions):
        return None
    requests = [
        ocr_execution.RectangleOcrRequest(
            region.rectangle,
            table_region_page_segmentation_mode(region),
            dict(OCR_TESSERACT_TABLE_VARIABLES),
            rotate_vertical=use_rotated_vertical,
        )
        for region in active_regions
    ]
    results = ocr_execution.ocr_image_regions_to_text_results_with_timeout(
        image,
        requests,
        timeout,
    )
    rows: dict[int, list[tuple[int, str]]] = defaultdict(list)
    confidences: list[int] = []
    for region, result in zip(active_regions, results, strict=False):
        text = normalize_table_region_ocr_text(result.text)
        if not text:
            continue
        rows[region.row_index].append((region.col_index or 0, text))
        if result.confidence is not None:
            confidences.append(result.confidence)
    lines = [
        " ".join(text for ignored_col, text in sorted(items)).strip()
        for ignored_row, items in sorted(rows.items())
        if items
    ]
    lines = [line for line in lines if line]
    if not lines:
        return None
    confidence = int(round(sum(confidences) / len(confidences))) if confidences else None
    name = "table_cells_rotated" if use_rotated_vertical else "table_cells"
    return ocr_candidates.OcrCandidate(
        name,
        OcrTextResult("\n".join(lines), confidence),
        region_count=sum(len(items) for items in rows.values()),
        image_width=image.width,
        image_height=image.height,
        image_resolution=image.resolution,
    )


def table_cell_consensus_candidate(
    page: TableOcrPage,
    image: OcrImage,
    regions: list[TableOcrRegion],
    *,
    page_width: float,
    page_height: float,
    timeout: float | None,
) -> OcrCandidate | None:
    if not regions:
        return None
    active_regions = [
        region
        for region in regions
        if region.col_index is not None and region.width >= 4 and region.height >= 4
    ]
    if not active_regions:
        return None
    ocr_results: list[OcrTextResult | None]
    if table_cell_consensus_ocr_is_tractable(active_regions):
        requests = [
            ocr_execution.RectangleOcrRequest(
                region.rectangle,
                table_region_page_segmentation_mode(region),
                dict(OCR_TESSERACT_TABLE_VARIABLES),
                rotate_vertical=region.rotate_vertical,
            )
            for region in active_regions
        ]
        ocr_results = list(
            ocr_execution.ocr_image_regions_to_text_results_with_timeout(
                image,
                requests,
                timeout,
            )
        )
    else:
        ocr_results = [None for _ in active_regions]
    rows: dict[int, list[tuple[int, str]]] = defaultdict(list)
    confidences: list[int] = []
    filled_cells = 0
    for region, result in zip(active_regions, ocr_results, strict=False):
        native_text = native_text_for_table_region(
            page,
            region,
            image,
            page_width=page_width,
            page_height=page_height,
        )
        ocr_text = normalize_table_region_ocr_text(result.text) if result else ""
        selected_text, selected_source = select_table_cell_text(
            native_text,
            ocr_text,
            result.confidence if result is not None else None,
        )
        if not selected_text:
            continue
        filled_cells += 1
        if (
            (selected_source == "ocr" or selected_source == "agreement")
            and result is not None
            and result.confidence is not None
        ):
            confidences.append(result.confidence)
        rows[region.row_index].append((region.col_index or 0, selected_text))
    lines = [
        " ".join(text for ignored_col, text in sorted(items)).strip()
        for ignored_row, items in sorted(rows.items())
        if items
    ]
    lines = [line for line in lines if line]
    if not lines:
        return None
    confidence = int(round(sum(confidences) / len(confidences))) if confidences else None
    return ocr_candidates.OcrCandidate(
        "table_cell_consensus",
        OcrTextResult("\n".join(lines), confidence),
        region_count=filled_cells,
        image_width=image.width,
        image_height=image.height,
        image_resolution=image.resolution,
    )


def table_cell_consensus_ocr_is_tractable(regions: list[TableOcrRegion]) -> bool:
    if len(regions) <= OCR_TABLE_CELL_TRACTABLE_REGION_COUNT:
        return True
    if len(regions) > OCR_TABLE_CONSENSUS_TRACTABLE_REGION_COUNT:
        return False
    areas = sorted(region.area for region in regions)
    if not areas:
        return False
    median_area = areas[len(areas) // 2]
    return median_area >= OCR_TABLE_CELL_TRACTABLE_MEDIAN_AREA


def native_text_for_table_region(
    page: TableOcrPage,
    region: TableOcrRegion,
    image: OcrImage,
    *,
    page_width: float,
    page_height: float,
) -> str:
    page_rect = table_ocr_region_to_page_rect(
        region,
        image,
        page_width=page_width,
        page_height=page_height,
    )
    if page_rect is None:
        return ""
    runs: list[Any] = []
    for run in getattr(page, "chars", ()):
        if not getattr(run, "has_text", False):
            continue
        if not getattr(run, "stripped_text", ""):
            continue
        if getattr(run, "visible", True) is False:
            continue
        run_rect = text_run_rect(run)
        if run_rect is None:
            continue
        if table_text_run_overlaps_cell(run_rect, page_rect):
            runs.append(run)
    return normalize_table_region_ocr_text(table_native_runs_to_text(runs))


def table_ocr_region_to_page_rect(
    region: TableOcrRegion,
    image: OcrImage,
    *,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float] | None:
    geometry = table_image_geometry(
        image,
        page_width=page_width,
        page_height=page_height,
    )
    if geometry is None:
        return None
    return page_geometry.image_bbox_to_page_bbox(region.rectangle, geometry)


def table_image_geometry(
    image: OcrImage,
    *,
    page_width: float,
    page_height: float,
) -> page_geometry.ImageSpace | None:
    if image.width <= 0 or image.height <= 0:
        return None
    if page_width <= 0.0 or page_height <= 0.0:
        return None
    return page_geometry.ImageSpace.from_dimensions(
        image_width=image.width,
        image_height=image.height,
        image_resolution=image.resolution,
        page_width=page_width,
        page_height=page_height,
        source=image.source,
    )


def text_run_rect(run: Any) -> tuple[float, float, float, float] | None:
    return page_geometry.text_run_bbox(run)


def table_text_run_overlaps_cell(
    run_rect: tuple[float, float, float, float],
    cell_rect: tuple[float, float, float, float],
) -> bool:
    run_mid_x = (run_rect[0] + run_rect[2]) * 0.5
    run_mid_y = (run_rect[1] + run_rect[3]) * 0.5
    slop = 0.75
    if (
        cell_rect[0] - slop <= run_mid_x <= cell_rect[2] + slop
        and cell_rect[1] - slop <= run_mid_y <= cell_rect[3] + slop
    ):
        return True
    overlap = max(0.0, min(run_rect[2], cell_rect[2]) - max(run_rect[0], cell_rect[0]))
    overlap *= max(0.0, min(run_rect[3], cell_rect[3]) - max(run_rect[1], cell_rect[1]))
    run_area = max(1.0, (run_rect[2] - run_rect[0]) * (run_rect[3] - run_rect[1]))
    return overlap / run_area >= 0.35


def table_native_runs_to_text(runs: list[Any]) -> str:
    if not runs:
        return ""
    line_groups: list[list[Any]] = []
    for run in sorted(
        runs,
        key=lambda item: (
            -float(
                getattr(
                    item,
                    "mid_y",
                    (getattr(item, "y0", 0.0) + getattr(item, "y1", 0.0)) * 0.5,
                )
            ),
            float(getattr(item, "x0", 0.0)),
        ),
    ):
        run_mid_y = float(
            getattr(
                run,
                "mid_y",
                (getattr(run, "y0", 0.0) + getattr(run, "y1", 0.0)) * 0.5,
            )
        )
        run_height = max(
            1.0,
            float(getattr(run, "height", getattr(run, "y1", 0.0) - getattr(run, "y0", 0.0))),
        )
        if line_groups:
            last = line_groups[-1]
            last_mid_y = sum(
                float(
                    getattr(
                        item,
                        "mid_y",
                        (getattr(item, "y0", 0.0) + getattr(item, "y1", 0.0)) * 0.5,
                    )
                )
                for item in last
            ) / len(last)
            if abs(run_mid_y - last_mid_y) <= max(1.5, run_height * 0.65):
                last.append(run)
                continue
        line_groups.append([run])
    return "\n".join(table_native_run_line_text(group) for group in line_groups)


def table_native_run_line_text(runs: list[Any]) -> str:
    text_parts: list[str] = []
    previous_x1: float | None = None
    previous_space_width = 0.0
    for run in sorted(runs, key=lambda item: float(getattr(item, "x0", 0.0))):
        run_text = str(getattr(run, "text", "") or "")
        if not run_text:
            continue
        x0 = float(getattr(run, "x0", 0.0))
        x1 = float(getattr(run, "x1", x0))
        height = max(
            1.0,
            float(getattr(run, "height", getattr(run, "y1", 0.0) - getattr(run, "y0", 0.0))),
        )
        space_width = float(getattr(run, "space_width", 0.0) or 0.0)
        if previous_x1 is not None:
            threshold = max(1.0, previous_space_width * 0.45, height * 0.28)
            if x0 - previous_x1 > threshold and text_parts and not text_parts[-1].endswith(" "):
                text_parts.append(" ")
        text_parts.append(run_text)
        previous_x1 = max(previous_x1 if previous_x1 is not None else x1, x1)
        previous_space_width = space_width
    return "".join(text_parts)


def select_table_cell_text(
    native_text: str,
    ocr_text: str,
    ocr_confidence: int | None,
) -> tuple[str, str]:
    native_text = normalize_table_region_ocr_text(native_text)
    ocr_text = normalize_table_region_ocr_text(ocr_text)
    if not native_text:
        return (ocr_text, "ocr" if ocr_text else "")
    if not ocr_text:
        return (native_text, "native")
    if table_cell_texts_agree(native_text, ocr_text):
        return (select_agreeing_table_cell_text(native_text, ocr_text), "agreement")
    if table_cell_native_preserves_numeric_recall(native_text, ocr_text):
        return (native_text, "native")
    native_tokens = extracted_text_token_count(native_text)
    ocr_tokens = extracted_text_token_count(ocr_text)
    native_quality = text_ocr_quality_score(native_text)
    ocr_quality = text_ocr_quality_score(ocr_text)
    confidence = ocr_confidence if ocr_confidence is not None else 50
    if (
        confidence >= 68
        and ocr_quality + 0.08 < native_quality
        and ocr_tokens >= max(1, int(native_tokens * 0.60))
    ):
        return (ocr_text, "ocr")
    if ocr_tokens > native_tokens and ocr_quality <= native_quality + 0.04:
        return (ocr_text, "ocr")
    return (native_text, "native")


def table_cell_texts_agree(left: str, right: str) -> bool:
    left_tokens = set(normalized_text_tokens(left))
    right_tokens = set(normalized_text_tokens(right))
    if not left_tokens or not right_tokens:
        return left.strip().casefold() == right.strip().casefold()
    overlap = len(left_tokens.intersection(right_tokens))
    return overlap / max(1, min(len(left_tokens), len(right_tokens))) >= 0.70


def select_agreeing_table_cell_text(left: str, right: str) -> str:
    left_tokens = extracted_text_token_count(left)
    right_tokens = extracted_text_token_count(right)
    if (
        right_tokens > left_tokens
        and text_ocr_quality_score(right) <= text_ocr_quality_score(left) + 0.05
    ):
        return right
    return left


def table_cell_native_preserves_numeric_recall(native_text: str, ocr_text: str) -> bool:
    native_digit_tokens = table_cell_digit_token_count(native_text)
    if native_digit_tokens == 0:
        return False
    ocr_digit_tokens = table_cell_digit_token_count(ocr_text)
    if ocr_digit_tokens + 1 < native_digit_tokens:
        return True
    native_tokens = extracted_text_token_count(native_text)
    ocr_tokens = extracted_text_token_count(ocr_text)
    if native_tokens >= 3 and ocr_tokens < max(1, int(native_tokens * 0.55)):
        return True
    return numeric_token_ratio(ocr_text) + 0.15 < numeric_token_ratio(native_text)


def table_cell_digit_token_count(text: str) -> int:
    return sum(1 for token in normalized_text_tokens(text) if any(ch.isdigit() for ch in token))


def table_cell_ocr_is_tractable(regions: list[TableOcrRegion]) -> bool:
    region_count = len(regions)
    if region_count <= OCR_TABLE_CELL_TRACTABLE_REGION_COUNT:
        return True
    if region_count > OCR_TABLE_CELL_TRACTABLE_EXTENDED_REGION_COUNT:
        return False
    areas = sorted(region.area for region in regions)
    if not areas:
        return False
    median_area = areas[len(areas) // 2]
    return median_area >= OCR_TABLE_CELL_TRACTABLE_MEDIAN_AREA


def table_vertical_cell_ocr_is_tractable(regions: list[TableOcrRegion]) -> bool:
    if len(regions) <= OCR_TABLE_VERTICAL_TRACTABLE_REGION_COUNT:
        return True
    areas = sorted(region.area for region in regions)
    if not areas:
        return False
    median_area = areas[len(areas) // 2]
    return (
        len(regions) <= OCR_TABLE_CELL_TRACTABLE_EXTENDED_REGION_COUNT
        and median_area >= OCR_TABLE_CELL_TRACTABLE_MEDIAN_AREA
    )


def table_region_page_segmentation_mode(region: TableOcrRegion) -> int:
    if region.rotate_vertical:
        return 7
    if region.height <= 96 and region.width >= region.height * 2:
        return 7
    if region.width <= 180 and region.height <= 96:
        return 13
    return OCR_TABLE_DEFAULT_PAGE_SEGMENTATION_MODE


def normalize_table_region_ocr_text(text: str) -> str:
    return " ".join(part for part in text.replace("\f", "\n").split() if part)


__all__ = (
    "OCR_TABLE_CANDIDATE_NAMES",
    "OCR_TESSERACT_TABLE_VARIABLES",
    "RasterTableLine",
    "TableOcrRegion",
    "collect_table_rectangle_ocr_candidates",
    "normalize_table_region_ocr_text",
    "page_display_dimensions_for_ocr",
    "raster_table_grid_for_ocr",
    "raster_table_lines_from_image",
    "refine_table_ocr_regions_with_textlines",
    "select_table_cell_text",
    "select_table_row_profile_result",
    "table_grid_for_ocr",
    "table_cell_consensus_candidate",
    "table_ocr_line_anchor_tokens",
    "table_ocr_line_has_table_signal",
    "table_ocr_regions_from_grid",
    "table_row_ocr_candidate",
)
