# SPDX-License-Identifier: AGPL-3.0-only
"""Propose, merge, and classify OCR regions from captured page evidence."""

from __future__ import annotations

import numpy

from core_pdf.impl.extract.contracts import (
    MAX_OCR_PIXELS,
    VECTOR_PAINT_KINDS,
    OcrPass,
    PageAnalysis,
    internal_bbox_tuple,
)
from core_pdf.impl.extract.grids import (
    internal_axis_segments,
    internal_grid_components,
    internal_line_coordinate_columns,
    internal_split_grid_component,
)
from core_pdf.impl.extract.ocr.raster import (
    internal_decoded_image_raster,
    internal_direct_image_orientation,
    internal_orient_direct_image_raster,
)
from core_pdf.impl.extract.ocr.types import (
    internal_ocr_region_box,
    internal_OcrRegion,
    internal_RasterRegion,
)
from core_pdf.impl.model.geometry import (
    bbox_area,
    bbox_intersection_area,
    bbox_union,
    rect_tuple,
)


def internal_page_image_regions(
    capture: PageAnalysis,
    *,
    minimum_area_ratio: float,
    max_pixels: int = MAX_OCR_PIXELS,
    maximum_axis_deviation: float = 1e-5,
    upscale: bool = True,
) -> tuple[internal_RasterRegion, ...]:
    page_width = capture.width
    page_height = capture.height
    page_area = max(1.0, page_width * page_height)
    regions: list[internal_RasterRegion] = []
    for image in capture.program.drawings:
        if image.kind != "image":
            continue
        orientation = internal_direct_image_orientation(
            image,
            maximum_axis_deviation=maximum_axis_deviation,
        )
        if orientation is None:
            continue
        box = rect_tuple(image.rect)
        if box is None:
            continue
        clipped = (
            max(0.0, box[0]),
            max(0.0, box[1]),
            min(page_width, box[2]),
            min(page_height, box[3]),
        )
        # A decoded source raster represents the full image. If the image is clipped by
        # the page, mapping that full raster onto the clipped rectangle would compress
        # its OCR coordinates. Let the page compositor produce the correct crop instead.
        clip_tolerance = max(2.0, max(page_width, page_height) * 0.005)
        if any(
            abs(float(original) - clipped_value) > clip_tolerance
            for original, clipped_value in zip(box, clipped, strict=True)
        ):
            continue
        display_area = bbox_area(clipped)
        if display_area / page_area < minimum_area_ratio:
            continue
        raster = internal_decoded_image_raster(
            image,
            display_area,
            max_pixels=max_pixels,
            upscale=upscale,
        )
        if raster is not None:
            oriented = internal_orient_direct_image_raster(
                image,
                raster,
                orientation=orientation,
            )
            regions.append(
                internal_RasterRegion(
                    oriented,
                    clipped,
                )
            )
    return tuple(regions)


def internal_dominant_image_region(
    capture: PageAnalysis,
    *,
    max_pixels: int = MAX_OCR_PIXELS,
    upscale: bool = True,
) -> internal_RasterRegion | None:
    regions = internal_page_image_regions(
        capture,
        minimum_area_ratio=0.65,
        max_pixels=max_pixels,
        upscale=upscale,
    )
    substantial = tuple(
        region for region in regions if region.raster.width * region.raster.height >= 4_096
    )
    if not substantial:
        return None
    if len(substantial) > 1:
        largest = max(substantial, key=lambda region: bbox_area(region.page_box))
        largest_area = max(1.0, bbox_area(largest.page_box))
        overlapping = sum(
            bbox_intersection_area(region.page_box, largest.page_box) / largest_area >= 0.90
            for region in substantial
            if region is not largest
        )
        if overlapping:
            return None
    return max(substantial, key=lambda region: region.raster.width * region.raster.height)


OCR_REGION_INITIAL_COUNT = 8


OCR_REGION_INITIAL_AREA_RATIO = 0.25


OCR_DIRECT_REGION_MIN_COVERAGE = 0.65


def internal_merge_ocr_regions(regions: list[internal_OcrRegion]) -> tuple[internal_OcrRegion, ...]:
    merged: list[internal_OcrRegion] = []
    merged_areas: list[float] = []
    for region in sorted(regions, key=lambda item: (-item.score, item.page_box)):
        region_box = region.page_box
        region_area = bbox_area(region_box)
        match = None
        for index, existing in enumerate(merged):
            smaller = min(merged_areas[index], region_area)
            if not smaller:
                continue
            if bbox_intersection_area(existing.page_box, region_box) >= smaller * 0.35:
                match = index
                break
        if match is None:
            merged.append(region)
            merged_areas.append(region_area)
            continue
        existing = merged[match]
        merged_box = bbox_union((existing.page_box, region_box))
        assert merged_box is not None
        merged[match] = internal_OcrRegion(
            merged_box,
            max(existing.score, region.score) + min(existing.score, region.score) * 0.15,
            tuple(dict.fromkeys((*existing.reasons, *region.reasons))),
        )
        merged_areas[match] = bbox_area(merged_box)
    return tuple(sorted(merged, key=lambda item: (-item.score, item.page_box)))


def internal_candidate_ocr_regions(capture: PageAnalysis) -> tuple[internal_OcrRegion, ...]:
    """Select likely OCR areas using capture-time geometry only.

    This deliberately does not render a preview image.  Native text, image bounds,
    captured paths, and grid lines are already available from the canonical page IR.
    """
    page_width = capture.width
    page_height = capture.height
    page_area = max(1.0, page_width * page_height)
    padding = max(6.0, min(36.0, min(page_width, page_height) * 0.01))
    candidates: list[internal_OcrRegion] = []

    for evidence_box in capture.evidence.image_boxes:
        image_box = internal_bbox_tuple(evidence_box)
        padded = internal_ocr_region_box(
            image_box,
            page_width=page_width,
            page_height=page_height,
            padding=padding,
        )
        if padded is not None:
            candidates.append(internal_OcrRegion(padded, 5.0, ("image",)))

    native = capture.observations
    native_boxes = native.bbox

    def native_overlap(box: tuple[float, float, float, float]) -> float:
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        overlap_width = numpy.maximum(
            0.0,
            numpy.minimum(native_boxes[:, 2], box[2]) - numpy.maximum(native_boxes[:, 0], box[0]),
        )
        overlap_height = numpy.maximum(
            0.0,
            numpy.minimum(native_boxes[:, 3], box[3]) - numpy.maximum(native_boxes[:, 1], box[1]),
        )
        return min(1.0, float(numpy.sum(overlap_width * overlap_height)) / area)

    for drawing in capture.program.drawings:
        if drawing.kind not in {"fill", "fillstroke", "stroke"}:
            continue
        box = rect_tuple(drawing.rect)
        if box is None:
            continue
        drawing_area = bbox_area(box)
        if drawing_area <= 0.0 or drawing_area >= page_area * 0.80:
            continue
        uncovered = native_overlap(box) < 0.25
        if uncovered and drawing.kind in {"fill", "fillstroke"}:
            padded = internal_ocr_region_box(
                box,
                page_width=page_width,
                page_height=page_height,
                padding=padding,
            )
            if padded is not None:
                candidates.append(internal_OcrRegion(padded, 3.5, ("uncovered-vector",)))

    horizontal, vertical = internal_axis_segments(capture)
    for component_horizontal, component_vertical in internal_grid_components(horizontal, vertical):
        x0 = min(float(component_horizontal[:, 0].min()), float(component_vertical[:, 0].min()))
        y0 = min(float(component_horizontal[:, 2].min()), float(component_vertical[:, 1].min()))
        x1 = max(float(component_horizontal[:, 1].max()), float(component_vertical[:, 0].max()))
        y1 = max(float(component_horizontal[:, 2].max()), float(component_vertical[:, 2].max()))
        for split_horizontal, split_vertical in internal_split_grid_component(
            component_horizontal,
            component_vertical,
        ):
            split_box = (
                min(float(split_horizontal[:, 0].min()), float(split_vertical[:, 0].min())),
                min(float(split_horizontal[:, 2].min()), float(split_vertical[:, 1].min())),
                max(float(split_horizontal[:, 1].max()), float(split_vertical[:, 0].max())),
                max(float(split_horizontal[:, 2].max()), float(split_vertical[:, 2].max())),
            )
            padded = internal_ocr_region_box(
                split_box,
                page_width=page_width,
                page_height=page_height,
                padding=padding,
            )
            if padded is not None and (
                (padded[2] - padded[0]) * (padded[3] - padded[1]) < page_area * 0.45
            ):
                candidates.append(internal_OcrRegion(padded, 4.0, ("grid",)))
        if not component_horizontal.size or not component_vertical.size:
            continue
        component_box = (x0, y0, x1, y1)
        component_area = (x1 - x0) * (y1 - y0)
        if component_area < page_area * 0.45 and native_overlap(component_box) < 0.25:
            padded = internal_ocr_region_box(
                component_box,
                page_width=page_width,
                page_height=page_height,
                padding=padding,
            )
            if padded is not None:
                candidates.append(internal_OcrRegion(padded, 3.0, ("grid-labels",)))

    columns = 6
    rows = max(2, min(8, int(round(columns * page_height / max(1.0, page_width)))))
    vector_density = numpy.zeros(rows * columns, dtype=numpy.float32)
    for drawing in capture.program.drawings:
        if drawing.kind not in VECTOR_PAINT_KINDS:
            continue
        box = rect_tuple(drawing.rect)
        if box is None:
            continue
        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        column = min(columns - 1, max(0, int(center_x * columns / max(1.0, page_width))))
        row = min(rows - 1, max(0, int(center_y * rows / max(1.0, page_height))))
        vector_density[row * columns + column] += 1.0
    grid_lines = capture.program.lines
    if len(grid_lines):
        # Bin every grid line at once.
        line_x0, line_y0, line_x1, line_y1 = internal_line_coordinate_columns(grid_lines)
        line_columns = numpy.clip(
            ((line_x0 + line_x1) * 0.5 * columns / max(1.0, page_width)).astype(numpy.int64),
            0,
            columns - 1,
        )
        line_rows = numpy.clip(
            ((line_y0 + line_y1) * 0.5 * rows / max(1.0, page_height)).astype(numpy.int64),
            0,
            rows - 1,
        )
        vector_density += (
            numpy.bincount(
                line_rows * columns + line_columns,
                minlength=rows * columns,
            ).astype(numpy.float32)
            * 0.5
        )

    native_counts = numpy.zeros(rows * columns, dtype=numpy.float32)
    for text, raw_box in zip(native.text, native.bbox, strict=True):
        box = internal_bbox_tuple(raw_box)
        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        column = min(columns - 1, max(0, int(center_x * columns / max(1.0, page_width))))
        row = min(rows - 1, max(0, int(center_y * rows / max(1.0, page_height))))
        native_counts[row * columns + column] += sum(not char.isspace() for char in text)

    for cell, density in enumerate(vector_density):
        if density <= 0.0:
            continue
        row, column = divmod(cell, columns)
        cell_box = (
            column * page_width / columns,
            row * page_height / rows,
            (column + 1) * page_width / columns,
            (row + 1) * page_height / rows,
        )
        sparse = native_counts[cell] < 8.0
        header_band = row in {0, rows - 1} and native_counts[cell] < 24.0
        if not sparse and not header_band:
            continue
        padded = internal_ocr_region_box(
            cell_box,
            page_width=page_width,
            page_height=page_height,
            padding=padding,
        )
        if padded is not None:
            reasons = ["vector-density"]
            if sparse:
                reasons.append("sparse-label")
            if header_band:
                reasons.append("header-band")
            candidates.append(
                internal_OcrRegion(
                    padded,
                    1.5 + min(2.0, float(density) / 8.0),
                    tuple(reasons),
                )
            )

    if (
        capture.evidence.vector_complexity >= 180
        and capture.evidence.text_coverage < 0.05
        and (len(native_boxes) == 0 or len(native_boxes) >= 8)
    ):
        # Component labels are often isolated from the larger paths they
        # annotate. Use finer cells for these vector-only pages so the region
        # budget can select several label clusters instead of one broad artwork
        # box. The existing coarse density pass remains responsible for larger
        # diagram areas.
        label_columns = 12
        label_rows = max(
            4,
            min(12, int(round(label_columns * page_height / max(1.0, page_width)))),
        )
        label_density = numpy.zeros(label_rows * label_columns, dtype=numpy.float32)
        label_boxes: list[list[tuple[float, float, float, float]]] = [
            [] for _ in range(label_rows * label_columns)
        ]
        for drawing in capture.program.drawings:
            if drawing.kind not in VECTOR_PAINT_KINDS:
                continue
            box = rect_tuple(drawing.rect)
            if box is None:
                continue
            center_x = (box[0] + box[2]) * 0.5
            center_y = (box[1] + box[3]) * 0.5
            column = min(
                label_columns - 1,
                max(0, int(center_x * label_columns / max(1.0, page_width))),
            )
            row = min(
                label_rows - 1,
                max(0, int(center_y * label_rows / max(1.0, page_height))),
            )
            label_density[row * label_columns + column] += 1.0
            label_boxes[row * label_columns + column].append(box)

        for cell, density in enumerate(label_density):
            if density <= 0.0:
                continue
            row, column = divmod(cell, label_columns)
            cell_box = (
                column * page_width / label_columns,
                row * page_height / label_rows,
                (column + 1) * page_width / label_columns,
                (row + 1) * page_height / label_rows,
            )
            component_boxes = label_boxes[cell]
            optional_component_box = bbox_union(component_boxes)
            assert optional_component_box is not None
            component_box = optional_component_box
            component_area = bbox_area(component_box)
            label_padding = max(
                padding,
                min(72.0, min(page_width, page_height) * 0.03),
            )
            candidate_box = component_box if component_area <= page_area * 0.08 else cell_box
            padded = internal_ocr_region_box(
                candidate_box,
                page_width=page_width,
                page_height=page_height,
                padding=label_padding if candidate_box == component_box else padding,
            )
            if padded is not None:
                candidates.append(
                    internal_OcrRegion(
                        padded,
                        1.0 + min(3.0, float(density) / 8.0),
                        ("vector-label-density", "vector-label-neighborhood")
                        if candidate_box == component_box
                        else ("vector-label-density",),
                    )
                )

    regions = internal_merge_ocr_regions(candidates)
    if not regions:
        regions = (
            internal_OcrRegion(
                (0.0, 0.0, page_width, page_height),
                0.0,
                ("page-fallback",),
            ),
        )
    return regions


def internal_has_distributed_outline_text(capture: PageAnalysis) -> bool:
    """Detect pages whose text was converted into many small filled vector paths."""
    page_width = capture.width
    page_height = capture.height
    max_width = max(24.0, page_width * 0.04)
    max_height = max(24.0, page_height * 0.04)
    boxes = tuple(
        box
        for drawing in capture.program.drawings
        if drawing.kind in {"fill", "fillstroke"}
        and (box := rect_tuple(drawing.rect)) is not None
        and 0.0 < box[2] - box[0] <= max_width
        and 0.0 < box[3] - box[1] <= max_height
    )
    if len(boxes) < 200:
        return False
    bounds = bbox_union(boxes)
    assert bounds is not None
    width_ratio = (bounds[2] - bounds[0]) / max(1.0, page_width)
    height_ratio = (bounds[3] - bounds[1]) / max(1.0, page_height)
    return width_ratio >= 0.60 and height_ratio >= 0.60


def internal_ocr_region_batch(
    regions: tuple[internal_OcrRegion, ...],
    ocr_pass: OcrPass,
    *,
    page_area: float,
) -> tuple[internal_OcrRegion, ...]:
    count_limit = max(ocr_pass.max_regions, OCR_REGION_INITIAL_COUNT)
    area_limit = OCR_REGION_INITIAL_AREA_RATIO
    selected: list[internal_OcrRegion] = []
    area = 0.0
    page_area = max(1.0, page_area)
    for region in regions:
        if len(selected) >= count_limit:
            break
        if selected and area + region.area > page_area * area_limit:
            continue
        selected.append(region)
        area += region.area
    return tuple(selected)
