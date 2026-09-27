# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_pdf.impl.geometry import bbox_area
from core_pdf.impl.render_model import RasterImage
from core_pdf.impl.types import GeneratedRecord


class Raster(GeneratedRecord):
    image: RasterImage
    resolution: int

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


class RasterRegion(GeneratedRecord):
    raster: Raster
    page_box: tuple[float, float, float, float]


class StrokedTextCell(GeneratedRecord):
    source_box: tuple[float, float, float, float]
    packed_box: tuple[float, float, float, float]
    drawing_indexes: tuple[int, ...]


class PackedStrokedTextRaster(GeneratedRecord):
    raster: Raster
    packed_box: tuple[float, float, float, float]
    cells: tuple[StrokedTextCell, ...]


class OcrRegion(GeneratedRecord):
    page_box: tuple[float, float, float, float]
    score: float
    reasons: tuple[str, ...]

    @property
    def area(self) -> float:
        return bbox_area(self.page_box)


class OcrTask(GeneratedRecord):
    mode: int
    image: RasterImage
    rectangle: tuple[int, int, int, int]
    page_box: tuple[float, float, float, float]
    resolution: int
    minimum_confidence: float = 20.0
    character_confidence_threshold: float | None = None
    recognize_words: bool = False
    collect_symbols: bool = False


def pixel_box_to_page_box(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    page_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    page_x0, page_y0, page_x1, page_y1 = page_box
    page_width = page_x1 - page_x0
    page_height = page_y1 - page_y0
    return (
        page_x0 + x0 * page_width / image_width,
        page_y1 - y1 * page_height / image_height,
        page_x0 + x1 * page_width / image_width,
        page_y1 - y0 * page_height / image_height,
    )


def map_ocr_box(
    task: OcrTask,
    bbox: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    return pixel_box_to_page_box(bbox, task.image.width, task.image.height, task.page_box)


def ocr_region_box(
    box: tuple[float, float, float, float],
    *,
    page_width: float,
    page_height: float,
    padding: float,
) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = box
    clipped = (
        max(0.0, x0 - padding),
        max(0.0, y0 - padding),
        min(page_width, x1 + padding),
        min(page_height, y1 + padding),
    )
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def raster_rectangle_page_box(
    raster: Raster,
    page_box: tuple[float, float, float, float],
    rectangle: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    x, y, width, height = rectangle
    return pixel_box_to_page_box(
        (x, y, x + width, y + height), raster.width, raster.height, page_box
    )
