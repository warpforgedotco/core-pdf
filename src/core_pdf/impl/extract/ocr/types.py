# SPDX-License-Identifier: AGPL-3.0-only
"""Records and box geometry shared across the OCR stage.

The recognition stage is large enough to have its own internal vocabulary: rasters,
the regions cut from them, and the tasks queued against Tesseract. These types are
referenced from every part of the stage, so they live here rather than in whichever
module happened to need them first.
"""

from __future__ import annotations

from dataclasses import dataclass

from core_pdf.impl.render.model import RasterImage


@dataclass(frozen=True, slots=True)
class internal_Raster:
    image: RasterImage
    resolution: int
    render_report: dict[str, object] | None = None

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    @property
    def nbytes(self) -> int:
        """Expose the wrapped pixel footprint to the document image cache."""
        return self.image.nbytes


@dataclass(frozen=True, slots=True)
class internal_RasterRegion:
    """A decoded raster coupled to the page-space area it actually represents."""

    raster: internal_Raster
    page_box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class internal_StrokedTextCell:
    """A translated vector-text run in a packed OCR raster."""

    source_box: tuple[float, float, float, float]
    packed_box: tuple[float, float, float, float]
    drawing_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class internal_PackedStrokedTextRaster:
    """One compact raster plus the piecewise map back into PDF page space."""

    raster: internal_Raster
    packed_box: tuple[float, float, float, float]
    cells: tuple[internal_StrokedTextCell, ...]


@dataclass(frozen=True, slots=True)
class internal_OcrRegion:
    """A ranked page-space region selected before compositor rasterization."""

    page_box: tuple[float, float, float, float]
    score: float
    reasons: tuple[str, ...]

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.page_box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass(frozen=True, slots=True)
class internal_OcrTask:
    mode: int
    image: RasterImage
    rectangle: tuple[int, int, int, int]
    page_box: tuple[float, float, float, float]
    resolution: int
    minimum_confidence: float = 20.0
    character_confidence_threshold: float | None = None
    recognize_words: bool = False
    collect_symbols: bool = False


def internal_pixel_box_to_page_box(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    page_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Map a top-left-origin pixel box into bottom-left PDF page space."""
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


def internal_map_ocr_box(
    task: internal_OcrTask,
    bbox: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Map one Tesseract pixel box into the task's PDF coordinate space."""
    return internal_pixel_box_to_page_box(bbox, task.image.width, task.image.height, task.page_box)


def internal_ocr_region_box(
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


def internal_raster_rectangle_page_box(
    raster: internal_Raster,
    page_box: tuple[float, float, float, float],
    rectangle: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Map a top-left raster rectangle into bottom-left PDF page space."""
    x, y, width, height = rectangle
    return internal_pixel_box_to_page_box(
        (x, y, x + width, y + height), raster.width, raster.height, page_box
    )
