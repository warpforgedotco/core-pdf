"""Raster image projections for captured axial and radial shading paints."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy

from core_pdf._vendor.mupdf.shade_raster import Raster
from core_pdf.api.compat._shared import float32
from core_pdf.api.compat.pymupdf.colors import scalar_colors_to_rgb
from core_pdf.api.compat.pymupdf.functions import compile_color_function
from core_pdf.api.compat.pymupdf.geometry import Matrix, Rect
from core_pdf.api.document import PdfPage
from core_pdf.impl._impl.capture.records import CapturedDrawing
from core_pdf.impl._impl.graphics.images import ImageRaster
from core_pdf.impl._impl.graphics.shading import PreparedShading, prepare_shading


@dataclass(frozen=True, slots=True)
class ShadingRaster:
    seqno: int
    bbox: Rect
    transform: Matrix
    raster: ImageRaster


def internal_shading_palette(shading: PreparedShading) -> numpy.ndarray:
    lower, upper = map(float32, shading.domain)
    span = float32(upper - lower)
    colors = []
    for index in range(256):
        value = float32(lower + float32(float32(index / 255.0) * span))
        colors.append(shading.evaluate(value))
    samples = numpy.asarray(colors, dtype=numpy.float32)
    if samples.shape[1] == 1:
        samples = scalar_colors_to_rgb(samples, "gray")
    elif shading.color_model.endswith("DeviceCMYK") and samples.shape[1] == 4:
        samples = scalar_colors_to_rgb(samples, "cmyk")
    return numpy.clip(samples[:, :3] * numpy.float32(255), 0, 255).astype(numpy.uint8)


def rasterize_shading(
    drawing: CapturedDrawing,
    *,
    page_matrix: Matrix,
    page_bounds: Rect,
    clip: Rect,
    flags: int,
) -> ShadingRaster | None:
    """Export the paint's clipped device raster, retaining its fractional bounds."""
    shading = prepare_shading(drawing.dictionary, compile_function=compile_color_function)
    if shading is None or drawing.shading_matrix is None:
        return None
    if drawing.fill_opacity is not None and drawing.fill_opacity < 0.5:
        return None
    transform = Matrix(drawing.shading_matrix) * page_matrix
    bounds = Rect(page_bounds)
    content_clip = (
        drawing.control_point_clip
        if drawing.control_point_clip is not None
        else drawing.shading_clip
    )
    if content_clip is not None:
        bounds &= Rect(content_clip) * page_matrix
    if flags & 64:
        bounds &= clip
    scissor = Rect(bounds)
    if shading.bbox is not None:
        bounds &= Rect(shading.bbox) * transform
    if bounds.is_empty or not all(math.isfinite(value) for value in bounds):
        return None
    # The image itself covers whole pixels. Structured text subsequently clips
    # that image placement back to the original fractional content scissor.
    integer_bounds = (
        math.floor(bounds.x0),
        math.floor(bounds.y0),
        math.ceil(bounds.x1),
        math.ceil(bounds.y1),
    )
    image_bounds = Rect(integer_bounds)
    if not (clip.intersects(image_bounds) if flags & 64 else clip.contains(image_bounds)):
        return None
    raster = Raster(integer_bounds)
    coords = tuple(map(float32, shading.coords))
    extend = (shading.extend_start, shading.extend_end)
    if shading.shading_type == 2:
        raster.axial(coords, tuple(transform), extend)
    else:
        raster.radial(coords, tuple(transform), extend)
    palette = internal_shading_palette(shading)
    pixels = numpy.zeros((*raster.indices.shape, 4), dtype=numpy.uint8)
    pixels[:, :, :3] = palette[raster.indices]
    pixels[~raster.coverage] = 0
    pixels[:, :, 3] = raster.coverage.astype(numpy.uint8) * 255
    background = drawing.dictionary.get("Background") if drawing.dictionary is not None else None
    if isinstance(background, (list, tuple)) and len(background) in (1, 3):
        values = tuple(background) * 3 if len(background) == 1 else background
        pixels[~raster.coverage, :3] = [int(max(0, min(255, float32(v * 255)))) for v in values]
        pixels = pixels[:, :, :3]
    return ShadingRaster(
        drawing.seqno,
        image_bounds & scissor if flags & 64 else image_bounds,
        Matrix(image_bounds.width, 0, 0, image_bounds.height, image_bounds.x0, image_bounds.y0),
        ImageRaster(pixels, "rgb", has_alpha=pixels.shape[2] == 4),
    )


def capture_shadings(
    page: PdfPage, *, clip: Rect, flags: int, matrix: Matrix | None = None
) -> list[ShadingRaster]:
    crop = Rect(page.crop_box or page.media_box)
    unit_value = page.document.resolver.resolve(page.page_dict.get("UserUnit"))
    unit = float32(unit_value) if isinstance(unit_value, (int, float)) else 1.0
    page_matrix = Matrix(unit, 0, 0, -unit, float32(-crop.x0 * unit), float32(crop.y1 * unit))
    if matrix is not None:
        page_matrix *= matrix
    page_bounds = crop * page_matrix
    result = []
    for drawing in page.get_page_program().drawings:
        if drawing.kind != "shading":
            continue
        raster = rasterize_shading(
            drawing, page_matrix=page_matrix, page_bounds=page_bounds, clip=clip, flags=flags
        )
        if raster is not None:
            result.append(raster)
    return result


__all__ = ("ShadingRaster", "capture_shadings", "rasterize_shading")
