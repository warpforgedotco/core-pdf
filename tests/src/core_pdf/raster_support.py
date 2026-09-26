"""Raster targets and page renders shared by the rasterizer tests."""

import numpy

from core_pdf import PdfDocument
from core_pdf.impl.render_clipping import ClipState
from core_pdf.impl.render_target import RasterTarget

BACKDROP_PIXEL = bytes([10, 200, 90, 180])


def make_target(width: int = 4, height: int = 1, *, pixel: bytes = bytes(4)) -> RasterTarget:
    """A scale-1 target over a width x height page filled with one RGBA pixel."""
    pixels = bytearray(pixel * (width * height))
    view = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(height, width, 4)
    clip = ClipState(crop_x0=0, crop_y1=height, scale=1, width=width, height=height)
    return RasterTarget(
        pixels,
        None,
        clip=clip,
        width=width,
        height=height,
        scale=1,
        crop_x0=0,
        crop_y0=0,
        crop_y1=height,
        page_view=view,
    )


def make_backdrop_target(width: int, height: int, *, planes: bool) -> RasterTarget:
    """A target over a translucent backdrop, optionally recording group planes."""
    target = make_target(width, height, pixel=BACKDROP_PIXEL)
    if planes:
        target.group_source_alpha = numpy.full((height, width), 0.25, dtype=numpy.float32)
        target.group_source_shape = numpy.full((height, width), 0.5, dtype=numpy.float32)
        target.paint_window = []
    return target


def rendered(data: bytes) -> bytes:
    """The first page of data rasterized at scale 1.5, as raw pixel bytes."""
    with PdfDocument(data) as document:
        return document.pages[0].render().rasterize(scale=1.5).array().tobytes()
