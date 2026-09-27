import numpy

from core_pdf import PdfDocument
from core_pdf.impl.render_clipping import ClipState
from core_pdf.impl.render_grid import DeviceGrid
from core_pdf.impl.render_model import PixelWindow
from core_pdf.impl.render_target import RasterTarget

BACKDROP_PIXEL = bytes([10, 200, 90, 180])


def make_target(width: int = 4, height: int = 1, *, pixel: bytes = bytes(4)) -> RasterTarget:
    pixels = bytearray(pixel * (width * height))
    view = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(height, width, 4)
    clip = ClipState(DeviceGrid(0, 0, height, 1, width, height))
    return RasterTarget(
        pixels,
        None,
        clip=clip,
        page_view=view,
    )


def make_backdrop_target(width: int, height: int, *, planes: bool) -> RasterTarget:
    target = make_target(width, height, pixel=BACKDROP_PIXEL)
    if planes:
        target.group_source_alpha = numpy.full((height, width), 0.25, dtype=numpy.float32)
        target.group_source_shape = numpy.full((height, width), 0.5, dtype=numpy.float32)
        target.paint_window = PixelWindow()
    return target


def rendered(data: bytes) -> bytes:
    with PdfDocument(data) as document:
        return document.pages[0].render().rasterize(scale=1.5).array().tobytes()
