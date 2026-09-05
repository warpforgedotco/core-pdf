import numpy

from core_pdf.impl.render.display import DisplayList
from core_pdf.impl.spec.s_07_content.capture import CapturedPath
from core_pdf_ocr.impl.extract.ocr.atlas import rasterize_packed_stroked_paths


def test_packed_stroked_rasterizer_draws_antialiased_skeleton() -> None:
    display_list = DisplayList(width=10.0, height=6.0)
    path = CapturedPath()
    path.move_to(1.0, 1.0)
    path.line_to(9.0, 5.0)
    display_list.append(
        "stroke",
        1,
        path=path,
        stroke_color=(0.0, 0.0, 0.0),
        stroke_opacity=1.0,
        line_width=0.48,
        line_cap=1,
        line_join=1,
        dash_pattern=([], 0.0),
    )

    raster = rasterize_packed_stroked_paths(tuple(display_list.items), 10.0, 6.0, 6.0)
    pixels = raster.array()

    assert (raster.width, raster.height) == (60, 36)
    assert numpy.count_nonzero(pixels[:, :, 0] < 255) > 0
    assert numpy.count_nonzero(pixels[:, :, 0] < 128) > 0
    assert numpy.all(pixels[:, :, 3] == 255)
