from __future__ import annotations

from io import BytesIO

import pytest

from core_pdf.impl.engine.extraction.page import PdfPage as ExtractedPdfPage
from core_pdf.impl.engine.layout.models import TextRun
from core_pdf.impl.engine.rendering.models import DisplayList, RenderedPage
from core_pdf.impl.engine.spec.s_07_document.document_source import DocumentSourceMixin
from core_pdf.impl.engine.spec.s_07_document.page import PdfPage
from core_pdf.impl.exceptions import PdfRasterTooLargeError


class StaticDisplayPage(PdfPage):
    def __init__(
        self,
        chars: list[TextRun],
        *,
        rotation: int = 0,
        width: float = 0,
        height: float = 0,
    ) -> None:
        self.static_chars = chars
        self.static_rotation = rotation
        self.static_width = width
        self.static_height = height

    @property
    def chars(self) -> list[TextRun]:
        return self.static_chars

    @property
    def rotation(self) -> int:
        return self.static_rotation

    @property
    def width(self) -> float:
        return self.static_width

    @property
    def height(self) -> float:
        return self.static_height


class StaticExtractedPage(ExtractedPdfPage):
    def __init__(self, display_chars: list[TextRun]) -> None:
        self.static_display_chars = display_chars

    @property
    def display_chars(self) -> list[TextRun]:
        return self.static_display_chars


def rendered_page(*, width: float = 5, height: float = 7, rotate: int = 0) -> RenderedPage:
    return RenderedPage(
        page_number=1,
        width=width,
        height=height,
        rotate=rotate,
        display_list=DisplayList(width=width, height=height),
    )


def test_raster_size_accounts_for_rotation_crop_and_scale() -> None:
    page = rendered_page(width=100, height=200, rotate=90)
    page.metadata["crop"] = (10, 20, 70, 100)

    assert page.unrotated_raster_size(scale=2) == (120, 160)
    assert page.raster_size(scale=2) == (160, 120)
    assert len(page.rasterize(scale=2)) == 160 * 120 * 4


def test_rasterize_rejects_oversized_canvas_before_allocation() -> None:
    page = rendered_page(width=100, height=200)

    with pytest.raises(PdfRasterTooLargeError, match="pixels=20000, maximum=19999"):
        page.rasterize(max_pixels=19_999)

    assert page.raster_cache == {}


def test_seekable_source_is_read_from_start_and_restored() -> None:
    source = BytesIO(b"complete PDF source")
    source.seek(9)
    loader = DocumentSourceMixin()

    data = loader.load_data(source)

    assert data == b"complete PDF source"
    assert source.tell() == 9


def test_display_chars_apply_page_rotation_to_text_geometry() -> None:
    run = TextRun(
        "text",
        10,
        20,
        30,
        40,
        10,
        20,
        12,
        4,
        0,
        0,
        0,
    )
    page = StaticDisplayPage([run], rotation=90, width=100, height=200)

    displayed = page.display_chars

    assert len(displayed) == 1
    assert (displayed[0].x0, displayed[0].y0, displayed[0].x1, displayed[0].y1) == (
        20,
        70,
        40,
        90,
    )
    assert displayed[0].rotation_angle == 270


def test_text_rotation_correction_uses_displayed_text_orientation() -> None:
    run = TextRun(
        "dominant text",
        0,
        0,
        10,
        10,
        0,
        0,
        10,
        3,
        0,
        0,
        0,
        rotation_angle=270,
    )
    page = StaticExtractedPage([run])

    correction = page.text_rotation_correction(threshold=0.95)

    assert correction == 90


def test_bitmap_text_rows_use_top_to_bottom_glyph_order() -> None:
    page = rendered_page()
    page.display_list.append(
        "text",
        1,
        text="L",
        bbox=(0, 0, 5, 7),
        fill_color=(0, 0, 0),
    )

    raster = page.rasterize(background=(255, 255, 255, 255))

    top_row = [raster[x * 4] for x in range(5)]
    bottom_row_start = 6 * 5 * 4
    bottom_row = [raster[bottom_row_start + x * 4] for x in range(5)]
    assert top_row == [0, 255, 255, 255, 255]
    assert bottom_row == [0, 0, 0, 0, 0]
