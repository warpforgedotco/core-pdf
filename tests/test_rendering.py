from __future__ import annotations

from core_pdf.impl.engine.rendering.models import DisplayList, RenderedPage


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
