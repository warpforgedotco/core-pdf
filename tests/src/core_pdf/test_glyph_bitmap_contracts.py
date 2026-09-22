import numpy as np
import pytest

from core_pdf.impl.render.clipping import ClipState
from core_pdf.impl.render.target import RasterTarget


def make_target():
    pixels = bytearray(12 * 12 * 4)
    view = np.frombuffer(pixels, dtype=np.uint8).reshape(12, 12, 4)
    return RasterTarget(
        pixels,
        None,
        clip=ClipState(crop_x0=0, crop_y1=12, scale=1, width=12, height=12),
        width=12,
        height=12,
        scale=1,
        crop_x0=0,
        crop_y0=0,
        crop_y1=12,
        page_view=view,
    ), view


@pytest.mark.parametrize("alpha", [128, 255])
@pytest.mark.parametrize("offset", [0, 0.25])
@pytest.mark.parametrize(
    "rows", [(0b101, 0b010), (0b101,), (0b101, 0b010, 0b111), (0b1101, 0b1010)]
)
def test_bitmap_paint_agrees_with_declared_cell_geometry(alpha, offset, rows):
    target, actual = make_target()
    reference, expected = make_target()
    color = (200, 30, 50, alpha)
    box = (2 + offset, 4, 8 + offset, 8)
    target.draw_glyph_bitmap(box, rows, color, bitmap_width=3, bitmap_height=2)
    for row_index, row in enumerate(rows[:2]):
        for column in range(3):
            if row & (1 << column):
                reference.fill_rect(
                    (
                        2 + offset + column * 2,
                        6 - row_index * 2,
                        4 + offset + column * 2,
                        8 - row_index * 2,
                    ),
                    color,
                )
    np.testing.assert_array_equal(actual, expected)
    assert not actual[8:, :, 3].any()


@pytest.mark.parametrize(
    ("box", "bitmap", "width", "height"),
    [
        (None, [1], 1, 1),
        ((0, 0, 0, 2), [1], 1, 1),
        ((0, 0, 2, 2), [], 1, 1),
        ((0, 0, 2, 2), ["bad", True], 1, 1),
        ((0, 0, 2, 2), [0], None, None),
        ((0, 0, 2, 2), [1], -1, 1),
        ((0, 0, 2, 2), [1], 1, -1),
    ],
)
def test_unpaintable_bitmap_inputs_leave_target_untouched(box, bitmap, width, height):
    target, view = make_target()
    target.draw_glyph_bitmap(
        box, bitmap, (200, 30, 50, 255), bitmap_width=width, bitmap_height=height
    )
    assert not view.any()


def test_inferred_bitmap_dimensions_match_explicit_dimensions():
    inferred, actual = make_target()
    explicit, expected = make_target()
    rows = (0b101, 0b010)
    inferred.draw_glyph_bitmap((2, 4, 8, 8), rows, (200, 30, 50, 255))
    explicit.draw_glyph_bitmap(
        (2, 4, 8, 8), rows, (200, 30, 50, 255), bitmap_width=3, bitmap_height=2
    )
    np.testing.assert_array_equal(actual, expected)
