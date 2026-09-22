from copy import replace

import numpy
import pytest

from core_pdf.impl._impl.capture.records import CapturedPath, CapturedSubpath
from core_pdf.impl._impl.render.model import DisplayListItem, PathPaintItem, PathPaintKind
from core_pdf_ocr.impl.extract.ocr.atlas import rasterize_packed_stroked_paths


def internal_item(points, *, closed=False, width=1):
    return PathPaintItem(
        PathPaintKind.STROKE,
        0,
        None,
        CapturedPath([CapturedSubpath(points, closed=closed)]),
        None,
        None,
        (0,),
        1,
        width,
        0,
        0,
        None,
        "nonzero",
        None,
        None,
    )


@pytest.mark.parametrize(
    "points", [[], [(2, 2)], [(2, 2), (2, 2)], [(2, 2), (2 + 9e-13, 2 + 9e-13)]]
)
def test_degenerate_strokes_leave_opaque_white_atlas(points) -> None:
    result = rasterize_packed_stroked_paths((internal_item(points),), 10, 10, 1)
    assert result.channels == 4
    assert min(result.pixels) == 255


@pytest.mark.parametrize("kind", ["nonpath", "fill", "foreign-path"])
def test_atlas_ignores_items_outside_its_stroke_contract(kind) -> None:
    item = internal_item([(2, 2), (8, 8)])
    if kind == "nonpath":
        item = DisplayListItem("clip", 0)
    elif kind == "fill":
        item = replace(item, paint_kind=PathPaintKind.FILL)
    else:
        item = replace(item, path=object())
    result = rasterize_packed_stroked_paths((item,), 10, 10, 1)
    assert min(result.pixels) == 255


@pytest.mark.parametrize(
    "points", [[(2, 2), (8, 2)], [(2, 2), (2, 8)], [(2, 2), (8, 7)], [(-5, 2), (15, 7)]]
)
def test_stroke_direction_does_not_change_clipped_coverage(points) -> None:
    forward = rasterize_packed_stroked_paths((internal_item(points),), 10, 10, 1)
    backward = rasterize_packed_stroked_paths((internal_item(list(reversed(points))),), 10, 10, 1)
    numpy.testing.assert_array_equal(forward.array(), backward.array())
    assert min(forward.pixels) < 255
    assert forward.array()[:, :, 3].min() == 255


def test_closed_subpath_matches_explicit_closing_segment() -> None:
    points = [(2, 2), (8, 2), (8, 8)]
    closed = rasterize_packed_stroked_paths((internal_item(points, closed=True),), 10, 10, 1)
    explicit = rasterize_packed_stroked_paths(
        (internal_item([*points, points[0]], closed=True),), 10, 10, 1
    )
    numpy.testing.assert_array_equal(closed.array(), explicit.array())
    assert closed.array()[5, 5, 0] < 255


def test_thicker_strokes_cover_more_pixels_and_tiny_dimensions_stay_valid() -> None:
    thin = rasterize_packed_stroked_paths((internal_item([(2, 5), (8, 5)]),), 10, 10, 1)
    thick = rasterize_packed_stroked_paths((internal_item([(2, 5), (8, 5)], width=6),), 10, 10, 1)
    assert numpy.count_nonzero(thick.array()[:, :, 0] < 255) > numpy.count_nonzero(
        thin.array()[:, :, 0] < 255
    )
    tiny = rasterize_packed_stroked_paths((), 0.01, 0.01, 0)
    assert (tiny.width, tiny.height) == (1, 1)
    assert bytes(tiny.pixels) == bytes([255]) * 4
