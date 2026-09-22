from copy import replace

import numpy as np
import pytest

from core_pdf.impl._impl.capture.program import CapturedProgram
from core_pdf.impl._impl.capture.records import CapturedPath, CapturedSoftMask, CapturedSubpath
from core_pdf.impl._impl.render.clipping import internal_ClipState
from core_pdf.impl._impl.render.commands import translated_command
from core_pdf.impl._impl.render.display import DisplayList
from core_pdf.impl._impl.render.model import DisplayListItem, PathPaintItem
from core_pdf.impl._impl.render.target import internal_RasterTarget


def internal_triangle_item():
    path = CapturedPath([CapturedSubpath([(1, 1), (4, 1), (2, 4)], closed=True)])
    edges = np.asarray(path.fill_edges(), dtype=np.float64)
    edges.flags.writeable = False
    display = DisplayList(12, 12)
    display.append("fill", 7, path=path, bbox=path.bbox(), edge_array=edges, fill=(1, 0, 0))
    item = display.items[0]
    assert isinstance(item, PathPaintItem)
    return item


def internal_render(item):
    pixels = bytearray(12 * 12 * 4)
    view = np.frombuffer(pixels, dtype=np.uint8).reshape(12, 12, 4)
    target = internal_RasterTarget(
        pixels,
        None,
        clip=internal_ClipState(crop_x0=0, crop_y1=12, scale=1, width=12, height=12),
        width=12,
        height=12,
        scale=1,
        crop_x0=0,
        crop_y0=0,
        crop_y1=12,
        page_view=view,
    )
    target.paint_item(item)
    return view.copy()


@pytest.mark.parametrize(("tx", "ty"), [(0, 0), (4, 0), (0, 4), (3, 5), (-1, -1), (0.25, 0.75)])
def test_translated_cached_edges_match_path_geometry_and_raster(tx, ty):
    original = internal_triangle_item()
    original_edges = original.edge_array.copy()
    placed = translated_command(original, tx, ty)
    assert isinstance(placed, PathPaintItem)
    np.testing.assert_array_equal(placed.edge_array, np.asarray(placed.path.fill_edges()))
    actual = internal_render(placed)
    expected = internal_render(replace(placed, edge_array=None))
    assert expected[:, :, 3].any()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(original.edge_array, original_edges)
    assert original.path.bbox() == (1, 1, 4, 4)
    assert placed.bbox == (1 + tx, 1 + ty, 4 + tx, 4 + ty)


@pytest.mark.parametrize("own_blend", [None, "Normal", "Multiply"])
@pytest.mark.parametrize(("tx", "ty"), [(0, 0), (2, -3)])
def test_path_translation_preserves_mask_program_and_explicit_blend(own_blend, tx, ty):
    mask = CapturedSoftMask(CapturedProgram(), offset=(5, 6))
    original = replace(
        internal_triangle_item(), edge_array=None, blend_mode=own_blend, graphics_soft_mask=mask
    )
    placed = translated_command(original, tx, ty, "Screen")
    assert isinstance(placed, PathPaintItem)
    assert placed.blend_mode == (own_blend or "Screen")
    assert placed.edge_array is None
    assert placed.graphics_soft_mask is not None
    assert placed.graphics_soft_mask.program is mask.program
    assert placed.graphics_soft_mask.offset == (5 + tx, 6 + ty)
    assert mask.offset == (5, 6)
    if tx == ty == 0:
        assert placed.graphics_soft_mask is mask


@pytest.mark.parametrize(
    ("shading_type", "coords", "expected"),
    [
        (2, [1, 2, 3, 4], [11, 22, 13, 24]),
        (3, [1, 2, 5, 3, 4, 6], [11, 22, 5, 13, 24, 6]),
        (2, [1], [1]),
        (3, [], []),
    ],
)
@pytest.mark.parametrize("container", [list, tuple])
def test_shading_translation_moves_centres_and_bounds_without_changing_radii(
    shading_type, coords, expected, container
):
    dictionary = {"ShadingType": shading_type, "Coords": container(coords), "BBox": (0, 0, 5, 6)}
    original = DisplayListItem("shading", 9, {"dictionary": dictionary, "rect": (0, 0, 5, 6)})
    placed = translated_command(original, 10, 20, "Multiply")
    assert isinstance(placed, DisplayListItem)
    assert placed.data["dictionary"]["Coords"] == expected
    assert placed.data["dictionary"]["BBox"] == (10, 20, 15, 26)
    assert placed.data["rect"] == (10, 20, 15, 26)
    assert placed.data["blend_mode"] == "Multiply"
    assert dictionary["Coords"] == container(coords)
    assert dictionary["BBox"] == (0, 0, 5, 6)


def test_generic_clip_translation_retains_shared_mask_capture_and_source_path():
    path = internal_triangle_item().path
    mask = CapturedSoftMask(CapturedProgram())
    original = DisplayListItem("clip", 1, {"path": path, "bbox": None, "graphics_soft_mask": mask})
    placed = translated_command(original, 2, 3)
    assert isinstance(placed, DisplayListItem)
    assert placed.data["path"].bbox() == (3, 4, 6, 7)
    assert placed.data["bbox"] is None
    assert placed.data["graphics_soft_mask"].offset == (2, 3)
    assert placed.data["graphics_soft_mask"].program is mask.program
    assert path.bbox() == (1, 1, 4, 4)
