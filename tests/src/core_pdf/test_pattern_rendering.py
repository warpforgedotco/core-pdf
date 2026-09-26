import numpy
import pytest

from core_pdf.impl import render_patterns as patterns
from core_pdf.impl.capture_program import CapturedProgram
from core_pdf.impl.capture_records import CapturedDrawing, CapturedPath, TilingPattern
from core_pdf.impl.render_model import DisplayListItem, PathPaintItem, PathPaintKind
from core_pdf_cythonized import shading_t
from tests.src.core_pdf.raster_support import make_target


@pytest.mark.parametrize(
    ("coords", "point", "expected"),
    [
        ((0, 0, 0, 0), (1, 1), None),
        ((0, 0, 10, 0), (5, 7), 0.5),
        ((0, 0, 10, 0), (-5, 0), -0.5),
        ((0, 0, 0, 10), (7, 15), 1.5),
        ((1, 1, 3, 3), (2, 2), 0.5),
    ],
)
def test_axial_projection_handles_degenerate_and_extended_axis(coords, point, expected):
    value = shading_t(2, coords, point[0], point[1], 0.5)
    assert value is None if expected is None else value == pytest.approx(expected)


@pytest.mark.parametrize(
    ("coords", "point", "expected"),
    [
        ((0, 0, 1, 0, 0, 1), (1, 0), None),
        ((0, 0, 0, 1, 0, 1), (1, 0), 0.5),
        ((0, 0, 1, 2, 0, 1), (1, 5), None),
        ((0, 0, 0, 0, 0, 10), (3, 4), 0.5),
        ((0, 0, 0, 0, 0, 10), (20, 0), 2),
        ((0, 0, 1, 2, 0, 1), (1, 0), 1),
        ((0, 0, 0, 0, 0, 10), (float("nan"), 0), None),
    ],
)
def test_radial_projection_selects_valid_circle_solution(coords, point, expected):
    value = shading_t(3, coords, point[0], point[1], 0.5)
    assert value is None if expected is None else value == pytest.approx(expected)


@pytest.mark.parametrize(
    ("model", "components", "opacity", "expected"),
    [
        ("DeviceGray", (), None, (0, 0, 0, 255)),
        ("DeviceGray", (0.5,), 0.5, (128, 128, 128, 128)),
        ("DeviceRGB", (0.25,), None, (64, 64, 64, 255)),
        ("", (), None, (0, 0, 0, 255)),
        ("DeviceRGB", (1, 0.5), None, (255, 128, 128, 255)),
        ("DeviceRGB", (-1, 0.5, 2), 2, (0, 128, 255, 255)),
        ("DeviceRGB", (1, 0, 0), -1, (255, 0, 0, 0)),
        ("DeviceRGB", (1, 0, 0), True, (255, 0, 0, 255)),
    ],
)
def test_shading_color_clamps_channels_and_fills_missing_components(
    model, components, opacity, expected
):
    assert patterns.shading_color_rgba(model, components, opacity) == expected


@pytest.mark.parametrize("extend", [False, True])
def test_axial_gradient_pixels_respect_extension_flags(extend):
    target = make_target()
    target.paint_shading(
        {
            "dictionary": {
                "ShadingType": 2,
                "ColorSpace": "DeviceGray",
                "Coords": [1, 0, 3, 0],
                "Extend": [extend, extend],
                "Function": {"FunctionType": 2, "Domain": [0, 1], "C0": [0], "C1": [1], "N": 1},
            }
        },
        None,
    )
    pixels = numpy.frombuffer(target.pixels, dtype=numpy.uint8).reshape(1, 4, 4)
    expected = [
        [0, 0, 0, 255 if extend else 0],
        [64, 64, 64, 255],
        [191, 191, 191, 255],
        [255, 255, 255, 255] if extend else [0, 0, 0, 0],
    ]
    numpy.testing.assert_array_equal(pixels[0], expected)


@pytest.mark.parametrize("dictionary", [None, {}, {"ShadingType": 1}])
def test_unsupported_shading_leaves_raster_untouched(dictionary):
    target = make_target()
    target.paint_shading({"dictionary": dictionary}, None)
    assert target.pixels == bytearray(16)


@pytest.mark.parametrize(
    ("mode", "expected"), [(None, True), ("Normal", True), ("nOrMaL", True), ("Multiply", False)]
)
def test_tiling_blend_optimization_requires_only_normal_paints(mode, expected):
    drawing = CapturedDrawing(0, None, None, blend_mode=mode)
    pattern = TilingPattern((0, 0, 2, 2), 2, 2, CapturedProgram(drawings=(drawing,)))
    active = set()
    assert patterns.tiling_pattern_uses_normal_blends(pattern, active) is expected
    assert active == set()


def test_nested_tiling_blends_and_cycles_disable_isolated_optimization():
    nested_drawing = CapturedDrawing(0, None, None, blend_mode="Multiply")
    nested = TilingPattern((0, 0, 2, 2), 2, 2, CapturedProgram(drawings=(nested_drawing,)))
    outer_drawing = CapturedDrawing(0, None, None, fill_pattern=nested)
    outer = TilingPattern((0, 0, 2, 2), 2, 2, CapturedProgram(drawings=(outer_drawing,)))
    assert not patterns.tiling_pattern_uses_normal_blends(outer)
    nested_drawing.blend_mode = None
    assert patterns.tiling_pattern_uses_normal_blends(outer)
    nested_drawing.stroke_pattern = outer
    active = set()
    assert not patterns.tiling_pattern_uses_normal_blends(outer, active)
    assert active == set()


def test_tiling_cell_cache_reuses_display_and_clip_for_same_pattern():
    target = make_target()
    pattern = TilingPattern((0, 0, 2, 2), 2, 2, CapturedProgram())
    first = patterns.tiling_cell(target, pattern)
    second = patterns.tiling_cell(target, pattern)
    assert first[0] is second[0]
    assert first[1] is second[1]
    target.group_source_shape = numpy.zeros((1, 4), dtype=numpy.float32)
    grouped = patterns.tiling_cell(target, pattern)
    assert grouped[0] is not first[0]
    assert len(target.tiling_cell_cache) == 2


def cell_item(box, line_width=0.0):
    path = CapturedPath()
    path.rect(box[0], box[1], box[2] - box[0], box[3] - box[1])
    return PathPaintItem(
        PathPaintKind.STROKE if line_width else PathPaintKind.FILL,
        0,
        box,
        path,
        None,
        None,
        (0, 0, 0, 255),
        None,
        line_width,
        0,
        0,
        None,
        "nonzero",
        None,
        None,
    )


def unit_cell_clip():
    clip = CapturedPath()
    clip.rect(0.0, 0.0, 1.0, 1.0)
    return clip


def test_a_cell_whose_content_lies_outside_its_clip_paints_nothing():
    items = [
        DisplayListItem("scope-begin", 0),
        DisplayListItem("clip", 0, {"bbox": (-2434.8, -26661.5, -2414.8, -26641.5)}),
        cell_item((-2434.8, -26661.5, -2414.8, -26641.5)),
        DisplayListItem("glyph", 0, {"bbox": (50.0, 50.0, 52.0, 52.0)}),
        DisplayListItem("scope-end", 0),
    ]
    assert patterns.cell_paints_nothing(items, unit_cell_clip(), 1.0)


@pytest.mark.parametrize(
    "item",
    [
        pytest.param(cell_item((0.5, 0.5, 2.0, 2.0)), id="overlapping fill"),
        pytest.param(cell_item((1.5, 1.5, 3.0, 3.0)), id="fill within the pixel margin"),
        pytest.param(
            cell_item((3.0, 3.0, 4.0, 4.0), line_width=0.5), id="stroke whose joins reach"
        ),
    ],
)
def test_content_that_can_reach_the_clip_is_painted(item):
    assert not patterns.cell_paints_nothing([item], unit_cell_clip(), 1.0)


@pytest.mark.parametrize(
    ("kind", "data"),
    [("shading", {"bbox": (100.0, 100.0, 101.0, 101.0)}), ("glyph", {}), ("annotation", {})],
)
def test_an_item_of_unknown_extent_is_assumed_to_paint(kind, data):
    assert not patterns.cell_paints_nothing([DisplayListItem(kind, 0, data)], unit_cell_clip(), 1.0)
