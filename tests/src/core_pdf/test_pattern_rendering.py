import numpy
import pytest

from core_pdf.impl._impl.capture.program import CapturedProgram
from core_pdf.impl._impl.capture.records import CapturedDrawing, TilingPattern
from core_pdf.impl._impl.render import patterns
from core_pdf.impl._impl.render.clipping import internal_ClipState
from core_pdf.impl._impl.render.target import internal_RasterTarget


def internal_target(width=4, height=1):
    pixels = bytearray(width * height * 4)
    view = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(height, width, 4)
    clip = internal_ClipState(crop_x0=0, crop_y1=height, scale=1, width=width, height=height)
    return internal_RasterTarget(
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
    value = patterns.axial_shading_t(coords, *point)
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
    value = patterns.radial_shading_t(coords, *point)
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
    assert patterns.internal_shading_color_rgba(model, components, opacity) == expected


@pytest.mark.parametrize("extend", [False, True])
def test_axial_gradient_pixels_respect_extension_flags(extend):
    target = internal_target()
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
    target = internal_target()
    target.paint_shading({"dictionary": dictionary}, None)
    assert target.pixels == bytearray(16)


@pytest.mark.parametrize(
    ("mode", "expected"), [(None, True), ("Normal", True), ("nOrMaL", True), ("Multiply", False)]
)
def test_tiling_blend_optimization_requires_only_normal_paints(mode, expected):
    drawing = CapturedDrawing(0, None, None, blend_mode=mode)
    pattern = TilingPattern((0, 0, 2, 2), 2, 2, CapturedProgram(drawings=(drawing,)))
    active = set()
    assert patterns.internal_tiling_pattern_uses_normal_blends(pattern, active) is expected
    assert active == set()


def test_nested_tiling_blends_and_cycles_disable_isolated_optimization():
    nested_drawing = CapturedDrawing(0, None, None, blend_mode="Multiply")
    nested = TilingPattern((0, 0, 2, 2), 2, 2, CapturedProgram(drawings=(nested_drawing,)))
    outer_drawing = CapturedDrawing(0, None, None, fill_pattern=nested)
    outer = TilingPattern((0, 0, 2, 2), 2, 2, CapturedProgram(drawings=(outer_drawing,)))
    assert not patterns.internal_tiling_pattern_uses_normal_blends(outer)
    nested_drawing.blend_mode = None
    assert patterns.internal_tiling_pattern_uses_normal_blends(outer)
    nested_drawing.stroke_pattern = outer
    active = set()
    assert not patterns.internal_tiling_pattern_uses_normal_blends(outer, active)
    assert active == set()


def test_tiling_cell_cache_reuses_display_and_clip_for_same_pattern():
    target = internal_target()
    pattern = TilingPattern((0, 0, 2, 2), 2, 2, CapturedProgram())
    first = patterns.internal_tiling_cell(target, pattern)
    second = patterns.internal_tiling_cell(target, pattern)
    assert first[0] is second[0]
    assert first[1] is second[1]
    target.group_source_shape = numpy.zeros((1, 4), dtype=numpy.float32)
    grouped = patterns.internal_tiling_cell(target, pattern)
    assert grouped[0] is not first[0]
    assert len(target.tiling_cell_cache) == 2
