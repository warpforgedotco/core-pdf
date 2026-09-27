import numpy
import pytest

from core_pdf.impl import render_blend
from core_pdf_cythonized import blend_normal_alpha_array_numpy, composite_normal_group


def source_over(destination, source, mode=None):
    sr, sg, sb, sa = source
    if sa == 0:
        return tuple(destination)
    source_alpha = sa / 255
    destination_alpha = destination[3] / 255
    output_alpha = source_alpha + destination_alpha * (1 - source_alpha)
    result = []
    for s, d in zip((sr, sg, sb), destination[:3], strict=True):
        s = s / 255
        d_unit = d / 255
        if mode == "Multiply":
            s = s * (1 - destination_alpha) + destination_alpha * (s * d_unit)
        elif mode == "Screen":
            s = s * (1 - destination_alpha) + destination_alpha * (1 - (1 - s) * (1 - d_unit))
        result.append(
            round(
                (s * 255 * source_alpha + d * destination_alpha * (1 - source_alpha)) / output_alpha
            )
        )
    return (*result, round(output_alpha * 255))


@pytest.mark.parametrize("mode", [None, "Multiply", "Screen"])
@pytest.mark.parametrize("source_alpha", [0, 64, 128, 255])
@pytest.mark.parametrize("destination_alpha", [0, 128, 255])
def test_solid_blend_writes_through_strided_views(mode, source_alpha, destination_alpha):
    backing = numpy.full((3, 5, 4), 17, dtype=numpy.uint8)
    target = backing[:, 1:4:2]
    destination = (51, 102, 153, destination_alpha)
    target[:] = destination
    source = (204, 85, 34, source_alpha)
    render_blend.blend_solid_array_numpy(target, source, mode)
    expected = source_over(destination, source, mode)
    numpy.testing.assert_array_equal(target, numpy.broadcast_to(expected, target.shape))
    numpy.testing.assert_array_equal(backing[:, ::2], 17)


@pytest.mark.parametrize("source_alpha", [0, 64, 128, 255])
@pytest.mark.parametrize("destination_alpha", [0, 128, 255])
def test_normal_fast_path_matches_source_over_on_strided_views(source_alpha, destination_alpha):
    backing = numpy.full((3, 5, 4), 17, dtype=numpy.uint8)
    target = backing[:, 1:4:2]
    destination = (51, 102, 153, destination_alpha)
    target[:] = destination
    source = (204, 85, 34, source_alpha)
    render_blend.blend_normal_solid_array_numpy(target, source)
    numpy.testing.assert_array_equal(
        target, numpy.broadcast_to(source_over(destination, source), target.shape)
    )
    numpy.testing.assert_array_equal(backing[:, ::2], 17)


@pytest.mark.parametrize("mode", [None, "Multiply", "Screen"])
@pytest.mark.parametrize(
    ("source_scale", "target_scale"), [(None, None), (0.5, 0.5), (0, 1), (1, 0), (1, 1)]
)
def test_group_blending_rounds_each_alpha_stage_and_preserves_unpainted_pixels(
    mode, source_scale, target_scale
):
    backing = numpy.full((2, 6, 4), 17, dtype=numpy.uint8)
    target = backing[:, ::2]
    target[:] = [(51, 102, 153, 0), (51, 102, 153, 128), (51, 102, 153, 255)]
    source = numpy.array(
        [[(204, 85, 34, 0), (204, 85, 34, 129), (204, 85, 34, 255)]] * 2, dtype=numpy.uint8
    )
    expected = target.copy()
    for row in range(2):
        for col in range(3):
            alpha = int(source[row, col, 3])
            for scale in (source_scale, target_scale):
                if scale is not None:
                    alpha = max(0, min(255, round(alpha * scale)))
            expected[row, col] = source_over(
                tuple(int(v) for v in target[row, col]), (204, 85, 34, alpha), mode
            )
    render_blend.composite_blended_group_numpy(target, source, source_scale, target_scale, mode)
    numpy.testing.assert_array_equal(target, expected)
    numpy.testing.assert_array_equal(backing[:, 1::2], 17)
    assert source[0, 1, 3] == 129


@pytest.mark.parametrize("source_scale", [0, 0.5, 1])
@pytest.mark.parametrize("target_scale", [0, 0.5, 1])
@pytest.mark.parametrize("destination_alpha", [0, 128, 255])
def test_normal_group_routes_match_general_compositing(
    source_scale, target_scale, destination_alpha
):
    destination = numpy.array([[(51, 102, 153, destination_alpha)] * 3] * 2, dtype=numpy.uint8)
    source = numpy.array(
        [[(204, 85, 34, 0), (204, 85, 34, 129), (204, 85, 34, 255)]] * 2, dtype=numpy.uint8
    )
    expected = destination.copy()
    render_blend.composite_blended_group_numpy(expected, source, source_scale, target_scale, None)
    composite_normal_group(destination, source, source_scale, target_scale)
    numpy.testing.assert_array_equal(destination, expected)


def test_normal_group_onto_an_empty_backdrop_writes_only_visible_pixels_through_a_view():
    backing = numpy.full((2, 6, 4), 17, dtype=numpy.uint8)
    destination = backing[:, ::2]
    destination[:] = (51, 102, 153, 0)
    source = numpy.array(
        [[(204, 85, 34, 0), (204, 85, 34, 129), (204, 85, 34, 255)]] * 2, dtype=numpy.uint8
    )
    expected = destination.copy()
    render_blend.composite_blended_group_numpy(expected, source, 1.0, 1.0, None)
    composite_normal_group(destination, source, 1.0)
    numpy.testing.assert_array_equal(destination, expected)
    numpy.testing.assert_array_equal(destination[:, 0], [(51, 102, 153, 0)] * 2)
    numpy.testing.assert_array_equal(backing[:, 1::2], 17)


@pytest.mark.parametrize("coverage", [0, 64, 128, 255])
def test_coverage_alpha_is_capped_by_source_opacity_and_zero_coverage_preserves_rgb(coverage):
    target = numpy.array([[(51, 102, 153, 0), (51, 102, 153, 255)]], dtype=numpy.uint8)
    expected = numpy.array(
        [
            [
                source_over((51, 102, 153, alpha), (204, 85, 34, min(coverage, 128)))
                for alpha in (0, 255)
            ]
        ],
        dtype=numpy.uint8,
    )
    blend_normal_alpha_array_numpy(
        target, (204, 85, 34, 128), numpy.full((1, 2), coverage, dtype=numpy.uint8)
    )
    numpy.testing.assert_array_equal(target, expected)


@pytest.mark.parametrize(
    ("color", "opacity"),
    [
        ((0.2,), None),
        ((0.1, 0.5, 0.9), 0.5),
        ((0.1, 0.2, 0.3, 0.4), 1.0),
        ((0.1, 0.2, 0.3, 0.4), 0),
        ((0.3, 0.3, 0.3), False),
    ],
)
def test_cached_colors_convert_as_uncached_ones(color, opacity):
    render_blend.COLOR_RGBA_CACHE.clear()
    first = render_blend.color_rgba(color, opacity)
    assert first == render_blend.convert_color_rgba(color, opacity)
    assert render_blend.color_rgba(color, opacity) is first


def test_color_cache_keeps_opacity_types_apart():
    render_blend.COLOR_RGBA_CACHE.clear()
    assert render_blend.color_rgba((0.5, 0.5, 0.5), False)[3] == 255
    assert render_blend.color_rgba((0.5, 0.5, 0.5), 0)[3] == 0


def test_colors_that_are_not_float_tuples_are_not_cached():
    render_blend.COLOR_RGBA_CACHE.clear()
    assert render_blend.color_rgba([0.5, 0.5, 0.5], 1.0) == render_blend.convert_color_rgba(
        [0.5, 0.5, 0.5], 1.0
    )
    assert render_blend.color_rgba((1, 0, 0), 1.0) == render_blend.convert_color_rgba(
        (1, 0, 0), 1.0
    )
    assert not render_blend.COLOR_RGBA_CACHE
