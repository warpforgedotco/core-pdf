import numpy as np
import pytest

from core_pdf.impl.render import blend
from core_pdf_cythonized import blend_normal_alpha_array_numpy


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
    backing = np.full((3, 5, 4), 17, dtype=np.uint8)
    target = backing[:, 1:4:2]
    destination = (51, 102, 153, destination_alpha)
    target[:] = destination
    source = (204, 85, 34, source_alpha)
    blend.blend_solid_array_numpy(target, source, mode)
    expected = source_over(destination, source, mode)
    np.testing.assert_array_equal(target, np.broadcast_to(expected, target.shape))
    np.testing.assert_array_equal(backing[:, ::2], 17)


@pytest.mark.parametrize("source_alpha", [0, 64, 128, 255])
@pytest.mark.parametrize("destination_alpha", [0, 128, 255])
def test_normal_fast_path_matches_source_over_on_strided_views(source_alpha, destination_alpha):
    backing = np.full((3, 5, 4), 17, dtype=np.uint8)
    target = backing[:, 1:4:2]
    destination = (51, 102, 153, destination_alpha)
    target[:] = destination
    source = (204, 85, 34, source_alpha)
    blend.blend_normal_solid_array_numpy(target, source)
    np.testing.assert_array_equal(
        target, np.broadcast_to(source_over(destination, source), target.shape)
    )
    np.testing.assert_array_equal(backing[:, ::2], 17)


@pytest.mark.parametrize("mode", [None, "Multiply", "Screen"])
@pytest.mark.parametrize(
    ("source_scale", "target_scale"), [(None, None), (0.5, 0.5), (0, 1), (1, 0), (1, 1)]
)
def test_group_blending_rounds_each_alpha_stage_and_preserves_unpainted_pixels(
    mode, source_scale, target_scale
):
    backing = np.full((2, 6, 4), 17, dtype=np.uint8)
    target = backing[:, ::2]
    target[:] = [(51, 102, 153, 0), (51, 102, 153, 128), (51, 102, 153, 255)]
    source = np.array(
        [[(204, 85, 34, 0), (204, 85, 34, 129), (204, 85, 34, 255)]] * 2, dtype=np.uint8
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
    blend.composite_blended_group_numpy(target, source, source_scale, target_scale, mode)
    np.testing.assert_array_equal(target, expected)
    np.testing.assert_array_equal(backing[:, 1::2], 17)
    assert source[0, 1, 3] == 129


@pytest.mark.parametrize("source_scale", [0, 0.5, 1])
@pytest.mark.parametrize("target_scale", [0, 0.5, 1])
@pytest.mark.parametrize("destination_alpha", [0, 128, 255])
def test_normal_group_routes_match_general_compositing(
    source_scale, target_scale, destination_alpha
):
    destination = np.array([[(51, 102, 153, destination_alpha)] * 3] * 2, dtype=np.uint8)
    source = np.array(
        [[(204, 85, 34, 0), (204, 85, 34, 129), (204, 85, 34, 255)]] * 2, dtype=np.uint8
    )
    expected = destination.copy()
    blend.composite_blended_group_numpy(expected, source, source_scale, target_scale, None)
    blend.composite_normal_group_numpy(destination, source, source_scale, target_scale)
    np.testing.assert_array_equal(destination, expected)


def test_normal_group_onto_an_empty_backdrop_writes_only_visible_pixels_through_a_view():
    # The destination is a strided view into a larger buffer, as a group's
    # window into the page is; unpainted pixels keep their colour bytes.
    backing = np.full((2, 6, 4), 17, dtype=np.uint8)
    destination = backing[:, ::2]
    destination[:] = (51, 102, 153, 0)
    source = np.array(
        [[(204, 85, 34, 0), (204, 85, 34, 129), (204, 85, 34, 255)]] * 2, dtype=np.uint8
    )
    expected = destination.copy()
    blend.composite_blended_group_numpy(expected, source, 1.0, 1.0, None)
    blend.composite_normal_group_numpy(destination, source, 1.0)
    np.testing.assert_array_equal(destination, expected)
    np.testing.assert_array_equal(destination[:, 0], [(51, 102, 153, 0)] * 2)
    np.testing.assert_array_equal(backing[:, 1::2], 17)


@pytest.mark.parametrize("coverage", [0, 64, 128, 255])
def test_coverage_alpha_is_capped_by_source_opacity_and_zero_coverage_preserves_rgb(coverage):
    target = np.array([[(51, 102, 153, 0), (51, 102, 153, 255)]], dtype=np.uint8)
    expected = np.array(
        [
            [
                source_over((51, 102, 153, alpha), (204, 85, 34, min(coverage, 128)))
                for alpha in (0, 255)
            ]
        ],
        dtype=np.uint8,
    )
    blend_normal_alpha_array_numpy(
        target, (204, 85, 34, 128), np.full((1, 2), coverage, dtype=np.uint8)
    )
    np.testing.assert_array_equal(target, expected)
