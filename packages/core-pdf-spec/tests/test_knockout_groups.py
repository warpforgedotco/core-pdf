# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any, Literal

import numpy
import pytest

from core_pdf_spec.s_11_transparency.groups import composite_knockout_element


def internal_samples() -> dict[str, Any]:
    return {
        "components": numpy.asarray([[0.9, 0.1, 0.1]]),
        "alpha": numpy.asarray([1.0]),
        "backdrop_components": numpy.asarray([[0.5, 0.5, 0.5]]),
        "backdrop_alpha": numpy.asarray([1.0]),
        "element_components": numpy.asarray([[0.375, 0.375, 0.625]]),
        "element_alpha": numpy.asarray([1.0]),
        "shape": numpy.asarray([1.0]),
        "group_alpha": numpy.asarray([0.8]),
        "element_group_alpha": numpy.asarray([0.25]),
    }


def test_full_shape_knocks_out_earlier_color_and_opacity() -> None:
    samples = internal_samples()
    color, alpha, group_alpha = composite_knockout_element(**samples)
    numpy.testing.assert_allclose(color, [[0.375, 0.375, 0.625]])
    numpy.testing.assert_array_equal(alpha, [1.0])
    numpy.testing.assert_array_equal(group_alpha, [0.25])


@pytest.mark.parametrize("shape", [0.0, 0.25, 1.0])
def test_zero_opacity_still_knocks_out_according_to_shape(shape: float) -> None:
    samples = internal_samples()
    samples["shape"][...] = shape
    samples["element_group_alpha"][...] = 0.0
    samples["element_components"][...] = samples["backdrop_components"]
    color, alpha, group_alpha = composite_knockout_element(**samples)
    numpy.testing.assert_allclose(
        color, [[0.9 - 0.4 * shape, 0.1 + 0.4 * shape, 0.1 + 0.4 * shape]]
    )
    numpy.testing.assert_array_equal(alpha, [1.0])
    numpy.testing.assert_allclose(group_alpha, [0.8 * (1.0 - shape)])


def test_fractional_shape_in_isolated_group_preserves_uncovered_previous_element() -> None:
    color, alpha, group_alpha = composite_knockout_element(
        numpy.asarray([[1.0, 0.0, 0.0]]),
        numpy.asarray([0.8]),
        backdrop_components=numpy.asarray([[0.0, 1.0, 0.0]]),
        backdrop_alpha=numpy.asarray([0.0]),
        element_components=numpy.asarray([[0.0, 0.0, 1.0]]),
        element_alpha=numpy.asarray([0.125]),
        shape=numpy.asarray([0.25]),
        group_alpha=numpy.asarray([0.8]),
        element_group_alpha=numpy.asarray([0.125]),
    )
    numpy.testing.assert_allclose(color, [[24.0 / 29.0, 0.0, 5.0 / 29.0]])
    numpy.testing.assert_allclose(alpha, [0.725])
    numpy.testing.assert_allclose(group_alpha, [0.725])


@pytest.mark.parametrize("initial_alpha", [0.0, 0.4, 1.0])
@pytest.mark.parametrize(("shape", "opacity"), [(0.0, 0.0), (0.25, 0.4), (0.75, 1.0), (1.0, 0.0)])
@pytest.mark.parametrize("mode", ["Normal", "Multiply", "Screen"])
def test_knockout_matches_specification_two_stage_shape_average(
    initial_alpha: float,
    shape: float,
    opacity: float,
    mode: Literal["Normal", "Multiply", "Screen"],
) -> None:
    initial = numpy.asarray([[0.2, 0.7, 0.5]])
    previous_source = numpy.asarray([[0.9, 0.1, 0.3]])
    previous_group_alpha = 0.6
    previous_alpha = initial_alpha + (1.0 - initial_alpha) * previous_group_alpha
    previous_premultiplied = (
        previous_group_alpha * previous_source
        + initial_alpha * (1.0 - previous_group_alpha) * initial
    )
    previous = previous_premultiplied / previous_alpha
    source = numpy.asarray([[0.1, 0.4, 0.8]])
    if mode == "Multiply":
        blended = initial * source
    elif mode == "Screen":
        blended = initial + source - initial * source
    else:
        blended = source
    temporary = (1.0 - opacity) * initial_alpha * initial + opacity * (
        (1.0 - initial_alpha) * source + initial_alpha * blended
    )
    expected_premultiplied = (1.0 - shape) * previous_premultiplied + shape * temporary
    expected_group_alpha = (1.0 - shape) * previous_group_alpha + shape * opacity
    expected_alpha = initial_alpha + (1.0 - initial_alpha) * expected_group_alpha
    source_alpha = shape * opacity
    element_alpha = initial_alpha + (1.0 - initial_alpha) * source_alpha
    element_premultiplied = (1.0 - source_alpha) * initial_alpha * initial + source_alpha * (
        (1.0 - initial_alpha) * source + initial_alpha * blended
    )
    element = element_premultiplied / element_alpha if element_alpha else numpy.zeros_like(source)
    color, alpha, group_alpha = composite_knockout_element(
        previous,
        numpy.asarray([previous_alpha]),
        backdrop_components=initial,
        backdrop_alpha=numpy.asarray([initial_alpha]),
        element_components=element,
        element_alpha=numpy.asarray([element_alpha]),
        shape=numpy.asarray([shape]),
        group_alpha=numpy.asarray([previous_group_alpha]),
        element_group_alpha=numpy.asarray([source_alpha]),
    )
    numpy.testing.assert_allclose(color * alpha[..., None], expected_premultiplied, atol=1e-15)
    numpy.testing.assert_allclose(alpha, [expected_alpha])
    numpy.testing.assert_allclose(group_alpha, [expected_group_alpha])


def test_transparent_knockout_erases_isolated_group_without_dividing_by_zero() -> None:
    samples = internal_samples()
    samples["components"][...] = (1.0, 0.0, 0.0)
    samples["alpha"][...] = 0.8
    samples["backdrop_alpha"][...] = 0.0
    samples["element_alpha"][...] = 0.0
    samples["element_group_alpha"][...] = 0.0
    with numpy.errstate(all="raise"):
        color, alpha, group_alpha = composite_knockout_element(**samples)
    numpy.testing.assert_array_equal(color, [[0.0, 0.0, 0.0]])
    numpy.testing.assert_array_equal(alpha, [0.0])
    numpy.testing.assert_array_equal(group_alpha, [0.0])


@pytest.mark.parametrize("color_shape", [(3,), (2, 1), (2, 3, 4), (0, 3)])
def test_knockout_outputs_are_new_arrays_with_color_space_dimensions(
    color_shape: tuple[int, ...],
) -> None:
    samples: dict[str, Any] = {
        name: numpy.full(color_shape if "components" in name else color_shape[:-1], 0.5)
        for name in internal_samples()
    }
    snapshots = {name: array.copy() for name, array in samples.items()}
    outputs = composite_knockout_element(**samples)
    for index, output in enumerate(outputs):
        assert output.dtype == numpy.float64
        assert output.shape == (color_shape if index == 0 else color_shape[:-1])
        output[...] = 0.1
    for name, original in samples.items():
        numpy.testing.assert_array_equal(original, snapshots[name])


def test_knockout_allows_independently_rounded_alpha_without_clipping_output() -> None:
    color, alpha, group_alpha = composite_knockout_element(
        numpy.asarray([[1.0]]),
        numpy.asarray([0.8]),
        backdrop_components=numpy.asarray([[0.0]]),
        backdrop_alpha=numpy.asarray([0.0]),
        element_components=numpy.asarray([[1.0]]),
        element_alpha=numpy.asarray([0.505]),
        shape=numpy.asarray([0.5]),
        group_alpha=numpy.asarray([0.8]),
        element_group_alpha=numpy.asarray([0.5]),
    )
    numpy.testing.assert_allclose(color, [[0.905 / 0.9]])
    assert color[0, 0] > 1.0
    numpy.testing.assert_allclose(alpha, [0.9])
    numpy.testing.assert_allclose(group_alpha, [0.9])


def test_knockout_does_not_clip_negative_color_from_inexact_raster_samples() -> None:
    color, alpha, group_alpha = composite_knockout_element(
        numpy.asarray([[0.001]]),
        numpy.asarray([1.0]),
        backdrop_components=numpy.asarray([[1.0]]),
        backdrop_alpha=numpy.asarray([1.0]),
        element_components=numpy.asarray([[0.4991]]),
        element_alpha=numpy.asarray([1.0]),
        shape=numpy.asarray([0.5]),
        group_alpha=numpy.asarray([1.0]),
        element_group_alpha=numpy.asarray([0.5]),
    )
    numpy.testing.assert_allclose(color, [[-0.0004]])
    numpy.testing.assert_array_equal(alpha, [1.0])
    numpy.testing.assert_array_equal(group_alpha, [1.0])


@pytest.mark.parametrize("name", internal_samples())
@pytest.mark.parametrize("value", [-0.01, 1.01, numpy.nan, numpy.inf, -numpy.inf])
def test_knockout_rejects_nonunit_or_nonfinite_samples(name: str, value: float) -> None:
    samples = internal_samples()
    samples[name][...] = value
    with pytest.raises(ValueError, match="knockout group"):
        composite_knockout_element(**samples)


@pytest.mark.parametrize("name", internal_samples())
def test_knockout_rejects_mismatched_sample_shapes(name: str) -> None:
    samples = internal_samples()
    samples[name] = samples[name][0]
    with pytest.raises(ValueError, match="knockout group"):
        composite_knockout_element(**samples)


@pytest.mark.parametrize("color_shape", [(), (2, 0)])
def test_knockout_requires_a_nonempty_component_axis(color_shape: tuple[int, ...]) -> None:
    samples: dict[str, Any] = {
        name: numpy.full(color_shape if "components" in name else color_shape[:-1], 0.5)
        for name in internal_samples()
    }
    with pytest.raises(ValueError, match="knockout group"):
        composite_knockout_element(**samples)


def test_knockout_rejects_element_alpha_greater_than_shape() -> None:
    samples = internal_samples()
    samples["shape"][...] = 0.2
    with pytest.raises(ValueError, match="knockout group"):
        composite_knockout_element(**samples)
