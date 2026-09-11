"""PDF Lab components and the normalized compatibility interface share XYZ semantics."""

from collections.abc import Callable

import numpy
import pytest

from core_pdf_spec.s_08_graphics.color_math import (
    ColorSamples,
    compensate_black_point_xyz,
    lab_components_to_xyz,
    lab_to_xyz,
    xyz_to_lab_components,
)


@pytest.mark.parametrize("white_point", [(1.0, 1.0, 1.0), (0.9505, 1.0, 1.089)])
def test_neutral_lab_preserves_black_white_and_gray_luminance(
    white_point: tuple[float, float, float],
) -> None:
    # ISO 32000-1, 8.6.5.4, pp. 148–149: neutral Lab scales the supplied white point.
    values = numpy.array([[0, 0, 0], [4, 0, 0], [50, 0, 0], [100, 0, 0]], dtype=numpy.float32)
    luminance = numpy.array([0.0, 108 / 24389, 0.18418651851244416, 1.0])
    expected = luminance[:, None] * numpy.array(white_point)
    result = lab_components_to_xyz(values, white_point)
    numpy.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-8)
    numpy.testing.assert_array_equal(result[0], [0, 0, 0])
    numpy.testing.assert_array_equal(result[-1], numpy.array(white_point, dtype=numpy.float32))


@pytest.mark.parametrize("white_point", [(1.0, 1.0, 1.0), (0.9642, 1.0, 0.8249)])
def test_colored_lab_matches_independent_xyz_vectors(
    white_point: tuple[float, float, float],
) -> None:
    # Forward Lab conversion of XYZ cube roots (.5,.6,.7), (.7,.5,.3), and
    # (.55,.75,.35), using ISO 32000-1 8.6.5.4, gives the Lab inputs below.
    # Fixed XYZ targets check channel order and signs without repeating the inverse.
    values = numpy.array([[53.6, -50, -20], [42, 100, 40], [71, -100, 80]], dtype=numpy.float32)
    expected = numpy.array(
        [[0.125, 0.216, 0.343], [0.343, 0.125, 0.027], [0.166375, 0.421875, 0.042875]]
    ) * numpy.array(white_point)
    numpy.testing.assert_allclose(
        lab_components_to_xyz(values, white_point), expected, rtol=2e-6, atol=1e-8
    )


def test_lab_piecewise_transition_applies_to_each_xyz_component() -> None:
    # ISO 32000-1 8.6.5.4: g changes branches at 6/29. Choosing intermediate
    # components 5/29, 6/29, 7/29 produces exact XYZ numerators 108, 216, 343.
    values = numpy.array(
        [
            [8, -500 / 29, -200 / 29],
            [8, 500 / 29, 200 / 29],
            [4, 500 / 29, -400 / 29],
            [12, -500 / 29, 400 / 29],
        ],
        dtype=numpy.float32,
    )
    expected = numpy.array([[108, 216, 343], [343, 216, 108], [216, 108, 343], [216, 343, 108]])
    numpy.testing.assert_allclose(
        lab_components_to_xyz(values, (1, 1, 1)), expected / 24389, rtol=2e-6, atol=1e-8
    )


def test_normalized_lab_wrapper_preserves_original_float32_results() -> None:
    values = numpy.array(
        [[0, 0.5, 0.5], [1, 1, 0], [0.08, 128 / 255, 128 / 255], [0.5, 0.2, 0.8]],
        dtype=numpy.float32,
    )
    # Captured from the public function before adding the components interface.
    # Bit patterns pin its normalization and arithmetic order, including negatives.
    expected_bits = numpy.array(
        [
            [3103783820, 0, 968314105],
            [1072687578, 1065353216, 1083815531],
            [1007283174, 1007753894, 1008600245],
            [1032526043, 1044159331, 1005182458],
        ],
        dtype=numpy.uint32,
    )
    result = lab_to_xyz(values, (0.9505, 1.0, 1.089))
    numpy.testing.assert_array_equal(result.view(numpy.uint32), expected_bits)


@pytest.mark.parametrize("convert", [lab_components_to_xyz, lab_to_xyz])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "empty"])
def test_lab_conversion_retains_shape_and_leaves_readonly_inputs_untouched(
    convert: Callable[[ColorSamples, tuple[float, float, float]], ColorSamples], layout: str
) -> None:
    backing = numpy.arange(24, dtype=numpy.float32).reshape(4, 6)
    values = backing[::2, ::2]
    if layout == "contiguous":
        values = values.copy()
    elif layout == "empty":
        values = values[:0]
    else:
        assert not values.flags.c_contiguous
    original = values.copy()
    backing_original = backing.copy()
    values.flags.writeable = False
    result = convert(values, (0.9505, 1.0, 1.089))
    assert result.shape == values.shape
    assert result.dtype == numpy.float32
    assert not numpy.shares_memory(result, values)
    numpy.testing.assert_array_equal(values, original)
    numpy.testing.assert_array_equal(backing, backing_original)
    numpy.testing.assert_array_equal(result, convert(original, (0.9505, 1.0, 1.089)))


def test_xyz_inverse_matches_independent_lab_vectors_across_piecewise_transition() -> None:
    # CIE 1976 equations: these XYZ values exercise both the linear and cube-root
    # branches, with unequal channels to detect transposition and sign errors.
    xyz = numpy.array([[0, 0, 0], [108, 216, 343]], dtype=numpy.float64) / 24389
    numpy.testing.assert_allclose(
        xyz_to_lab_components(xyz, (1, 1, 1)),
        [[0, 0, 0], [8, -500 / 29, -200 / 29]],
        atol=1e-12,
    )
    xyz = numpy.array([[0.125, 0.216, 0.343], [0.343, 0.125, 0.027]])
    white = (0.9505, 1.0, 1.089)
    numpy.testing.assert_allclose(
        xyz_to_lab_components(xyz * white, white), [[53.6, -50, -20], [42, 100, 40]], atol=1e-12
    )


def test_black_point_compensation_maps_endpoints_and_preserves_white() -> None:
    # ISO 18619's endpoint stage fixes white and maps source black to destination
    # black. Endpoint detection and adaptation are the caller's responsibility.
    white = (0.9642, 1.0, 0.8249)
    source = (0.04, 0.03, 0.02)
    destination = (0.01, 0.02, 0.03)
    values = numpy.array([source, white, numpy.mean([source, white], axis=0)])
    values.flags.writeable = False
    result = compensate_black_point_xyz(values, white, source, destination)
    numpy.testing.assert_allclose(result[0], destination, atol=1e-15)
    numpy.testing.assert_allclose(result[1], white, atol=1e-15)
    numpy.testing.assert_allclose(result[2], numpy.mean([destination, white], axis=0), atol=1e-15)
    numpy.testing.assert_allclose(
        compensate_black_point_xyz(result, white, destination, source), values, atol=1e-15
    )
    numpy.testing.assert_array_equal(values[0], source)


@pytest.mark.parametrize("endpoint", [(float("nan"), 0, 0), (-1, 0, 0), (1, 0, 0)])
def test_black_point_compensation_rejects_invalid_endpoints(
    endpoint: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError, match="endpoints"):
        compensate_black_point_xyz(numpy.zeros((1, 3)), (1, 1, 1), endpoint, (0, 0, 0))


def test_color_equations_accept_empty_sample_batches() -> None:
    values = numpy.empty((0, 3))
    assert xyz_to_lab_components(values, (1, 1, 1)).shape == (0, 3)
    assert compensate_black_point_xyz(values, (1, 1, 1), (0, 0, 0), (0, 0, 0)).shape == (0, 3)


def test_color_equations_reject_nonfinite_xyz() -> None:
    values = numpy.array([[numpy.nan, 0, 0]])
    with pytest.raises(ValueError, match="XYZ"):
        xyz_to_lab_components(values, (1, 1, 1))
    with pytest.raises(ValueError, match="XYZ"):
        compensate_black_point_xyz(values, (1, 1, 1), (0, 0, 0), (0, 0, 0))
