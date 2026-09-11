# SPDX-License-Identifier: AGPL-3.0-only

from typing import Literal

import numpy
import pytest

from core_pdf_spec.s_11_transparency.groups import remove_group_backdrop


def internal_composite(
    backdrop: numpy.ndarray,
    backdrop_alpha: numpy.ndarray,
    source: numpy.ndarray,
    source_alpha: numpy.ndarray,
    mode: Literal["Normal", "Multiply", "Screen"] = "Normal",
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Independent three-contribution form of ISO 32000-1/2 11.3.3."""
    if mode == "Multiply":
        blended = backdrop * source
    elif mode == "Screen":
        blended = backdrop + source - backdrop * source
    else:
        blended = source
    alpha = backdrop_alpha + source_alpha - backdrop_alpha * source_alpha
    color = (
        ((1.0 - source_alpha) * backdrop_alpha)[..., None] * backdrop
        + ((1.0 - backdrop_alpha) * source_alpha)[..., None] * source
        + (backdrop_alpha * source_alpha)[..., None] * blended
    )
    numpy.divide(color, alpha[..., None], out=color, where=alpha[..., None] != 0)
    return color, alpha


def test_group_backdrop_removal_recovers_single_normal_source() -> None:
    # Cn = 0.75 * gray + 0.25 * red over an opaque backdrop.
    color, alpha = remove_group_backdrop(
        numpy.asarray([[0.625, 0.375, 0.375]]),
        numpy.asarray([1.0]),
        numpy.asarray([[0.5, 0.5, 0.5]]),
        numpy.asarray([1.0]),
        numpy.asarray([0.25]),
    )
    numpy.testing.assert_allclose(color, [[1.0, 0.0, 0.0]])
    numpy.testing.assert_array_equal(alpha, [0.25])


def test_group_source_retains_inner_blend_interaction_with_initial_backdrop() -> None:
    backdrop = numpy.asarray([[0.8, 0.3, 0.1]])
    initial = numpy.asarray([0.4])
    source_alpha = numpy.asarray([0.25])
    rendered, complete = internal_composite(
        backdrop, initial, numpy.asarray([[0.25, 0.5, 0.9]]), source_alpha, "Multiply"
    )
    color, alpha = remove_group_backdrop(rendered, complete, backdrop, initial, source_alpha)
    # The source must preserve (1 - alpha0) * Cs + alpha0 * Multiply(C0, Cs).
    numpy.testing.assert_allclose(color, [[0.23, 0.36, 0.576]])
    numpy.testing.assert_array_equal(alpha, source_alpha)


@pytest.mark.parametrize("initial_alpha", [0.0, 0.2, 0.75, 1.0])
@pytest.mark.parametrize("mode", ["Normal", "Multiply", "Screen"])
def test_nonisolated_group_matches_ungrouped_painting(
    initial_alpha: float, mode: Literal["Normal", "Multiply", "Screen"]
) -> None:
    # ISO 32000-1/2 11.4.4 Note 5: grouping with outer Normal/full opacity
    # does not change the appearance, including internal non-Normal blends.
    backdrop = numpy.asarray([[0.15, 0.75, 0.4]])
    initial = numpy.asarray([initial_alpha])
    first, first_alpha = internal_composite(
        backdrop, initial, numpy.asarray([[1.0, 0.1, 0.3]]), numpy.asarray([0.4]), mode
    )
    rendered, complete = internal_composite(
        first, first_alpha, numpy.asarray([[0.2, 0.5, 0.9]]), numpy.asarray([0.5]), mode
    )
    # Source-only alpha is 0.4 + (1 - 0.4) * 0.5, independently of alpha0.
    color, alpha = remove_group_backdrop(
        rendered, complete, backdrop, initial, numpy.asarray([0.7])
    )
    recomposed, recomposed_alpha = internal_composite(backdrop, initial, color, alpha)
    numpy.testing.assert_allclose(recomposed, rendered)
    numpy.testing.assert_allclose(recomposed_alpha, complete)


@pytest.mark.parametrize("initial_alpha", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("opacity", [0.0, 0.25, 1.0])
def test_outer_opacity_applies_once_after_multiple_group_elements(
    initial_alpha: float, opacity: float
) -> None:
    backdrop = numpy.asarray([[0.2, 0.6, 0.8]])
    initial = numpy.asarray([initial_alpha])
    first, first_alpha = internal_composite(
        backdrop, initial, numpy.asarray([[1.0, 0.0, 0.0]]), numpy.asarray([0.5])
    )
    rendered, complete = internal_composite(
        first,
        first_alpha,
        numpy.asarray([[0.0, 1.0, 0.0]]),
        numpy.asarray([0.5]),
        "Multiply",
    )
    color, alpha = remove_group_backdrop(
        rendered, complete, backdrop, initial, numpy.asarray([0.75])
    )
    actual, actual_alpha = internal_composite(backdrop, initial, color, alpha * opacity)
    # Normal outer opacity interpolates the premultiplied before/after result.
    expected_premultiplied = (
        opacity * rendered * complete[..., None] + (1.0 - opacity) * backdrop * initial[..., None]
    )
    numpy.testing.assert_allclose(actual * actual_alpha[..., None], expected_premultiplied)
    numpy.testing.assert_allclose(actual_alpha, opacity * complete + (1.0 - opacity) * initial)


def test_outer_blend_uses_removed_source_and_only_group_alpha() -> None:
    # The two marks yield RGB (0.3, 0.3, 0.2), including an opaque backdrop.
    backdrop = numpy.asarray([[0.2, 0.6, 0.8]])
    initial = numpy.asarray([1.0])
    color, alpha = remove_group_backdrop(
        numpy.asarray([[0.3, 0.3, 0.2]]),
        numpy.asarray([1.0]),
        backdrop,
        initial,
        numpy.asarray([0.75]),
    )
    # Source color = (1/3, 0.2, 0); Screen = (7/15, 0.68, 0.8).
    actual, actual_alpha = internal_composite(backdrop, initial, color, alpha * 0.4, "Screen")
    numpy.testing.assert_allclose(actual, [[0.28, 0.624, 0.8]])
    numpy.testing.assert_array_equal(actual_alpha, [1.0])


@pytest.mark.parametrize("initial_alpha", [0.0, 0.5, 1.0])
def test_nested_nonisolated_groups_remove_their_own_initial_backdrop(
    initial_alpha: float,
) -> None:
    backdrop = numpy.asarray([[0.2, 0.4, 0.7]])
    initial = numpy.asarray([initial_alpha])
    first, first_alpha = internal_composite(
        backdrop, initial, numpy.asarray([[0.6, 0.2, 0.5]]), numpy.asarray([0.2]), "Screen"
    )
    nested, nested_complete = internal_composite(
        first, first_alpha, numpy.asarray([[0.1, 0.9, 0.3]]), numpy.asarray([0.4]), "Multiply"
    )
    nested_color, nested_alpha = remove_group_backdrop(
        nested, nested_complete, first, first_alpha, numpy.asarray([0.4])
    )
    nested_output, nested_output_alpha = internal_composite(
        first, first_alpha, nested_color, nested_alpha * 0.5
    )
    # Parent tracks the nested group's returned alpha times outer opacity.
    parent_color, parent_alpha = remove_group_backdrop(
        nested_output, nested_output_alpha, backdrop, initial, numpy.asarray([0.36])
    )
    recomposed, recomposed_alpha = internal_composite(backdrop, initial, parent_color, parent_alpha)
    numpy.testing.assert_allclose(recomposed, nested_output)
    numpy.testing.assert_allclose(recomposed_alpha, nested_output_alpha)


def test_transparent_initial_backdrop_has_no_color_contribution() -> None:
    source = numpy.asarray([[0.25, 0.5, 0.75], [0.1, 0.2, 0.3]])
    alpha = numpy.asarray([0.5, 1.0])
    actual, actual_alpha = remove_group_backdrop(
        source, alpha, numpy.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), numpy.zeros(2), alpha
    )
    numpy.testing.assert_array_equal(actual, source)
    numpy.testing.assert_array_equal(actual_alpha, alpha)


@pytest.mark.parametrize("initial_alpha", [0.0, 0.5, 1.0])
def test_empty_group_is_transparent_without_division(initial_alpha: float) -> None:
    backdrop = numpy.asarray([[0.25, 0.5, 0.75]])
    initial = numpy.asarray([initial_alpha])
    with numpy.errstate(all="raise"):
        color, alpha = remove_group_backdrop(
            backdrop, initial, backdrop, initial, numpy.asarray([0.0])
        )
    numpy.testing.assert_array_equal(color, [[0.0, 0.0, 0.0]])
    numpy.testing.assert_array_equal(alpha, [0.0])


@pytest.mark.parametrize("shape", [(3,), (4, 1), (2, 3, 4), (0, 3)])
def test_group_samples_accept_color_channels_and_do_not_mutate_inputs(
    shape: tuple[int, ...],
) -> None:
    color = numpy.full(shape, 0.5)
    backdrop = numpy.zeros(shape)
    complete = numpy.ones(shape[:-1])
    initial = numpy.ones(shape[:-1])
    accumulated = numpy.full(shape[:-1], 0.5)
    actual, alpha = remove_group_backdrop(color, complete, backdrop, initial, accumulated)
    assert actual.dtype == numpy.float64
    assert alpha.dtype == numpy.float64
    assert actual.shape == color.shape
    assert alpha.shape == complete.shape
    numpy.testing.assert_array_equal(actual, numpy.ones(shape))
    actual[...] = 0.1
    alpha[...] = 0.2
    numpy.testing.assert_array_equal(color, numpy.full(shape, 0.5))
    numpy.testing.assert_array_equal(backdrop, numpy.zeros(shape))
    numpy.testing.assert_array_equal(complete, numpy.ones(shape[:-1]))
    numpy.testing.assert_array_equal(initial, numpy.ones(shape[:-1]))
    numpy.testing.assert_array_equal(accumulated, numpy.full(shape[:-1], 0.5))


def test_group_removal_does_not_clip_or_quantize_inexact_raster_inputs() -> None:
    color, alpha = remove_group_backdrop(
        numpy.asarray([[0.499, 0.751, 0.623]]),
        numpy.asarray([0.751]),
        numpy.asarray([[1.0, 0.0, 0.5]]),
        numpy.asarray([0.5]),
        numpy.asarray([0.5]),
    )
    numpy.testing.assert_allclose(color, [[0.249498, 1.128002, 0.685746]])
    numpy.testing.assert_array_equal(alpha, [0.5])


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("value", [-0.01, 1.01, numpy.nan, numpy.inf, -numpy.inf])
def test_group_removal_rejects_nonunit_or_nonfinite_inputs(index: int, value: float) -> None:
    inputs = [
        numpy.asarray([[0.5, 0.5, 0.5]]),
        numpy.asarray([1.0]),
        numpy.asarray([[0.5, 0.5, 0.5]]),
        numpy.asarray([1.0]),
        numpy.asarray([0.5]),
    ]
    inputs[index][...] = value
    with pytest.raises(ValueError, match="transparency group"):
        remove_group_backdrop(*inputs)


@pytest.mark.parametrize("index", range(5))
def test_group_removal_rejects_mismatched_sample_shapes(index: int) -> None:
    inputs = [numpy.full((2, 3), 0.5) if i in {0, 2} else numpy.ones(2) for i in range(5)]
    inputs[index] = inputs[index][0]
    with pytest.raises(ValueError, match="transparency group"):
        remove_group_backdrop(*inputs)


@pytest.mark.parametrize("shape", [(), (2, 0)])
def test_group_removal_requires_a_nonempty_component_axis(shape: tuple[int, ...]) -> None:
    color = numpy.zeros(shape)
    alpha = numpy.zeros(shape[:-1])
    with pytest.raises(ValueError, match="transparency group"):
        remove_group_backdrop(color, alpha, color, alpha, alpha)
