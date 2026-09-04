from typing import cast

import imagecodecs
import numpy
import pytest

from core_pdf.impl.spec.s_08_graphics.device_profiles import (
    cmyk_bytes_to_srgb,
    cmyk_floats_to_srgb,
    default_cmyk_transform,
)
from core_pdf.impl.spec.s_08_graphics.icc_profiles import (
    INTERNAL_DEDUPLICATE_MIN_ROWS,
    IccProfileError,
    IccSampleError,
    IccTransform,
    parse_icc_transform,
)


def srgb_transform() -> IccTransform:
    return parse_icc_transform(imagecodecs.cms_profile("srgb"))


def gray_transform() -> IccTransform:
    # The built-in grey profile carries no tone curve, and lcms cannot build a
    # transform from one; a gamma makes it the shaper profile a PDF would embed.
    return parse_icc_transform(imagecodecs.cms_profile("gray", gamma=2.2))


def test_parser_reports_the_profile_colour_space() -> None:
    transform = srgb_transform()

    assert transform.color_space == "RGB"
    assert transform.pcs == "XYZ"
    assert transform.input_channels == 3
    assert transform.output_channels == 3
    assert transform.alternate_color_space == "DeviceRGB"


def test_parser_rejects_invalid_profile() -> None:
    with pytest.raises(IccProfileError, match="ICC"):
        parse_icc_transform(b"not an ICC profile")


def test_matrix_transform_accepts_float32_batches() -> None:
    transform = srgb_transform()
    samples = numpy.asarray([[0.0, 0.5, 1.0], [1.0, 0.0, 0.25]], dtype=numpy.float32)

    result = transform.apply(samples)

    assert result.shape == (2, 3)
    assert result.dtype == numpy.float32
    # sRGB in, sRGB out: the transform is the identity up to lcms' rounding.
    numpy.testing.assert_allclose(result, samples, atol=2e-3)


def test_gray_transform_expands_to_rgb() -> None:
    transform = gray_transform()

    assert transform.color_space == "GRAY"
    assert transform.input_channels == 1
    assert transform.alternate_color_space == "DeviceGray"

    result = transform.apply_uint8(numpy.asarray([[0], [128], [255]], dtype=numpy.uint8))

    assert result.shape == (3, 3)
    numpy.testing.assert_array_equal(result[0], (0, 0, 0))
    numpy.testing.assert_array_equal(result[2], (255, 255, 255))
    # Grey stays neutral, and the gamma keeps mid-grey near where it started.
    assert result[1][0] == result[1][1] == result[1][2]
    assert 120 <= result[1][0] <= 140


@pytest.mark.parametrize(
    "samples",
    [
        numpy.zeros(3, dtype=numpy.float32),
        numpy.zeros((1, 3), dtype=numpy.float64),
        numpy.zeros((1, 2), dtype=numpy.float32),
        numpy.zeros((2, 6), dtype=numpy.float32)[:, ::2],
    ],
)
def test_transform_rejects_invalid_sample_contract(samples: numpy.ndarray) -> None:
    with pytest.raises(IccSampleError):
        srgb_transform().apply(samples)


def test_uint8_transform_rejects_the_wrong_dtype_or_shape() -> None:
    transform = srgb_transform()
    with pytest.raises(IccSampleError):
        transform.apply_uint8(cast(numpy.ndarray, numpy.zeros((1, 3), dtype=numpy.float32)))
    with pytest.raises(IccSampleError):
        transform.apply_uint8(numpy.zeros((1, 4), dtype=numpy.uint8))


def test_uint8_transform_agrees_with_the_float_path() -> None:
    transform = default_cmyk_transform()
    assert transform is not None
    values = numpy.asarray(
        [[0, 0, 0, 0], [255, 255, 255, 255], [64, 128, 192, 32], [7, 200, 33, 90]],
        dtype=numpy.uint8,
    )
    floats = numpy.ascontiguousarray(values.astype(numpy.float32) / 255.0)

    expected = numpy.rint(transform.apply(floats) * 255.0)

    # The two paths differ only in where lcms rounds, so allow a single step.
    assert numpy.abs(transform.apply_uint8(values).astype(int) - expected).max() <= 1


def test_uint8_transform_deduplicates_without_changing_results() -> None:
    transform = default_cmyk_transform()
    assert transform is not None
    palette = numpy.asarray(
        [[0, 0, 0, 0], [255, 0, 0, 0], [0, 255, 0, 0], [0, 0, 255, 0]],
        dtype=numpy.uint8,
    )
    # Well over the row floor and far under the distinct-ratio floor, so this
    # batch takes the deduplicating path while the palette itself does not.
    rows = INTERNAL_DEDUPLICATE_MIN_ROWS * 4
    batch = numpy.ascontiguousarray(numpy.resize(palette, (rows, 4)))

    result = transform.apply_uint8(batch)

    assert result.shape == (rows, 3)
    numpy.testing.assert_array_equal(result[: len(palette)], transform.apply_uint8(palette))
    numpy.testing.assert_array_equal(result[-len(palette) :], transform.apply_uint8(palette))


def test_empty_batches_keep_their_shape() -> None:
    transform = srgb_transform()

    assert transform.apply_uint8(numpy.zeros((0, 3), dtype=numpy.uint8)).shape == (0, 3)
    assert transform.apply(numpy.zeros((0, 3), dtype=numpy.float32)).shape == (0, 3)


def test_default_cmyk_profile_loads() -> None:
    transform = default_cmyk_transform()

    assert transform is not None
    assert transform.color_space == "CMYK"
    assert transform.pcs == "Lab"
    assert transform.input_channels == 4
    assert transform.alternate_color_space == "DeviceCMYK"


def test_default_cmyk_profile_reproduces_the_process_inks() -> None:
    transform = default_cmyk_transform()
    assert transform is not None
    inks = numpy.asarray(
        [[0, 0, 0, 0], [255, 0, 0, 0], [0, 255, 0, 0], [0, 0, 255, 0], [0, 0, 0, 255]],
        dtype=numpy.uint8,
    )

    white, cyan, magenta, yellow, black = transform.apply_uint8(inks)

    numpy.testing.assert_array_equal(white, (255, 255, 255))
    assert cyan[2] > cyan[1] > cyan[0]
    assert magenta[0] > magenta[2] > magenta[1]
    assert yellow[0] > yellow[1] > yellow[2]
    # Pinned against the converter this replaced, which produced (0, 174, 240),
    # (236, 10, 141), (255, 243, 0) and (41, 39, 40) for the same four inks.
    for measured, previous in (
        (cyan, (0, 174, 240)),
        (magenta, (236, 10, 141)),
        (yellow, (255, 243, 0)),
        (black, (41, 39, 40)),
    ):
        assert numpy.abs(measured.astype(int) - numpy.asarray(previous)).max() <= 2


def test_cmyk_helpers_agree_on_one_colour() -> None:
    batch = cmyk_bytes_to_srgb(numpy.asarray([[26, 51, 77, 102]], dtype=numpy.uint8))[0]

    single = cmyk_floats_to_srgb(26 / 255.0, 51 / 255.0, 77 / 255.0, 102 / 255.0)

    assert single == tuple(int(component) for component in batch)
