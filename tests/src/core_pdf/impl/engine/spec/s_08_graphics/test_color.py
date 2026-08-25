from dataclasses import replace
from typing import cast

import numpy
import pytest

from core_pdf.impl.engine.spec.s_08_graphics import (
    IccSampleError,
    IccTransform,
    parse_icc_transform,
)
from core_pdf.impl.engine.spec.s_08_graphics.device_profiles import (
    default_cmyk_transform,
)
from core_pdf.impl.engine.spec.s_08_graphics.icc_profiles import (
    INTERNAL_D50_WHITE,
    INTERNAL_TRANSFORM_BLOCK_ROWS,
    IccCurve,
    IccLutProfile,
    IccMatrixProfile,
    internal_compensate_black_point,
    select_icc_lut_tag,
)


def matrix_profile(color_space: str = "RGB") -> IccMatrixProfile:
    channels = 1 if color_space == "GRAY" else 3
    matrix = (
        ((1.0, 0.0, 0.0),) if channels == 1 else ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    return IccMatrixProfile(
        color_space=color_space,
        pcs="XYZ",
        white_point=(0.9642, 1.0, 0.8249),
        matrix=matrix,
        curves=tuple(IccCurve("identity", ()) for _ in range(channels)),
    )


def lut_profile() -> IccLutProfile:
    clut = tuple(tuple(float((index >> bit) & 1) for bit in range(3)) for index in range(8))
    return IccLutProfile(
        color_space="RGB",
        pcs="XYZ",
        input_channels=3,
        output_channels=3,
        grid_points=2,
        matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        input_tables=((0.0, 1.0),) * 3,
        clut=clut,
        output_tables=((0.0, 1.0),) * 3,
    )


def test_matrix_transform_accepts_float32_batches() -> None:
    transform = IccTransform(matrix_profile())
    samples = numpy.asarray([[0.0, 0.5, 1.0], [1.0, 0.0, 0.25]], dtype=numpy.float32)

    result = transform.apply(samples)

    assert result.shape == (2, 3)
    assert result.dtype == numpy.float32
    assert result.flags.c_contiguous
    numpy.testing.assert_allclose(
        result,
        ((0.0, 0.99651, 1.0), (1.0, 0.0, 0.68244)),
        atol=2e-5,
    )


def test_gray_transform_expands_to_rgb() -> None:
    transform = IccTransform(matrix_profile("GRAY"))
    result = transform.apply(numpy.asarray([[0.0], [1.0]], dtype=numpy.float32))

    assert result.shape == (2, 3)
    assert numpy.allclose(result[0], 0.0, atol=1e-5)
    assert numpy.allclose(result[1], 1.0, atol=2e-3)


def test_lut_transform_interpolates_a_batch() -> None:
    transform = IccTransform(lut_profile())
    samples = numpy.asarray(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.25, 0.5, 0.75]],
        dtype=numpy.float32,
    )

    result = transform.apply(samples)

    numpy.testing.assert_allclose(
        result,
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.98733, 1.0),
            (1.0, 0.51948, 0.57567),
        ),
        atol=2e-5,
    )


@pytest.mark.parametrize(
    "samples",
    [
        numpy.zeros(3, dtype=numpy.float32),
        numpy.zeros((1, 3), dtype=numpy.float64),
        numpy.zeros((1, 2), dtype=numpy.float32),
    ],
)
def test_transform_rejects_invalid_sample_contract(samples: numpy.ndarray) -> None:
    with pytest.raises(IccSampleError):
        IccTransform(matrix_profile()).apply(samples)


def test_parser_rejects_invalid_profile() -> None:
    with pytest.raises(ValueError, match="ICC"):
        parse_icc_transform(b"not an ICC profile")


def test_uint8_transform_matches_the_float_path_exactly() -> None:
    transform = IccTransform(lut_profile())
    values = numpy.asarray(
        [[0, 0, 0], [255, 255, 255], [64, 128, 192], [7, 200, 33]],
        dtype=numpy.uint8,
    )
    floats = numpy.ascontiguousarray(values.astype(numpy.float32) / 255.0)

    expected = numpy.rint(numpy.clip(transform.apply(floats) * 255.0, 0.0, 255.0))

    numpy.testing.assert_array_equal(transform.apply_uint8(values), expected)


def test_uint8_transform_spans_more_than_one_block() -> None:
    transform = IccTransform(lut_profile())
    rows = INTERNAL_TRANSFORM_BLOCK_ROWS + 5
    values = numpy.zeros((rows, 3), dtype=numpy.uint8)
    values[-1] = 255

    result = transform.apply_uint8(values)

    assert result.shape == (rows, 3)
    numpy.testing.assert_array_equal(result[0], transform.apply_uint8(values[:1])[0])
    numpy.testing.assert_array_equal(result[-1], transform.apply_uint8(values[-1:])[0])


def test_uint8_transform_rejects_the_wrong_dtype_or_shape() -> None:
    transform = IccTransform(lut_profile())
    with pytest.raises(IccSampleError):
        # Deliberately the wrong dtype: the guard is a runtime contract, since
        # a float array reaching the byte gather would index out of range.
        transform.apply_uint8(cast(numpy.ndarray, numpy.zeros((1, 3), dtype=numpy.float32)))
    with pytest.raises(IccSampleError):
        transform.apply_uint8(numpy.zeros((1, 4), dtype=numpy.uint8))


def test_default_cmyk_profile_loads_with_a_detected_black_point() -> None:
    transform = default_cmyk_transform()

    assert transform is not None
    assert transform.color_space == "CMYK"
    assert transform.input_channels == 4
    profile = transform.profile
    assert isinstance(profile, IccLutProfile)
    # Lab PCS in an ICC v2 lut16 tag is the legacy encoding, and the profile
    # carries a B2A table, so black point compensation has a source black.
    assert profile.pcs == "Lab"
    assert profile.legacy_lab is True
    assert profile.black_point is not None
    assert all(0.0 < component < 0.1 for component in profile.black_point)


def test_default_cmyk_profile_reproduces_the_process_inks() -> None:
    transform = default_cmyk_transform()
    assert transform is not None
    inks = numpy.asarray(
        [[0, 0, 0, 0], [255, 0, 0, 0], [0, 255, 0, 0], [0, 0, 255, 0]],
        dtype=numpy.uint8,
    )

    white, cyan, magenta, yellow = transform.apply_uint8(inks)

    numpy.testing.assert_array_equal(white, (255, 255, 255))
    assert cyan[2] > cyan[1] > cyan[0]
    assert magenta[0] > magenta[2] > magenta[1]
    assert yellow[0] > yellow[1] > yellow[2]


def test_legacy_lab_scaling_puts_pcs_white_at_full_white() -> None:
    """Without the 0xFF00 correction, L* = 100 decodes as 99.6 and white greys."""
    profile = IccLutProfile(
        color_space="CMYK",
        pcs="Lab",
        input_channels=4,
        output_channels=3,
        grid_points=2,
        matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        input_tables=((0.0, 1.0),) * 4,
        # Every grid node is legacy-encoded L* = 100, a* = b* = 0.
        clut=((65280.0 / 65535.0, 32768.0 / 65535.0, 32768.0 / 65535.0),) * 16,
        output_tables=((0.0, 1.0),) * 3,
    )
    samples = numpy.zeros((1, 4), dtype=numpy.float32)

    unscaled = IccTransform(profile).apply(samples)
    scaled = IccTransform(replace(profile, legacy_lab=True)).apply(samples)

    # 99.6 of 100 in L* costs whole 8-bit steps of grey once it reaches sRGB.
    assert numpy.rint(unscaled * 255.0).min() < 255.0
    numpy.testing.assert_allclose(scaled, 1.0, atol=1e-3)
    assert numpy.rint(scaled * 255.0).min() == 255.0


def test_black_point_compensation_maps_the_source_black_to_zero() -> None:
    black = (0.02, 0.021, 0.018)
    xyz = numpy.asarray([list(black), list(INTERNAL_D50_WHITE)], dtype=numpy.float32)

    result = internal_compensate_black_point(xyz, black)

    numpy.testing.assert_allclose(result[0], 0.0, atol=1e-6)
    numpy.testing.assert_allclose(result[1], INTERNAL_D50_WHITE, rtol=1e-5)


def test_lut_tag_selection_prefers_relative_colorimetric() -> None:
    tags = {b"A2B0": b"perceptual", b"A2B1": b"relative", b"A2B2": b"saturation"}

    assert select_icc_lut_tag(tags, b"A2B") == b"relative"
    assert select_icc_lut_tag({b"A2B0": b"perceptual"}, b"A2B") == b"perceptual"
    assert select_icc_lut_tag(tags, b"B2A") is None
