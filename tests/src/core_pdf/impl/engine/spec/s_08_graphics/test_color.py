import numpy
import pytest

from core_pdf.impl.engine.spec.s_08_graphics import (
    IccSampleError,
    IccTransform,
    parse_icc_transform,
)
from core_pdf.impl.engine.spec.s_08_graphics.icc_profiles import (
    IccCurve,
    IccLutProfile,
    IccMatrixProfile,
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
    assert numpy.all(result >= 0.0)
    assert numpy.all(result <= 1.0)


def test_gray_transform_expands_to_rgb() -> None:
    transform = IccTransform(matrix_profile("GRAY"))
    result = transform.apply(numpy.asarray([[0.0], [1.0]], dtype=numpy.float32))

    assert result.shape == (2, 3)
    assert numpy.allclose(result[0], 0.0, atol=1e-5)
    assert numpy.allclose(result[1], 1.0, atol=2e-3)


def test_lut_transform_is_vectorized() -> None:
    transform = IccTransform(lut_profile())
    samples = numpy.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=numpy.float32)

    result = transform.apply(samples)

    assert result.shape == (2, 3)
    assert numpy.all(numpy.isfinite(result))


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
