from __future__ import annotations

import numpy

from core_pdf.impl.engine.spec.s_08_graphics import IccTransform
from core_pdf.impl.engine.spec.s_08_graphics.icc_profiles import (
    IccCurve,
    IccLutProfile,
    IccMatrixProfile,
)


def matrix_transform() -> IccTransform:
    profile = IccMatrixProfile(
        color_space="RGB",
        pcs="XYZ",
        white_point=(0.9642, 1.0, 0.8249),
        matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        curves=(IccCurve("gamma", (2.2,)),) * 3,
    )
    return IccTransform(profile)


def lut_transform() -> IccTransform:
    profile = IccLutProfile(
        color_space="RGB",
        pcs="XYZ",
        input_channels=3,
        output_channels=3,
        grid_points=2,
        matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        input_tables=((0.0, 1.0),) * 3,
        clut=tuple(tuple(float((index >> bit) & 1) for bit in range(3)) for index in range(8)),
        output_tables=((0.0, 1.0),) * 3,
    )
    return IccTransform(profile)


MATRIX = matrix_transform()
LUT = lut_transform()
SAMPLES = numpy.random.default_rng(42).random((65_536, 3), dtype=numpy.float32)


def test_matrix_transform_benchmark(benchmark) -> None:
    result = benchmark(MATRIX.apply, SAMPLES)
    assert result.shape == SAMPLES.shape


def test_lut_transform_benchmark(benchmark) -> None:
    result = benchmark(LUT.apply, SAMPLES)
    assert result.shape == SAMPLES.shape
