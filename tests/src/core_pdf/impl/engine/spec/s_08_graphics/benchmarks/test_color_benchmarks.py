from __future__ import annotations

import numpy

from core_pdf.impl.engine.spec.s_08_graphics import IccTransform
from core_pdf.impl.engine.spec.s_08_graphics.icc_profiles import (
    IccCurve,
    IccLutProfile,
    IccMatrixProfile,
)

# Deliberately NOT benchmark_high_impact, so these stay out of the pull-request
# CodSpeed run. Both benchmarks push 65,536x3 float32 through IccTransform.apply,
# and profiling that payload puts almost all of it in numpy elementwise work --
# linear_to_srgb dominates, and its hot operation is
# `numpy.power(clipped, 1.0 / 2.4)`, a transcendental ufunc that numpy
# runtime-dispatches to CPU-specific SIMD kernels.
#
# CodSpeed measures here in Simulation mode, counting instructions under
# Valgrind. Different SIMD kernels are genuinely different instruction streams,
# so on a heterogeneous runner pool these numbers move with the CPU that
# happened to pick up the job rather than with any change to core-pdf.
# Observed: 2.9s / 2.0s / 1.2s / 0.93s across four runs on code that never
# touched ICC, and once a "x3.1 improvement" attributed to a tables-only change.
#
# This is not flaky measurement to be re-run until it settles; the benchmark is
# faithfully measuring code that differs per machine. Re-running cannot fix it
# and neither can a tolerance. They still run in the weekly full sweep, where a
# human reads CodSpeed's "Environment Differences / CPU" block alongside them.


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
