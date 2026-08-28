from __future__ import annotations

import numpy
import pytest

from core_pdf.impl.spec.s_08_graphics.icc_profiles import (
    IccCurve,
    IccLutProfile,
    IccMatrixProfile,
    IccTransform,
)

# These two were demoted out of the pull-request set for swinging 282ms to 2s on
# unchanged code, on the theory that `numpy.power` in linear_to_srgb was being
# dispatched to CPU-specific SIMD kernels. That was wrong. The other 24
# benchmarks agree to three significant figures across the very same runs, and
# several of them also go through numpy -- a per-CPU dispatch would not have
# spared them.
#
# The cause is OpenBLAS. Both benchmarks reach sgemm (apply_matrix_transform
# does `curves @ matrix`), and CodSpeed counts instructions on a simulated CPU
# that serializes threads, so the thread pool's idle spin-waiting is counted in
# full rather than overlapping with real work. Three iterations of MATRIX.apply
# under callgrind: 766M instructions on one thread, 41.9B on two, 159.4B on
# eight, and 80.1B then 77.7B on two runs of the same binary.
#
# The benchmark job now pins OPENBLAS_NUM_THREADS and OMP_NUM_THREADS to 1,
# which makes these reproducible (18.81ms twice running under the simulator), so
# they are back in the high-impact set. Read any ICC benchmark history from
# before that pinning as noise.
pytestmark = pytest.mark.benchmark_high_impact


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
