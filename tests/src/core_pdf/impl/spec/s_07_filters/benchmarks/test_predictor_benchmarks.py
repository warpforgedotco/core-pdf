# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import numpy
import pytest

from core_pdf.impl.spec.s_07_filters.decode_spec import FilterParams
from core_pdf.impl.spec.s_07_filters.predictors import (
    apply_png_predictor,
    apply_tiff_predictor,
)

COLUMNS = 512
COLORS = 3
ROWS = 256
ROW_LENGTH = COLUMNS * COLORS


def build_png_stream(filter_type: int) -> bytes:
    rng = numpy.random.default_rng(filter_type)
    rows = rng.integers(0, 256, size=(ROWS, ROW_LENGTH), dtype=numpy.uint8)
    out = bytearray()
    for row in rows:
        out.append(filter_type)
        out.extend(row.tobytes())
    return bytes(out)


def build_tiff_stream(rows: int, row_length: int, seed: int) -> bytes:
    rng = numpy.random.default_rng(seed)
    return rng.integers(0, 256, size=(rows, row_length), dtype=numpy.uint8).tobytes()


PNG_PAETH = build_png_stream(4)

TIFF_4BIT_PARAMS_LARGE = FilterParams(
    predictor=2, columns=COLUMNS, colors=COLORS, bits_per_component=4
)
TIFF_4BIT_DATA_LARGE = build_tiff_stream(
    rows=ROWS, row_length=(COLUMNS * COLORS * 4 + 7) // 8, seed=102
)

PNG_PARAMS = FilterParams(predictor=15, columns=COLUMNS, colors=COLORS, bits_per_component=8)


@pytest.mark.benchmark_high_impact
def test_png_predictor_paeth_benchmark(benchmark) -> None:
    result = benchmark(apply_png_predictor, PNG_PAETH, PNG_PARAMS)
    assert len(result) == ROWS * ROW_LENGTH


@pytest.mark.benchmark_high_impact
def test_tiff_predictor_4bit_numpy_benchmark(benchmark) -> None:
    result = benchmark(apply_tiff_predictor, TIFF_4BIT_DATA_LARGE, TIFF_4BIT_PARAMS_LARGE)
    assert result
