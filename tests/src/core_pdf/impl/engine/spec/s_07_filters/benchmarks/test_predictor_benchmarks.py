# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import numpy
import pytest

from core_pdf.impl.engine.spec.s_07_filters.decode_spec import FilterParams
from core_pdf.impl.engine.spec.s_07_filters.predictors import (
    apply_png_predictor,
    apply_tiff_predictor,
)

pytestmark = pytest.mark.benchmark_high_impact

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


PNG_NONE = build_png_stream(0)
PNG_SUB = build_png_stream(1)
PNG_UP = build_png_stream(2)
PNG_AVERAGE = build_png_stream(3)
PNG_PAETH = build_png_stream(4)

TIFF_8BIT_PARAMS = FilterParams(predictor=2, columns=COLUMNS, colors=COLORS, bits_per_component=8)
TIFF_8BIT_DATA = build_tiff_stream(ROWS, ROW_LENGTH, seed=100)

TIFF_4BIT_PARAMS_SMALL = FilterParams(predictor=2, columns=64, colors=1, bits_per_component=4)
TIFF_4BIT_DATA_SMALL = build_tiff_stream(rows=8, row_length=32, seed=101)

TIFF_4BIT_PARAMS_LARGE = FilterParams(
    predictor=2, columns=COLUMNS, colors=COLORS, bits_per_component=4
)
TIFF_4BIT_DATA_LARGE = build_tiff_stream(
    rows=ROWS, row_length=(COLUMNS * COLORS * 4 + 7) // 8, seed=102
)

PNG_PARAMS = FilterParams(predictor=15, columns=COLUMNS, colors=COLORS, bits_per_component=8)


def test_png_predictor_none_benchmark(benchmark) -> None:
    result = benchmark(apply_png_predictor, PNG_NONE, PNG_PARAMS)
    assert len(result) == ROWS * ROW_LENGTH


def test_png_predictor_sub_benchmark(benchmark) -> None:
    result = benchmark(apply_png_predictor, PNG_SUB, PNG_PARAMS)
    assert len(result) == ROWS * ROW_LENGTH


def test_png_predictor_up_benchmark(benchmark) -> None:
    result = benchmark(apply_png_predictor, PNG_UP, PNG_PARAMS)
    assert len(result) == ROWS * ROW_LENGTH


def test_png_predictor_average_benchmark(benchmark) -> None:
    result = benchmark(apply_png_predictor, PNG_AVERAGE, PNG_PARAMS)
    assert len(result) == ROWS * ROW_LENGTH


def test_png_predictor_paeth_benchmark(benchmark) -> None:
    result = benchmark(apply_png_predictor, PNG_PAETH, PNG_PARAMS)
    assert len(result) == ROWS * ROW_LENGTH


def test_tiff_predictor_8bit_benchmark(benchmark) -> None:
    result = benchmark(apply_tiff_predictor, TIFF_8BIT_DATA, TIFF_8BIT_PARAMS)
    assert len(result) == ROWS * ROW_LENGTH


def test_tiff_predictor_4bit_scalar_benchmark(benchmark) -> None:
    result = benchmark(apply_tiff_predictor, TIFF_4BIT_DATA_SMALL, TIFF_4BIT_PARAMS_SMALL)
    assert result


def test_tiff_predictor_4bit_numpy_benchmark(benchmark) -> None:
    result = benchmark(apply_tiff_predictor, TIFF_4BIT_DATA_LARGE, TIFF_4BIT_PARAMS_LARGE)
    assert result
