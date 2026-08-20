# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import zlib

import numpy
import pytest

from core_pdf.impl.engine.spec.s_07_filters.flate import apply_flate, looks_like_pdf_content_stream

pytestmark = pytest.mark.benchmark_high_impact


def build_content_stream(operations: int) -> bytes:
    rng = numpy.random.default_rng(7)
    lines = []
    for _ in range(operations):
        x, y = rng.uniform(0, 612, size=2)
        lines.append(f"{x:.2f} {y:.2f} l")
    lines.append("S")
    return "\n".join(lines).encode()


CONTENT_STREAM = build_content_stream(4_000)
FLATE_COMPRESSED = zlib.compress(CONTENT_STREAM, level=6)
RAW_DEFLATE_COMPRESSED = zlib.compress(CONTENT_STREAM, level=6, wbits=-15)
BINARY_NOISE = numpy.random.default_rng(9).integers(0, 256, size=1024, dtype=numpy.uint8).tobytes()


def test_apply_flate_zlib_wrapped_benchmark(benchmark) -> None:
    result = benchmark(apply_flate, FLATE_COMPRESSED, None)
    assert result == CONTENT_STREAM


def test_apply_flate_raw_deflate_benchmark(benchmark) -> None:
    result = benchmark(apply_flate, RAW_DEFLATE_COMPRESSED, None)
    assert result == CONTENT_STREAM


def test_looks_like_pdf_content_stream_positive_benchmark(benchmark) -> None:
    result = benchmark(looks_like_pdf_content_stream, CONTENT_STREAM)
    assert result is True


def test_looks_like_pdf_content_stream_negative_benchmark(benchmark) -> None:
    result = benchmark(looks_like_pdf_content_stream, BINARY_NOISE)
    assert result is False
