# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import numpy

from core_pdf.impl.engine.spec.s_07_content.operations import (
    content_stream_may_show_text,
    count_content_stream_operators,
)

TEXT_BLOCK = b"""\
q
1 0 0 1 100 100 cm
0.2 0.4 0.6 rg
BT
/F1 12 Tf
14 TL
(Hello World) Tj
T*
[(Hello) -250 (World) -300 (Benchmark) 120 (Text)] TJ
ET
100 100 200 200 re
f
Q
"""
TEXT_CONTENT_STREAM = TEXT_BLOCK * 200


def build_vector_only_stream(operations: int) -> bytes:
    rng = numpy.random.default_rng(3)
    lines = []
    for _ in range(operations):
        x, y = rng.uniform(0, 612, size=2)
        lines.append(f"{x:.2f} {y:.2f} l")
    lines.append("S")
    return "\n".join(lines).encode()


VECTOR_ONLY_STREAM = build_vector_only_stream(4_000)


def test_content_stream_may_show_text_positive_benchmark(benchmark) -> None:
    result = benchmark(content_stream_may_show_text, TEXT_CONTENT_STREAM)
    assert result is True


def test_content_stream_may_show_text_negative_benchmark(benchmark) -> None:
    result = benchmark(content_stream_may_show_text, VECTOR_ONLY_STREAM)
    assert result is False


def test_count_content_stream_operators_benchmark(benchmark) -> None:
    result = benchmark(count_content_stream_operators, TEXT_CONTENT_STREAM)
    assert result.text > 0
    assert result.malformed == 0
