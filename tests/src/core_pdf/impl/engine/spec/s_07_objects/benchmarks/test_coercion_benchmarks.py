# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.engine.spec.s_07_objects.coercion import coerce_value, parse_float, parse_int
from core_pdf.impl.primitives import PdfString

NESTED_VALUE = {
    "Type": "Page",
    "MediaBox": [0, 0, 612, 792],
    "Annots": [
        {"Subtype": "Link", "Rect": [10, 10, 100, 20], "Contents": PdfString(b"note-1")},
        {"Subtype": "Link", "Rect": [10, 30, 100, 40], "Contents": PdfString(b"note-2")},
        {"Subtype": "Text", "Rect": [10, 50, 100, 60], "Contents": PdfString(b"note-3")},
    ],
    "Resources": {"Font": {"F1": "5 0 R"}, "XObject": {"Im0": "12 0 R"}},
}


def decode_pdf_string(data: bytes) -> str:
    return data.decode("latin-1")


@pytest.mark.benchmark_high_impact
def test_parse_int_fast_path_benchmark(benchmark) -> None:
    result = benchmark(parse_int, 1234, None)
    assert result == 1234


def test_parse_int_from_bytes_benchmark(benchmark) -> None:
    result = benchmark(parse_int, b"1234", None)
    assert result == 1234


def test_parse_float_from_bytes_benchmark(benchmark) -> None:
    result = benchmark(parse_float, b"1234.5", 0.0)
    assert result == 1234.5


@pytest.mark.benchmark_high_impact
def test_coerce_value_nested_benchmark(benchmark) -> None:
    result = benchmark(coerce_value, NESTED_VALUE, decode_pdf_string)
    assert result["Annots"][0]["Contents"] == "note-1"
