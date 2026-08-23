# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.engine.spec.s_07_objects.coercion import (
    coerce_to_bytes,
    coerce_value,
    normalize_pdf_name,
    parse_float,
    parse_int,
)
from core_pdf.impl.engine.spec.s_09_fonts.encoding import decode_pdf_text_string
from core_pdf.impl.primitives import PdfName, PdfString


class ConversionHook:
    def __int__(self) -> int:
        return 7

    def __float__(self) -> float:
        return 7.0


class DataWrapper:
    data = b"unexpected"


def test_scalar_parsers_accept_explicit_pdf_scalar_forms() -> None:
    assert parse_int(7) == 7
    assert parse_int(b"-12") == -12
    assert parse_int(bytearray(b"13")) == 13
    assert parse_int(memoryview(b"14")) == 14
    assert parse_float(2) == 2.0
    assert parse_float(b"1.25") == 1.25


def test_scalar_parsers_do_not_call_arbitrary_conversion_hooks() -> None:
    value = ConversionHook()

    assert parse_int(value, default=None) is None
    assert parse_float(value, default=None) is None


def test_pdf_name_and_string_conversion_is_explicit() -> None:
    assert normalize_pdf_name(PdfName.of("/Type")) == "Type"
    assert normalize_pdf_name(b"/Subtype") == "Subtype"
    assert normalize_pdf_name(DataWrapper(), default="fallback") == "fallback"
    assert coerce_to_bytes(PdfString(b"payload")) == b"payload"


def test_byte_conversion_does_not_probe_arbitrary_data_attributes() -> None:
    with pytest.raises(TypeError, match="DataWrapper"):
        coerce_to_bytes(DataWrapper())


def test_coerce_value_preserves_existing_containers_when_no_transformation_is_needed() -> None:
    value = {"nested": [1, {"ok": True}]}

    assert coerce_value(value) is value
    assert coerce_value(value, decode_pdf_text_string) is value
    assert coerce_value(value, None) is value
