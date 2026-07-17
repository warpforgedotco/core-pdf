# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.engine.spec.s_07_syntax.xref import (
    XRefScanner,
    find_eof_marker,
    find_previous_object_marker,
    key_for,
)
from core_pdf.impl.primitives import PdfName


def build_xref_prev_chain(section_count: int) -> tuple[bytes, int]:
    data = bytearray(b"%PDF-1.4\n")
    prev: int | None = None
    for index in range(section_count):
        obj_num = index + 1
        offset = len(data)
        data.extend(f"xref\n{obj_num} 1\n0000000000 00000 n \ntrailer\n<< ".encode())
        data.extend(f"/Size {section_count + 1}".encode())
        if prev is not None:
            data.extend(f" /Prev {prev}".encode())
        data.extend(b" >>\n")
        prev = offset

    assert prev is not None
    return bytes(data), prev


def test_xref_prev_chain_loads_iteratively() -> None:
    data, start = build_xref_prev_chain(1500)

    entries, trailer = XRefScanner.load_section_chain(data, start, set())

    assert len(entries) == 1500
    assert key_for(1) in entries
    assert key_for(1500) in entries
    assert trailer[PdfName.of(b"Size")] == 1501


def test_xref_nearby_recovery_tries_candidates_until_one_parses() -> None:
    valid = b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 >>\n"
    false_stream = b"9 0 obj\n<< /Type /XRef /Size 1 >>\nnot-a-stream\nendobj\n"
    data = valid + b"padding\n" + false_stream
    damaged_offset = data.index(false_stream) + 12

    entries, trailer = XRefScanner.load_section_chain(data, damaged_offset, set())

    assert key_for(0, 65535) in entries
    assert trailer[PdfName.of(b"Size")] == 1


@pytest.mark.parametrize("marker", [b"%%XOF", b"%%EXF", b"%%EOX"])
def test_find_eof_marker_recovers_one_substitution(marker: bytes) -> None:
    data = b"%PDF-1.7\n% comment\n" + marker + b"\n% trailing comment"

    assert find_eof_marker(data) == data.index(marker)


def test_find_eof_marker_prefers_last_recoverable_marker() -> None:
    data = b"%PDF-1.7\n%%XOF\n% comment\n%%EOX\n"

    assert find_eof_marker(data) == data.index(b"%%EOX")


def test_find_eof_marker_rejects_unrelated_percent_tokens() -> None:
    assert find_eof_marker(b"%PDF-1.7\n% comment\n%%XYZ\n") == -1


def test_find_eof_marker_skips_embedded_exact_marker() -> None:
    data = b"%PDF-1.7\n%%EOF\nstream payload %%EOF-not-a-marker"

    assert find_eof_marker(data) == data.index(b"%%EOF")


def test_find_eof_marker_prefers_delimited_recovery_over_embedded_exact_marker() -> None:
    data = b"%PDF-1.7\n%%EOX\nstream payload %%EOF-not-a-marker"

    assert find_eof_marker(data) == data.index(b"%%EOX")


def test_find_eof_marker_preserves_compatibility_for_undelimited_marker() -> None:
    data = b"%PDF-1.7\ntrailing-%%EOF-junk"

    assert find_eof_marker(data) == data.index(b"%%EOF")


@pytest.mark.parametrize("marker", [b"%%XOF", b"%%EXF", b"%%EOX"])
def test_find_startxref_accepts_recovered_eof_marker(marker: bytes) -> None:
    data = b"%PDF-1.7\nstartxref\n123\n" + marker + b"\n"

    assert XRefScanner.find_startxref(data) == 123


def test_find_startxref_fallback_skips_trailing_false_section() -> None:
    valid = b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 >>\n"
    false_section = b"xref\nnot-an-entry\ntrailer\n<< /Type /XRef >>\n"
    data = b"%PDF-1.7\n" + valid + false_section

    assert XRefScanner.find_startxref(data) == data.index(valid)


def test_find_previous_object_marker_returns_last_valid_object() -> None:
    data = b"1 0 obj\nendobj\nnot-an-object\n27 3 obj\nendobj"

    assert find_previous_object_marker(data, len(data)) == data.index(b"27 3 obj")


def test_find_previous_object_marker_respects_upper_bound() -> None:
    data = b"1 0 obj\nendobj\n2 0 obj\nendobj"
    second_object = data.index(b"2 0 obj")

    assert find_previous_object_marker(data, second_object) == 0


def test_find_previous_object_marker_skips_invalid_candidates() -> None:
    data = b"3 0 obj\nendobj\nobject obj subjective"

    assert find_previous_object_marker(data, len(data)) == 0


def test_xref_stream_salvage_preserves_declared_binary_data() -> None:
    raw_data = b"\nentry\nendstream\ninside\nendobj\ninside\r"
    data = (
        b"1 0 obj\n<< /Type /XRef /Length "
        + str(len(raw_data)).encode()
        + b" >>\nstream\n"
        + raw_data
        + b"\nendstream\nendobj"
    )

    stream = XRefScanner.parse_xref_stream_salvage(data, 0)

    assert stream is not None
    assert stream.raw_data == raw_data


def test_xref_stream_salvage_uses_delimited_endstream_without_length() -> None:
    raw_data = b"xref bytes"
    data = b"1 0 obj\n<< /Type /XRef >>\nstream\r\n" + raw_data + b"\r\nendstream\nendobj"

    stream = XRefScanner.parse_xref_stream_salvage(data, 0)

    assert stream is not None
    assert stream.raw_data == raw_data + b"\r\n"
