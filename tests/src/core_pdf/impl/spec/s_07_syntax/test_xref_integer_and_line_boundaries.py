# SPDX-License-Identifier: AGPL-3.0-only
"""Strict xref integers and line endings remain distinct from recovery."""

import pytest

from core_pdf.impl._impl.document.recovery.xref import XRefScanner as RecoveryScanner
from core_pdf.impl._impl.document.recovery.xref import (
    parse_xref_entry_at as recover_entry,
)
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.xref import XRefScanner, parse_xref_entry_at


@pytest.mark.parametrize("header", [b"1_0 1", b"1 0_1"])
def test_subsection_underscores_are_only_accepted_by_recovery(header: bytes) -> None:
    data = b"xref\n" + header + b"\n0000000017 00000 n \ntrailer\n<< /Size 11 >>"
    with pytest.raises(PdfParseError, match="subsection"):
        XRefScanner.parse_table_section(data, 0)
    entries, _, _, _ = RecoveryScanner.parse_table_section(data, 0)
    assert len(entries) == 1
    assert next(iter(entries.values())).offset == 17


def test_subsection_pdf_signed_integer_syntax_still_parses() -> None:
    data = b"xref\n+1 +1\n0000000017 00000 n \ntrailer\n<< /Size 2 >>"
    entries, _, _, _ = XRefScanner.parse_table_section(data, 0)
    assert len(entries) == 1


@pytest.mark.parametrize(
    ("source", "strict_end", "recovery_end"),
    [
        (b"line\n\rnext", 5, 6),
        (b"line\r\nnext", 6, 6),
        (b"line\r\n\rnext", 6, 6),
        (b"line\nnext", 5, 5),
        (b"line\rnext", 5, 5),
    ],
)
def test_line_reading_only_combines_crlf_in_spec(
    source: bytes, strict_end: int, recovery_end: int
) -> None:
    assert XRefScanner.read_line(source, 0) == (b"line", strict_end)
    assert RecoveryScanner.read_line(source, 0) == (b"line", recovery_end)


def test_lfcr_subsection_ending_is_only_collapsed_by_recovery() -> None:
    data = b"xref\n1 1\n\r0000000017 00000 n \ntrailer\n<< /Size 2 >>"
    with pytest.raises(PdfParseError, match="xref table entry"):
        XRefScanner.parse_table_section(data, 0)
    entries, _, _, _ = RecoveryScanner.parse_table_section(data, 0)
    assert next(iter(entries.values())).offset == 17


@pytest.mark.parametrize("marker", [b"f", b"n"])
@pytest.mark.parametrize("generation", [65536, 99999])
def test_fixed_width_generation_limit_is_strict_only(marker: bytes, generation: int) -> None:
    row = b"0000000017 " + f"{generation:05d}".encode() + b" " + marker + b" \n"
    with pytest.raises(PdfParseError, match="generation"):
        parse_xref_entry_at(row, 0)
    assert recover_entry(row, 0) == (17, generation, marker == b"n", 20)


@pytest.mark.parametrize("entry_type", [0, 1])
def test_xref_stream_generation_limit_is_strict_only(entry_type: int) -> None:
    row = bytes((entry_type, 17)) + (65536).to_bytes(3, "big")
    stream = PdfStream(
        {"Type": "XRef", "Size": 2, "Index": [1, 1], "W": [1, 1, 3]},
        decoded_data=row,
    )
    with pytest.raises(PdfParseError, match="generation"):
        XRefScanner.parse_stream(stream)
    entries, _ = RecoveryScanner.parse_stream(stream)
    assert next(iter(entries.values())).generation == 65536


@pytest.mark.parametrize("generation", [0, 65535])
def test_generation_range_boundaries_are_valid(generation: int) -> None:
    row = b"0000000017 " + f"{generation:05d}".encode() + b" n \n"
    assert parse_xref_entry_at(row, 0) == (17, generation, True, 20)
    binary_row = bytes((1, 17)) + generation.to_bytes(3, "big")
    stream = PdfStream(
        {"Type": "XRef", "Size": 2, "Index": [1, 1], "W": [1, 1, 3]},
        decoded_data=binary_row,
    )
    entries, _ = XRefScanner.parse_stream(stream)
    assert next(iter(entries.values())).generation == generation


def test_compressed_object_index_is_not_a_generation_number() -> None:
    stream = PdfStream(
        {"Type": "XRef", "Size": 2, "Index": [1, 1], "W": [1, 1, 3]},
        decoded_data=bytes((2, 17)) + (65536).to_bytes(3, "big"),
    )
    entries, _ = XRefScanner.parse_stream(stream)
    entry = next(iter(entries.values()))
    assert entry.generation == 0
    assert entry.index_in_stream == 65536
