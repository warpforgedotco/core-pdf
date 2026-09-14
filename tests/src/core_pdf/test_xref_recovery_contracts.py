"""Reader recovery preserves entry meaning across encoding and damage boundaries."""

import zlib

import pytest

from core_pdf.impl._impl.document.recovery import xref
from core_pdf.impl.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for


@pytest.mark.parametrize("ending", [b"", b"\n", b"\r", b"\r\n", b"\n\r", b" \n", b"\t\r\n"])
@pytest.mark.parametrize("separator", [b" ", b"\t"])
@pytest.mark.parametrize("marker", [b"f", b"n"])
def test_fixed_and_loose_xref_entries_agree_and_consume_line(ending, separator, marker):
    fixed = separator.join([b"0000000123", b"00007", marker]) + ending
    loose = separator.join([b"123", b"7", marker]) + ending
    expected = (123, 7, marker == b"n")
    for line in (fixed, loose):
        assert xref.parse_xref_entry_line(line) == expected
        assert xref.parse_xref_entry_at(b"prefix" + line, 6) == (*expected, 6 + len(line))


@pytest.mark.parametrize(
    "fields", [("-1", "0"), ("1", "-1"), ("1", "65536"), ("bad", "0"), ("0", "bad")]
)
def test_fixed_and_loose_xref_entries_reject_same_invalid_numbers(fields):
    offset, generation = fields
    for line in (f"{offset} {generation} n\n", f"{offset:0>10} {generation:0>5} n\n"):
        with pytest.raises(PdfParseError):
            xref.parse_xref_entry_at(line.encode(), 0)
    # Sign-first padding is also accepted by int() and must not bypass validation.
    if offset == "-1":
        with pytest.raises(PdfParseError):
            xref.parse_xref_entry_at(b"-000000001 00000 n\n", 0)


@pytest.mark.parametrize(
    "line", [b"", b"1", b"1 0 n extra", b"1 0 x", b"1\x0b0 n", b"1.0 0 n", b"0000000001 00000 nX"]
)
def test_xref_entries_reject_bad_fields_and_delimiters(line):
    with pytest.raises(PdfParseError):
        xref.parse_xref_entry_at(line, 0)


@pytest.mark.parametrize("offset", [0, 12])
def test_missing_entry_marker_is_inferred_from_offset(offset):
    assert xref.parse_xref_entry_line(f"{offset} 0".encode()) == (offset, 0, offset != 0)


def entry_values(entries):
    return {
        key: (
            entry.offset,
            entry.generation,
            entry.in_use,
            entry.object_stream,
            entry.index_in_stream,
        )
        for key, entry in entries.items()
    }


@pytest.mark.parametrize("widths", [[1, 2, 1], [0, 2, 1], [1, 0, 1]])
@pytest.mark.parametrize("truncated", [False, True])
def test_xref_stream_recovers_only_complete_rows_and_missing_fields(widths, truncated):
    rows = [(1, 25, 0), (1, 51, 3)]
    data = b"".join(
        value.to_bytes(width, "big")
        for row in rows
        for value, width in zip(row, widths, strict=True)
        if width
    )
    if truncated:
        data = data[:-1]
    dictionary = {"Size": 2, "W": widths}
    entries, trailer = xref.XRefScanner.parse_stream(PdfStream(dictionary, data))
    expected = {
        key_for(index, row[2]): (row[1] if widths[1] else 0, row[2], True, None, None)
        for index, row in enumerate(rows[:1] if truncated else rows)
    }
    assert entry_values(entries) == expected
    assert trailer is dictionary


@pytest.mark.parametrize("index", [[1, 2], [1, 2, 99], (1, 2)])
def test_xref_stream_repairs_size_and_ignores_incomplete_index_pair(index):
    entries, _ = xref.XRefScanner.parse_stream(
        PdfStream({"Size": 2, "W": [1, 1, 1], "Index": index}, b"\1\x10\0\1\x20\0")
    )
    assert entry_values(entries) == {
        key_for(1): (16, 0, True, None, None),
        key_for(2): (32, 0, True, None, None),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Size", 0),
        ("Size", True),
        ("W", [1, 2]),
        ("W", [1, -1, 1]),
        ("W", [1, True, 1]),
        ("W", [0, 0, 0]),
        ("Index", "bad"),
        ("Index", [0, True]),
        ("Index", [-1, 1]),
        ("Index", [0, -1]),
    ],
)
def test_xref_stream_rejects_invalid_shape_before_decoding(field, value):
    dictionary = {"Size": 1, "W": [1, 1, 1], field: value}
    with pytest.raises(PdfParseError):
        xref.XRefScanner.parse_stream(PdfStream(dictionary, b"\1\0\0"))


@pytest.mark.parametrize("size", [None, 0, -1, True, 1, 4])
def test_trailer_size_repair_preserves_input_and_valid_identity(size):
    trailer: PdfDict = {"Size": size, "Root": "preserved"}
    result = xref.XRefScanner.validate_trailer_size(trailer, 2)
    assert trailer["Size"] is size
    assert result == {"Size": 4 if size == 4 else 3, "Root": "preserved"}
    assert (result is trailer) == (size == 4)


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("trailer_keyword", [b"trailer\n", b""])
def test_table_recovery_reconciles_subsection_counts_and_missing_trailer_keyword(
    count, trailer_keyword
):
    data = (
        b"xref\n2 "
        + str(count).encode()
        + b"\n12 0 n\n24 1 n\n"
        + trailer_keyword
        + b"<< /Size 1 >>"
    )
    entries, trailer = xref.XRefScanner.parse_table_section(data, 0)
    assert entry_values(entries) == {
        key_for(2): (12, 0, True, None, None),
        key_for(3, 1): (24, 1, True, None, None),
    }
    assert trailer["Size"] == max(4, count + 2)


@pytest.mark.parametrize("subsection", [b"-1 2", b"0 -1", b"x 1", b"1", b"1 2 3", b"0\x0b1"])
def test_table_recovery_rejects_invalid_subsection_headers(subsection):
    with pytest.raises(PdfParseError):
        xref.XRefScanner.parse_table_section(
            b"xref\n" + subsection + b"\ntrailer\n<< /Size 1 >>", 0
        )


@pytest.mark.parametrize("marker", [b"%%EOF", b"%%EOX", b"%%XOF", b"%%EXF"])
def test_eof_discovery_accepts_one_substitution_and_prefers_delimited_marker(marker):
    data = b"header\n" + marker + b"\ntrailing garbage"
    assert xref.find_eof_marker(data) == 7


@pytest.mark.parametrize(
    "suffix", [b"\n%%EOF", b"\r\n%%EOF", b" % comment\n%%EOF", b"%comment\n%%EOF", b""]
)
def test_startxref_allows_comments_and_missing_eof(suffix):
    assert xref.XRefScanner.find_startxref(b"startxref\n123" + suffix) == 123


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"nothing",
        b"xstartxref\n123\n%%EOF",
        b"startxrefx\n123\n%%EOF",
        b"startxref\nBAD\n%%EOF",
        b"startxref\n123 trailing\n%%EOF",
        b"startxref\n12\x0b3\n%%EOF",
    ],
)
def test_startxref_ignores_invalid_markers_and_numbers(data):
    assert xref.XRefScanner.find_startxref(data) is None


@pytest.mark.parametrize("start", [0, 3, 10])
def test_nearby_section_recovery_returns_actual_table_offset(start):
    data = b"junk\nxref\n0 1\n0 65535 f\ntrailer\n<< /Size 1 >>"
    result = xref.XRefScanner.recover_section_at(data, start)
    assert result.offset == 5
    assert result.kind == "table"
    assert entry_values(result.entries) == {key_for(0, 65535): (0, 65535, False, None, None)}
    with pytest.raises(PdfParseError):
        xref.XRefScanner.recover_section_at(data, start, stream_only=True)


@pytest.mark.parametrize("separator", [b"\n", b"\r\n", b" \t\n"])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("length", [True, False])
def test_stream_salvage_preserves_rows_across_length_and_compression_forms(
    separator, compressed, length
):
    rows = b"\1\x10\0\1\x20\0"
    payload = zlib.compress(rows) if compressed else rows
    dictionary = b"<< /Type /XRef /Size 2 /W [1 1 1] /Index [0 2]"
    if compressed:
        dictionary += b" /Filter /FlateDecode"
    if length:
        dictionary += b" /Length " + str(len(payload)).encode()
    data = b"1 0 obj " + dictionary + b" >> stream" + separator + payload + b"\nendstream\nendobj"
    stream = xref.XRefScanner.parse_xref_stream_salvage(data, 0)
    assert stream is not None
    entries, _ = xref.XRefScanner.parse_stream(stream)
    assert entry_values(entries) == {
        key_for(0): (16, 0, True, None, None),
        key_for(1): (32, 0, True, None, None),
    }


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"x 0 obj",
        b" 1 0 obj",
        b"1 0 obj 42",
        b"1 0 obj <<",
        b"1 0 obj << /Type /Other >>",
        b"1 0 obj << /Type /XRef >> notstream",
        b"1 0 obj << /Type /XRef >> stream",
        b"1 0 obj << /Type /XRef >> streamX",
        b"1 0 obj << /Type /XRef /Length 0 >> stream\n",
        b"1 0 obj << /Type /XRef >> stream\nno terminator",
    ],
)
def test_stream_salvage_declines_unrecognizable_or_unterminated_objects(data):
    assert xref.XRefScanner.parse_xref_stream_salvage(data, 0) is None


@pytest.mark.parametrize("generation", [0, 65535])
def test_object_scan_ignores_fake_headers_inside_valid_stream_payload(generation):
    payload = b"9 0 obj 999 endobj"
    data = (
        b"1 0 obj << /Length "
        + str(len(payload)).encode()
        + b" >> stream\n"
        + payload
        + b"\nendstream\nendobj\n"
    )
    offset = len(data)
    data += f"2 {generation} obj 42 endobj".encode()
    entries = xref.XRefScanner.brute_force_scan(data)
    assert entry_values(entries) == {
        key_for(1): (0, 0, True, None, None),
        key_for(2, generation): (offset, generation, True, None, None),
    }
    assert list(xref.XRefScanner.brute_force_scan(data, max_entries=1)) == [key_for(1)]
