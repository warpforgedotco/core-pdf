import zlib

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.document.document import object_headers_present
from core_pdf.impl.document.recovery import xref
from core_pdf.impl.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.types import PdfName, PdfReference


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
    trailer: PdfDict = {"Size": size, "Root": PdfReference(1)}
    result = xref.XRefScanner.validate_trailer_size(trailer, 2)
    assert trailer["Size"] is size
    assert result == {"Size": 4 if size == 4 else 3, "Root": PdfReference(1)}
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


@pytest.mark.parametrize("generation", [0, 7])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("limit", [1, 2, 10])
def test_recovered_object_stream_entries_preserve_parser_indexes(generation, existing, limit):
    from core_pdf.impl.document.recovery.objects import PdfObjectStream
    from core_pdf_spec.s_07_syntax.xref import PdfXRefEntry

    header = b"20 0 21 3 22 6 "
    stream = PdfStream({"N": 3, "First": len(header)}, header + b"10 20 30")
    container_key = key_for(5, generation)
    entries = {container_key: PdfXRefEntry(100, generation, True)}
    preserved = PdfXRefEntry(500, 0, True)
    if existing:
        entries[key_for(20)] = preserved
    initial_count = len(entries)
    xref.XRefScanner.recover_object_stream_entries(entries, {container_key: (100, stream)}, limit)
    assert len(entries) == max(initial_count, min(limit, 4))
    parser = PdfObjectStream(stream)
    try:
        for index, number in enumerate((20, 21, 22)):
            entry = entries.get(key_for(number))
            if entry is None:
                continue
            if existing and number == 20:
                assert entry is preserved
                continue
            assert entry.object_stream == 5
            assert entry.index_in_stream == index
            assert (
                parser.get_at_index(entry.index_in_stream, expected_reference=PdfReference(number))
                == (index + 1) * 10
            )
    finally:
        parser.close()


@pytest.mark.parametrize(
    "kind", ["free", "compressed", "negative", "absent", "stale", "wrong-type", "bad-header"]
)
def test_object_stream_recovery_skips_unusable_containers(kind):
    from core_pdf_spec.s_07_syntax.xref import PdfXRefEntry

    stream = PdfStream({"N": 1, "First": 4}, b"2 0 42")
    entry = PdfXRefEntry(100, 0, True)
    parsed = {key_for(1): (100, stream)}
    if kind == "free":
        entry = entry._replace(in_use=False)
    elif kind == "compressed":
        entry = entry._replace(object_stream=9)
    elif kind == "negative":
        entry = entry._replace(offset=-1)
    elif kind == "absent":
        parsed.clear()
    elif kind == "stale":
        parsed[key_for(1)] = (101, stream)
    elif kind == "wrong-type":
        # A PDF name, not a bare str: PdfObject has no str member, and the
        # reader reaches this through recover_pdf_name, which takes either.
        stream.dictionary = {"Type": PdfName.of(b"Other")}
    else:
        stream.dictionary = {"N": -1, "First": 0}
    entries = {key_for(1): entry}
    xref.XRefScanner.recover_object_stream_entries(entries, parsed)
    assert entries == {key_for(1): entry}


@pytest.mark.parametrize("representation", ["bytes", "view", "slice", "strided"])
@pytest.mark.parametrize("allow_prefix", [False, True])
def test_bounded_header_discovery_agrees_for_buffer_representations(representation, allow_prefix):
    data = b"1 0 obj 42 endobj\n2 0 obj true endobj\n"
    source = data
    if representation == "view":
        source = memoryview(data)
    elif representation == "slice":
        source = memoryview(b"prefix" + data + b"suffix")[6:-6]
    elif representation == "strided":
        source = memoryview(b"".join(bytes((byte, 0)) for byte in data))[::2]
    second = data.index(b"2 0 obj")
    expected = [(0, 1, 0), (second, 2, 0)] if allow_prefix else [(second, 2, 0)]
    assert (
        list(
            xref.iter_indirect_object_headers(
                source, 4, len(data), allow_prefix_before_start=allow_prefix
            )
        )
        == expected
    )
    assert list(xref.iter_indirect_object_headers(source, -5, second)) == [(0, 1, 0)]
    assert list(xref.iter_indirect_object_headers(source, len(data), len(data) + 20)) == []


@pytest.mark.parametrize(
    "data", [b"obj", b"1 obj", b"1 65536 obj", b"x1 0 obj", b"1 0 object", b"1 0 xyz"]
)
def test_header_discovery_rejects_malformed_prefixes_and_keyword_suffixes(data):
    marker = data.find(b"obj")
    assert xref.parse_object_marker_prefix(data, marker) is None
    assert xref.find_previous_object_marker(data, len(data)) is None


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"", -1),
        (b"%", -1),
        (b"garbage%%EOF", 7),
        (b"garbage%%EOX", 7),
        (b"%%BAD", -1),
        (b"\n%%EOFjunk\n%%EOF\n", 11),
    ],
)
def test_eof_recovery_keeps_raw_fallback_but_prefers_delimited_exact_marker(data, expected):
    assert xref.find_eof_marker(data) == expected


@pytest.mark.parametrize("delimiter", [b"<<", b"[", b"(", b"/", b"%", b"{"])
def test_an_object_header_may_end_in_a_delimiter(delimiter: bytes) -> None:
    assert object_headers_present(b"12 0 obj" + delimiter, [(key_for(12, 0), 0)]) == [True]


def test_an_object_header_keyword_is_not_a_prefix_match() -> None:
    assert object_headers_present(b"12 0 objx", [(key_for(12, 0), 0)]) == [False]


def pdf_with_headers_ending_in_dictionaries() -> bytes:
    content = b"BT /F1 12 Tf 20 100 Td (Hello maintenance) Tj ET"
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    )
    data = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, value in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj".encode() + value + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(data)


def test_objects_written_without_a_space_after_obj_need_no_offset_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every entry of such a file used to look stale, which scanned the whole
    # file for replacement offsets that the scan could not find either.
    scans: list[int] = []

    def brute_force_xref(self: PdfDocument) -> dict:
        scans.append(1)
        return {}

    monkeypatch.setattr(PdfDocument, "brute_force_xref", brute_force_xref)
    with PdfDocument(pdf_with_headers_ending_in_dictionaries()) as document:
        assert not document.xref_was_recovered
        assert "Hello maintenance" in document.pages[0].extract().text
    assert scans == []
