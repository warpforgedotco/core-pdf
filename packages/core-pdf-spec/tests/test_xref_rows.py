"""Xref row and stream APIs enforce the same field layout and row semantics."""

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.xref import (
    PdfXRefEntry,
    XRefScanner,
    decode_xref_row,
    decode_xref_rows,
    key_for,
)


def internal_entries(table: dict[int, PdfXRefEntry]) -> dict[int, tuple[object, ...]]:
    return {
        key: (
            entry.offset,
            entry.generation,
            entry.in_use,
            entry.object_stream,
            entry.index_in_stream,
        )
        for key, entry in table.items()
    }


@pytest.mark.parametrize(
    ("widths", "data", "index", "size", "expected"),
    [
        (
            [1, 1, 1],
            bytes([0, 0, 255, 1, 20, 2, 2, 7, 3, 7, 0, 0]),
            [1, 2, 5, 2],
            7,
            {
                key_for(1, 255): (0, 255, False, None, None),
                key_for(2, 2): (20, 2, True, None, None),
                key_for(5): (0, 0, True, 7, 3),
                key_for(6): (0, 0, False, None, None),
            },
        ),
        (
            [0, 1, 0],
            b"\x10\x20",
            [2, 2],
            4,
            {key_for(2): (16, 0, True, None, None), key_for(3): (32, 0, True, None, None)},
        ),
        (
            [1, 0, 0],
            b"\x00\x01\x07",
            [1, 3],
            4,
            {
                key_for(1): (0, 0, False, None, None),
                key_for(2): (0, 0, True, None, None),
                key_for(3): (0, 0, False, None, None),
            },
        ),
    ],
)
def test_row_bulk_and_stream_decoding_share_defaults_and_entry_types(
    widths: list[int],
    data: bytes,
    index: list[int],
    size: int,
    expected: dict[int, tuple[object, ...]],
) -> None:
    # ISO 32000-2:2020, 7.5.8.3 supplies zero-width defaults and null future entry types.
    rows = {}
    position = 0
    for pair in range(0, len(index), 2):
        for object_number in range(index[pair], index[pair] + index[pair + 1]):
            key, entry, position = decode_xref_row(data, position, widths, object_number)
            rows[key] = entry
    assert position == len(data)
    assert internal_entries(rows) == expected
    assert internal_entries(decode_xref_rows(data, widths, index, size)) == expected
    dictionary = {"Type": "XRef", "Size": size, "W": widths, "Index": index}
    stream_entries, trailer = XRefScanner.parse_stream(PdfStream(dictionary, data))
    assert internal_entries(stream_entries) == expected
    assert trailer is dictionary


@pytest.mark.parametrize("widths", [[], [1, 1], [-1, 1, 1], [True, 1, 1], [0, 0, 0]])
def test_all_xref_entrypoints_reject_invalid_widths(widths: list[int]) -> None:
    with pytest.raises(PdfParseError, match="invalid xref stream W"):
        decode_xref_row(b"\x00" * 3, 0, widths, 0)
    with pytest.raises(PdfParseError, match="invalid xref stream W"):
        decode_xref_rows(b"", widths, [0, 0], 1)
    with pytest.raises(PdfParseError, match="invalid xref stream W"):
        XRefScanner.parse_stream(
            PdfStream({"Type": "XRef", "Size": 1, "W": widths, "Index": [0, 0]}, b"")
        )


@pytest.mark.parametrize("index", [[0], [0, -1], [True, 1], [1, 2], [0, 1.0]])
def test_bulk_and_stream_reject_invalid_index(index: list[int]) -> None:
    with pytest.raises(PdfParseError, match="invalid xref stream Index"):
        decode_xref_rows(b"", [1, 1, 1], index, 2)
    with pytest.raises(PdfParseError, match="invalid xref stream Index"):
        XRefScanner.parse_stream(
            PdfStream({"Type": "XRef", "Size": 2, "W": [1, 1, 1], "Index": index}, b"")
        )


@pytest.mark.parametrize(
    ("data", "widths", "message"),
    [(b"\x01\x00", [1, 1, 1], "length"), (b"\x01\x00\x01\x00\x00", [1, 1, 3], "generation")],
)
def test_all_xref_entrypoints_reject_invalid_rows(
    data: bytes, widths: list[int], message: str
) -> None:
    with pytest.raises(PdfParseError, match=message):
        decode_xref_row(data, 0, widths, 1)
    with pytest.raises(PdfParseError, match=message):
        decode_xref_rows(data, widths, [1, 1], 2)
    with pytest.raises(PdfParseError, match=message):
        XRefScanner.parse_stream(
            PdfStream({"Type": "XRef", "Size": 2, "W": widths, "Index": [1, 1]}, data)
        )


def test_single_xref_row_retains_position_validation() -> None:
    for position in (-1, 1):
        with pytest.raises(PdfParseError, match="length"):
            decode_xref_row(b"\x00\x00\x00", position, [1, 1, 1], 1)
    with pytest.raises(PdfParseError, match="Index"):
        decode_xref_row(b"\x00\x00\x00", 0, [1, 1, 1], -1)


def test_xref_stream_default_index_covers_size() -> None:
    entries, _ = XRefScanner.parse_stream(
        PdfStream({"Type": "XRef", "Size": 1, "W": [1, 1, 2]}, b"\x00\x00\xff\xff")
    )
    assert internal_entries(entries) == {key_for(0, 65535): (0, 65535, False, None, None)}
