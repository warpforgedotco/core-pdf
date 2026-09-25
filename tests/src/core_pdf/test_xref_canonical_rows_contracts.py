"""Canonical xref rows read at once read as the per-row parse reads them."""

import pytest

from core_pdf.impl.document.recovery import xref
from core_pdf.impl.exceptions import PdfParseError


def row(offset: int, generation: int, marker: bytes, eol: bytes = b"\r\n") -> bytes:
    return b"%010d %05d " % (offset, generation) + marker + eol


TABLES = [
    pytest.param(row(0, 65535, b"f") + row(17, 0, b"n") + row(81, 0, b"n"), 3, id="crlf"),
    pytest.param(row(0, 65535, b"f", b" \n") + row(17, 0, b"n", b" \r"), 2, id="space-eol"),
    pytest.param(row(0, 0, b"f", b" \n") + b"\r" + row(9, 0, b"n"), 2, id="eol-runs-on"),
    pytest.param(row(0, 0, b"n") + row(9, 70000, b"n") + row(10, 0, b"n"), 3, id="big-generation"),
    pytest.param(
        row(0, 0, b"n") + b"0000000009 00000 x\r\n" + row(10, 0, b"n"), 3, id="bad-marker"
    ),
    pytest.param(row(0, 0, b"n") + row(9, 0, b"n"), 5, id="short"),
    pytest.param(row(5, 0, b"n") + row(9, 0, b"n", b" \n"), 2, id="last-space-lf"),
    pytest.param(row(5, 0, b"n") + b"trailer", 3, id="early-trailer"),
    pytest.param(row(5, 0, b"n") * 3 + row(7, 1, b"n"), 3, id="extra-row"),
    pytest.param(b"%010d %05d n\r\n" % (12, 0), 1, id="single"),
]


def read(data: bytes, count: int) -> object:
    try:
        entries, pos, maximum = xref.XRefScanner.read_subsection(
            data + b"\ntrailer\n<<>>", 0, 4, count
        )
    except PdfParseError as error:
        return f"raised {error}"
    rows = [(key, e.offset, e.generation, e.in_use) for key, e in entries.items()]
    return rows, pos, maximum


@pytest.mark.parametrize(("data", "count"), TABLES)
def test_canonical_rows_read_as_the_per_row_parse(
    data: bytes, count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = read(data, count)
    monkeypatch.setattr(xref, "canonical_xref_rows", lambda *_: None)
    assert fast == read(data, count)


def test_a_canonical_table_takes_the_row_block() -> None:
    data = row(0, 65535, b"f") + row(17, 0, b"n") + row(81, 0, b"n") + b"trailer"
    assert xref.canonical_xref_rows(data, 0, 2) is not None
    assert xref.canonical_xref_rows(data, 0, 3) is None
