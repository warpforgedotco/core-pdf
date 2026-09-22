import re
from io import BytesIO
from typing import Any

import pytest

from core_pdf.impl.exceptions import PdfParseError
from core_pdf_compat.pdfminer import extract_pages

reference = pytest.importorskip("pdfminer.high_level")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("damage", ["none", "startxref", "xref-header", "stale-page"])
def test_recovered_page_selection_matches_pdfminer(text_pdf_bytes, damage):
    data = text_pdf_bytes
    if damage == "startxref":
        data = re.sub(rb"(startxref\s+)(\d+)", lambda match: match[1] + b"0" * len(match[2]), data)
    elif damage == "xref-header":
        data = data.replace(b"\nxref\n", b"\nxxxx\n", 1)
    elif damage == "stale-page":
        xref = data.index(b"\nxref\n") + 1
        lines = data[xref:].splitlines(keepends=True)
        lines[5] = lines[3]
        data = data[:xref] + b"".join(lines)

    def snapshot(read: Any):
        pages = list(read(BytesIO(data)))
        return [
            (
                tuple(page.bbox),
                "".join(item.get_text() for item in page if hasattr(item, "get_text")),
            )
            for page in pages
        ]

    snapshots = [snapshot(reference.extract_pages), snapshot(extract_pages)]
    assert snapshots[1] == snapshots[0]
    if damage == "stale-page":
        assert snapshots[0] == []
    else:
        assert len(snapshots[0]) == 1
        assert snapshots[0][0][1] == "Hello maintenance\n"


@pytest.mark.parametrize(
    "offset",
    ["exact", "inside-1", "inside-2", "inside-3", "zero", "overflow", "eof"],
)
def test_startxref_boundaries_preserve_page_content(text_pdf_bytes: bytes, offset: str) -> None:
    xref = text_pdf_bytes.index(b"\nxref\n") + 1
    offsets = {
        "exact": xref,
        "inside-1": xref + 1,
        "inside-2": xref + 2,
        "inside-3": xref + 3,
        "zero": 0,
        "overflow": 2**31,
        "eof": len(text_pdf_bytes),
    }
    data = re.sub(
        rb"(startxref\s+)\d+",
        lambda match: match[1] + str(offsets[offset]).encode("ascii"),
        text_pdf_bytes,
    )
    readers: tuple[Any, ...] = (reference.extract_pages, extract_pages)
    for read in readers:
        pages = list(read(BytesIO(data)))
        assert len(pages) == 1
        assert "".join(item.get_text() for item in pages[0] if hasattr(item, "get_text")) == (
            "Hello maintenance\n"
        )


def test_negative_startxref_records_native_rejection(text_pdf_bytes: bytes) -> None:
    data = re.sub(rb"(startxref\s+)\d+", rb"\g<1>-1", text_pdf_bytes)
    pages = list(reference.extract_pages(BytesIO(data)))
    assert len(pages) == 1
    assert "".join(item.get_text() for item in pages[0] if hasattr(item, "get_text")) == (
        "Hello maintenance\n"
    )
    with pytest.raises(PdfParseError, match="invalid xref section"):
        list(extract_pages(BytesIO(data)))
