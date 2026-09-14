"""PDFMiner's recovery policy remains owned by its compatibility facade."""

import re
from io import BytesIO
from typing import Any

import pytest

from core_pdf.api.compat.pdfminer import extract_pages

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
        # The fourth entry names page object 3; redirect it to catalog object 1.
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
