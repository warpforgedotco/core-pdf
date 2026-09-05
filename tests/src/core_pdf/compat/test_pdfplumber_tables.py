# SPDX-License-Identifier: AGPL-3.0-only
"""Native structured tables may retain text without cell or table geometry."""

from io import BytesIO

import pytest

from core_pdf.api.compat import pdfplumber
from core_pdf.impl._impl.output.model import Document, Page, Table, TableCell
from tests.helpers.pdf_bytes import one_page_pdf


def test_table_merge_preserves_rows_without_cell_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    tables = (
        Table(
            order=0,
            bbox=(0, 50, 100, 100),
            rows=((TableCell(0, 0, "Upper"),), ()),
        ),
        Table(order=1, bbox=(0, 0, 100, 50), rows=((TableCell(0, 0, "Lower"),),)),
    )
    result = Document(pages=(Page(page_number=1, tables=tables),))
    with pdfplumber.open(BytesIO(one_page_pdf(b""))) as pdf:
        monkeypatch.setattr(pdf._document, "extract", lambda **kwargs: result)

        merged = pdf.pages[0].find_tables()

        assert len(merged) == 1
        assert merged[0].bbox == (0, 0, 100, 100)
        assert merged[0].extract() == [["Upper"], [""], ["Lower"]]
        assert merged[0].extract(text_layout=True) == [["Upper"], [""], ["Lower"]]


def test_table_with_missing_bounds_remains_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    tables = (
        Table(order=0, rows=((TableCell(0, 0, "Unpositioned"),),)),
        Table(order=1, bbox=(0, 0, 100, 50), rows=((TableCell(0, 0, "Positioned"),),)),
    )
    result = Document(pages=(Page(page_number=1, tables=tables),))
    with pdfplumber.open(BytesIO(one_page_pdf(b""))) as pdf:
        monkeypatch.setattr(pdf._document, "extract", lambda **kwargs: result)

        found = pdf.pages[0].find_tables()

        assert len(found) == 2
        assert [table.extract() for table in found] == [[["Unpositioned"]], [["Positioned"]]]
