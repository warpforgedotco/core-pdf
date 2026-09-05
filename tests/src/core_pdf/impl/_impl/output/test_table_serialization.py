"""Structured table HTML honors optional associations and explicit row semantics."""

import pytest

from core_pdf.impl._impl.output.model import (
    Table,
    TableAssociatedText,
    TableCell,
    TableRowBand,
)
from core_pdf.impl._impl.output.serialize import table_to_html, table_to_markdown


@pytest.mark.parametrize("association", ["title", "caption"])
def test_table_associated_text_does_not_require_explicit_row_bands(association: str) -> None:
    text = TableAssociatedText("Report context", kind=association)
    table = Table(
        order=0,
        rows=((TableCell(0, 0, "Name"), TableCell(0, 1, "Value")),),
        title=text if association == "title" else None,
        caption=text if association == "caption" else None,
    )

    html = table_to_html(table)

    assert f'<div data-table-associated="{association}">Report context</div>' in html
    assert "<th>Name</th><th>Value</th>" in html
    assert table_to_markdown(table) == html


def test_explicit_body_row_is_not_promoted_to_a_header() -> None:
    table = Table(
        order=0,
        rows=((TableCell(0, 0, "Body content"),),),
        row_bands=(TableRowBand(0, kind="body"),),
    )

    html = table_to_html(table)

    assert "<td>Body content</td>" in html
    assert "<th>" not in html
    assert "<thead>" not in html


def test_multiple_explicit_header_rows_remain_headers() -> None:
    table = Table(
        order=0,
        rows=((TableCell(0, 0, "Heading"),), (TableCell(1, 0, "Subheading"),)),
        row_bands=(TableRowBand(0, kind="header"), TableRowBand(1, kind="header")),
    )

    html = table_to_html(table)

    assert "<th>Heading</th>" in html
    assert "<th>Subheading</th>" in html
    assert "<td>" not in html
