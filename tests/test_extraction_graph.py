from __future__ import annotations

from typing import Any, cast

from core_pdf.impl.engine.structured import (
    Block,
    BlockKind,
    ContentNode,
    Diagnostic,
    Document,
    Page,
    Table,
    TableAssociatedText,
    TableCell,
    TableColumnBand,
    TableReference,
    TableRowBand,
    TextLine,
    TextLineReference,
)


def test_page_exposes_shared_nodes_and_independent_views() -> None:
    block = Block(order=0, kind=BlockKind.PARAGRAPH)
    table = Table(
        order=1,
        rows=((TableCell(row=0, column=0, text="value"),),),
    )
    page = Page(page_number=1, blocks=(block,), tables=(table,))

    assert tuple(node.payload for node in page.nodes) == (block, table)
    assert all(isinstance(node, ContentNode) for node in page.nodes)
    assert [node.page_number for node in page.nodes] == [1, 1]
    assert page.text_view.elements == page.elements
    assert page.table_view.tables == (table,)
    assert page.text == "value"


def test_table_exposes_geometry_bands_and_associated_text() -> None:
    table = Table(
        order=0,
        rows=(
            (
                TableCell(0, 0, "A", bbox=(0.0, 10.0, 10.0, 20.0)),
                TableCell(0, 1, "B", bbox=(10.0, 10.0, 20.0, 20.0)),
            ),
            (TableCell(1, 0, "C", bbox=(0.0, 0.0, 10.0, 10.0)),),
        ),
        bbox=(0.0, 0.0, 20.0, 20.0),
        caption=TableAssociatedText("Caption", bbox=(-5.0, 20.0, 25.0, 25.0)),
        row_bands=(
            TableRowBand(0, bbox=(0.0, 10.0, 20.0, 20.0), kind="header"),
            TableRowBand(1, bbox=(0.0, 0.0, 10.0, 10.0), kind="body"),
        ),
        column_bands=(
            TableColumnBand(0, bbox=(0.0, 0.0, 10.0, 20.0)),
            TableColumnBand(1, bbox=(10.0, 10.0, 20.0, 20.0)),
        ),
    )

    assert table.caption is not None
    assert table.caption.text == "Caption"
    assert table.layout_bbox == (-5.0, 0.0, 25.0, 25.0)
    assert table.content_bbox == (0.0, 0.0, 20.0, 20.0)
    assert all(isinstance(band, TableRowBand) for band in table.row_bands)
    assert all(isinstance(band, TableColumnBand) for band in table.column_bands)
    assert len(table.row_bands) == 2
    assert len(table.column_bands) == 2
    assert [band.kind for band in table.row_bands] == ["header", "body"]


def test_text_view_exposes_lines_and_words() -> None:
    page = Page(
        page_number=1,
        blocks=(
            Block(
                order=0,
                kind=BlockKind.PARAGRAPH,
                lines=(
                    TextLine(text="one two", source="native"),
                    TextLine(text="three", source="ocr"),
                ),
            ),
        ),
    )

    assert [line.text for line in page.text_view.lines] == ["one two", "three"]
    assert [word.text for word in page.text_view.words] == ["one", "two", "three"]
    assert [word.block_index for word in page.text_view.words] == [0, 0, 0]
    assert [word.page_number for word in page.text_view.words] == [1, 1, 1]


def test_nodes_are_stable_in_reading_order() -> None:
    first = Block(order=4, kind=BlockKind.PARAGRAPH)
    second = Block(order=1, kind=BlockKind.PARAGRAPH)
    page = Page(page_number=1, blocks=(first, second))

    assert [node.node_id for node in page.nodes] == [0, 1]
    assert [node.payload for node in page.nodes] == [second, first]


def test_document_views_preserve_page_boundaries() -> None:
    first = Page(page_number=1, blocks=(Block(order=0, kind=BlockKind.PARAGRAPH),))
    second = Page(
        page_number=2,
        tables=(Table(rows=((TableCell(row=0, column=0, text="cell"),),), order=0),),
    )
    document = Document(pages=(first, second))

    assert len(document.text_view.pages) == 2
    assert document.table_view.tables == second.tables
    assert document.text == "\fcell\f"
    assert [node.page_number for node in document.nodes] == [1, 2]
    assert [node.node_id for node in document.nodes] == [0, 1]


def test_document_text_view_preserves_word_page_ownership() -> None:
    document = Document(
        pages=(
            Page(
                page_number=1,
                blocks=(Block(0, BlockKind.PARAGRAPH, (TextLine("first"),)),),
            ),
            Page(
                page_number=7,
                blocks=(Block(0, BlockKind.PARAGRAPH, (TextLine("second"),)),),
            ),
        )
    )

    assert [word.text for word in document.text_view.words] == ["first", "second"]
    assert [word.page_number for word in document.text_view.words] == [1, 7]


def test_document_text_view_preserves_line_page_ownership() -> None:
    line = TextLine("line")
    document = Document(
        pages=(Page(page_number=4, blocks=(Block(0, BlockKind.PARAGRAPH, (line,)),)),)
    )

    assert document.text_view.line_references == (
        TextLineReference(page_number=4, line_index=0, line=line),
    )
    assert cast(Any, document.to_json_dict())["lines"][0]["page_id"] == "p4"


def test_reference_ownership_preserves_zero_page_numbers() -> None:
    line = TextLine("zero")
    table = Table(order=0, rows=((TableCell(0, 0, "cell"),),))
    document = Document(
        pages=(
            Page(
                page_number=0,
                blocks=(Block(0, BlockKind.PARAGRAPH, (line,)),),
                tables=(table,),
            ),
        )
    )

    assert document.text_view.line_references[0].page_number == 0
    assert document.table_view.references[0].page_number == 0


def test_document_table_view_preserves_table_page_ownership() -> None:
    table = Table(order=0, rows=((TableCell(0, 0, "cell"),),))
    document = Document(pages=(Page(page_number=4, tables=(table,)),))

    references = document.table_view.references

    assert references == (TableReference(page_number=4, table_index=0, table=table),)
    payload = cast(Any, document.to_json_dict())
    assert payload["tables"][0]["page_id"] == "p4"


def test_node_provenance_and_page_diagnostics_are_serializable() -> None:
    table = Table(
        order=0,
        rows=((TableCell(row=0, column=0, text="cell"),),),
        metadata={"source": "stream"},
    )
    page = Page(
        page_number=1,
        tables=(table,),
        diagnostics=(Diagnostic(code="table-check", message="checked"),),
    )

    assert page.nodes[0].provenance == ("stream",)
    payload = cast(Any, Document(pages=(page,)).to_json_dict())
    assert payload["pages"][0]["diagnostics"][0]["code"] == "table-check"
    assert payload["pages"][0]["node_ids"] == ["p1:node:0"]
    assert payload["nodes"][0]["id"] == "p1:node:0"
    assert payload["nodes"][0]["target_id"] == "p1:table:0"
    assert payload["nodes"][0]["provenance"] == ["stream"]


def test_canonical_table_projection_is_serializable() -> None:
    table = Table(
        order=0,
        rows=((TableCell(row=0, column=0, text="cell"),),),
    )
    page = Page(page_number=1, tables=(table,))

    payload = cast(Any, Document(pages=(page,)).to_json_dict())

    assert payload["tables"] == [
        {
            "id": "p1:table:0",
            "page_id": "p1",
            "order": 0,
            "bbox": None,
            "layout_bbox": None,
            "content_bbox": None,
            "confidence": None,
            "title": None,
            "caption": None,
            "row_bands": [],
            "column_bands": [],
            "metadata": {},
            "rows": [
                [
                    {
                        "row": 0,
                        "column": 0,
                        "text": "cell",
                        "row_span": 1,
                        "column_span": 1,
                        "bbox": None,
                    }
                ]
            ],
        }
    ]


def test_structured_table_serializes_associations_and_band_geometry() -> None:
    table = Table(
        order=2,
        rows=(
            (TableCell(0, 0, "Header", bbox=(0.0, 10.0, 20.0, 20.0)),),
            (TableCell(1, 0, "Value", bbox=(0.0, 0.0, 20.0, 10.0)),),
        ),
        bbox=(0.0, 0.0, 20.0, 20.0),
        title=TableAssociatedText("Table title", bbox=(0.0, 22.0, 20.0, 30.0), kind="title"),
        row_bands=(
            TableRowBand(0, bbox=(0.0, 10.0, 20.0, 20.0), kind="header"),
            TableRowBand(1, bbox=(0.0, 0.0, 20.0, 10.0), kind="body"),
        ),
        column_bands=(TableColumnBand(0, bbox=(0.0, 0.0, 20.0, 20.0)),),
    )

    document = Document(pages=(Page(page_number=1, tables=(table,)),))
    payload = cast(Any, document.to_json_dict())
    serialized = payload["tables"][0]

    assert serialized["layout_bbox"] == [0.0, 0.0, 20.0, 30.0]
    assert serialized["content_bbox"] == [0.0, 0.0, 20.0, 20.0]
    assert serialized["title"] == {
        "text": "Table title",
        "bbox": [0.0, 22.0, 20.0, 30.0],
        "kind": "title",
        "confidence": None,
    }
    assert serialized["caption"] is None
    assert serialized["row_bands"]
    assert serialized["column_bands"]


def test_structured_table_rendering_uses_associations_and_row_kinds() -> None:
    table = Table(
        order=0,
        rows=(
            (TableCell(0, 0, "Table title"),),
            (TableCell(1, 0, "Name"), TableCell(1, 1, "Value")),
            (TableCell(2, 0, "A"), TableCell(2, 1, "1")),
        ),
        title=TableAssociatedText("Table title", kind="title"),
        row_bands=(
            TableRowBand(0, kind="title"),
            TableRowBand(1, kind="header"),
            TableRowBand(2, kind="body"),
        ),
    )

    html = Page(page_number=1, tables=(table,)).to_html()

    assert '<div data-table-associated="title">Table title</div>' in html
    assert '<tr data-row-kind="header"><th>Name</th><th>Value</th></tr>' in html
    assert "<td>Table title</td>" not in html
