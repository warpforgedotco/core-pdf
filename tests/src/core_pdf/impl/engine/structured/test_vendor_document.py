from typing import Any, cast

import pytest

from core_pdf.impl.engine.structured import (
    Block,
    BlockKind,
    Document,
    DocumentEditor,
    Figure,
    Page,
    Table,
    TableCell,
    TextLine,
    TextSpan,
)


def test_text_span_serialization_preserves_inline_styles() -> None:
    document = Document(
        pages=(
            Page(
                page_number=1,
                blocks=(
                    Block(
                        1,
                        BlockKind.PARAGRAPH,
                        (
                            TextLine(
                                "plainboldunderlined",
                                spans=(
                                    TextSpan("plain"),
                                    TextSpan("bold", bold=True),
                                    TextSpan("underlined", underline=True),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    assert document.to_markdown() == "plain**bold**<u>underlined</u>\f"
    assert "plain<strong>bold</strong><u>underlined</u>" in document.to_html()


def test_document_editor_commits_without_mutating_the_original() -> None:
    original = Document(
        metadata={"title": "before"},
        pages=(Page(page_number=1, blocks=(Block(1, BlockKind.PARAGRAPH, (TextLine("old"),)),)),),
    )
    replacement = Page(
        page_number=99,
        blocks=(Block(1, BlockKind.PARAGRAPH, (TextLine("new"),)),),
    )

    editor = original.edit()
    assert isinstance(editor, DocumentEditor)
    updated = editor.set_metadata("title", "after").replace_page(1, replacement).commit()

    assert original.metadata["title"] == "before"
    assert original.pages[0].text == "old"
    assert updated.metadata["title"] == "after"
    assert updated.pages[0].text == "new"
    assert updated.pages[0].page_number == 1


def test_document_editor_rollback_closes_transaction() -> None:
    editor = Document().edit()

    editor.rollback()

    with pytest.raises(RuntimeError, match="rolled back"):
        editor.commit()


def test_nested_ir_metadata_is_immutable() -> None:
    document = Document(metadata={"nested": {"values": [1]}})

    with pytest.raises(TypeError):
        document.metadata["nested"]["values"] = (2,)

    assert document.to_json_dict()["metadata"] == {"nested": {"values": [1]}}


def test_document_serializes_to_versioned_json() -> None:
    document = Document(
        metadata={"title": "Example"},
        pages=(
            Page(
                page_number=1,
                width=612.0,
                height=792.0,
                blocks=(
                    Block(
                        order=1,
                        kind=BlockKind.HEADING,
                        level=1,
                        lines=(TextLine("Example"),),
                    ),
                ),
            ),
        ),
    )

    payload = document.to_json_dict()

    assert payload["schema_version"] == "4.0"
    payload_any = cast(Any, payload)
    assert payload_any["pages"][0]["blocks"][0]["kind"] == "heading"


def test_document_views_escape_html_and_render_semantics() -> None:
    document = Document(
        pages=(
            Page(
                page_number=1,
                blocks=(
                    Block(
                        order=1,
                        kind=BlockKind.HEADING,
                        lines=(TextLine("<Title>"),),
                    ),
                ),
            ),
        )
    )

    assert "## <Title>" in document.to_markdown()
    assert '<h2 data-block-kind="heading">&lt;Title&gt;</h2>' in document.to_html()


def test_page_views_are_available_on_the_canonical_page() -> None:
    page = Page(
        page_number=1,
        blocks=(Block(1, BlockKind.PARAGRAPH, (TextLine("hello"),)),),
    )

    assert page.to_markdown() == "hello"
    assert 'data-page-number="1"' in page.to_html()


def test_document_serializes_tables_in_all_views() -> None:
    table = Table(
        order=1,
        rows=(
            (TableCell(0, 0, "Name"), TableCell(0, 1, "Value")),
            (TableCell(1, 0, "A"), TableCell(1, 1, "1")),
        ),
    )
    document = Document(pages=(Page(page_number=1, tables=(table,)),))

    assert "<table><thead><tr><th>Name</th><th>Value</th></tr></thead>" in document.to_markdown()
    assert "<th>Name</th>" in document.to_html()
    payload = cast(Any, document.to_json_dict())
    rows = payload["pages"][0]["tables"][0]["rows"]
    assert [[cell["text"] for cell in row] for row in rows] == [
        ["Name", "Value"],
        ["A", "1"],
    ]


def test_document_serializes_table_spans_without_losing_them_in_markdown() -> None:
    table = Table(
        order=1,
        rows=(
            (TableCell(0, 0, "Section", column_span=2),),
            (TableCell(1, 0, "Group", row_span=2), TableCell(1, 1, "A")),
            (TableCell(2, 1, "B"),),
        ),
    )

    document = Document(pages=(Page(page_number=1, tables=(table,)),))

    assert '<th colspan="2">Section</th>' in document.to_markdown()
    assert '<td rowspan="2">Group</td>' in document.to_markdown()
    assert '<th colspan="2">Section</th>' in document.to_html()


def test_page_views_follow_unified_element_order() -> None:
    page = Page(
        page_number=1,
        blocks=(Block(2, BlockKind.PARAGRAPH, (TextLine("after"),)),),
        figures=(Figure(1, kind="image"),),
        tables=(Table(3, rows=((TableCell(0, 0, "table"),),)),),
    )

    assert page.elements[0].order == 1
    assert page.to_markdown().splitlines() == [
        "> [Figure: image]",
        "",
        "after",
        "",
        "<table><thead><tr><th>table</th></tr></thead><tbody></tbody></table>",
    ]
    payload = cast(Any, Document(pages=(page,)).to_json_dict())
    assert payload["pages"][0]["elements"][0]["element_type"] == "figure"
