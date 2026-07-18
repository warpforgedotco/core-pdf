from typing import Any, cast

import pytest
from core_document import (
    Block,
    BlockKind,
    Document,
    DocumentAdapter,
    DocumentEditor,
    Page,
    Table,
    TableCell,
    TextLine,
)


def test_document_adapter_protocol_is_runtime_checkable_by_shape() -> None:
    class Adapter:
        def apply(self, document: Document) -> Document:
            return document

    adapter: DocumentAdapter = Adapter()

    assert adapter.apply(Document()).pages == ()


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

    assert payload["schema_version"] == "1.0"
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

    assert "| Name | Value |" in document.to_markdown()
    assert "<th>Name</th>" in document.to_html()
    payload = cast(Any, document.to_json_dict())
    assert payload["pages"][0]["tables"]
