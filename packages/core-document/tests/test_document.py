from typing import Any, cast

from core_document import Block, BlockKind, Document, Page, Table, TableCell, TextLine


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
