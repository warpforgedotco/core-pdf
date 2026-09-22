from copy import replace
from typing import Any

import pytest

from core_pdf.impl.output.model import (
    Annotation,
    Block,
    BlockKind,
    ContentNode,
    Document,
    DocumentTableView,
    DocumentTextView,
    Figure,
    Page,
    Table,
    TableCell,
    TableView,
    TextLine,
    TextSpan,
    TextView,
)
from core_pdf.impl.types import TextWord


def test_word_projection_stamps_ownership_without_mutating_shared_lines() -> None:
    first = TextWord("alpha", source="unknown", page_number=99, line_index=99)
    second = TextWord("beta", source="ocr", word_index=99, block_index=99)
    line = TextLine("alpha beta", words=(first, second), source="native")
    block = Block(2, BlockKind.PARAGRAPH, (TextLine(""), line))
    tail = Block(4, BlockKind.PARAGRAPH, (TextLine("tail", words=(TextWord("tail"),)),))
    page = Page(7, blocks=(tail, block), tables=(Table(3),), figures=(Figure(0),))
    other = replace(page, page_number=12)
    document = Document((page, other))
    words = document.words
    assert [word.text for word in words] == ["alpha", "beta", "tail"] * 2
    assert [word.page_number for word in words] == [7] * 3 + [12] * 3
    assert [(word.line_index, word.word_index, word.block_index) for word in words[:3]] == [
        (1, 0, 0),
        (1, 1, 0),
        (2, 0, 1),
    ]
    assert [word.source for word in words[:3]] == ["native", "ocr", "unknown"]
    assert page.words == words[:3]
    assert line.words == (first, second)
    assert document.lines == (block.lines[0], line, tail.lines[0]) * 2
    assert document.lines[1] is line
    assert document.blocks == (block, tail) * 2


def test_document_nodes_keep_order_payload_identity_and_page_ownership() -> None:
    block = Block(3, BlockKind.PARAGRAPH, (TextLine("hello"),), provenance=("native",))
    table = Table(2, metadata={"source": "ocr"}, bbox=(1, 2, 3, 4))
    figure = Figure(1)
    page = Page(8, blocks=(block,), tables=(table,), figures=(figure,))
    document = Document((page, Page(9), replace(page, page_number=10)))
    nodes = document.nodes
    assert [node.node_id for node in nodes] == list(range(6))
    assert [node.page_number for node in nodes] == [8] * 3 + [10] * 3
    assert [node.kind for node in nodes] == ["figure", "table", "block"] * 2
    assert nodes[1].payload is nodes[4].payload is table
    assert nodes[1].bbox == (1, 2, 3, 4)
    assert [node.provenance for node in nodes[:3]] == [(), ("ocr",), ("native",)]
    assert ContentNode(0, "table", Table(0, metadata={"source": 42})).provenance == ("42",)


def test_reference_views_preserve_identity_and_reset_indices_at_page_boundaries() -> None:
    line = TextLine("shared")
    block = Block(0, BlockKind.PARAGRAPH, (line, line))
    view = DocumentTextView((TextView((block,), 9), TextView(()), TextView((block,))))
    assert [(ref.page_number, ref.line_index) for ref in view.line_references] == [
        (9, 0),
        (9, 1),
        (3, 0),
        (3, 1),
    ]
    assert all(ref.line is line for ref in view.line_references)
    table = Table(0)
    tables = DocumentTableView((TableView((table, table), 9), TableView(()), TableView((table,))))
    assert tables.tables == (table,) * 3
    assert [(ref.page_number, ref.table_index) for ref in tables.references] == [
        (9, 0),
        (9, 1),
        (3, 0),
    ]
    assert all(ref.table is table for ref in tables.references)


def test_text_view_omits_empty_elements_but_preserves_table_and_page_boundaries() -> None:
    block = Block(1, BlockKind.PARAGRAPH, (TextLine("one"), TextLine("two", break_before=2)))
    table = Table(2, rows=((TableCell(0, 0, "A"), TableCell(0, 1, "B")),))
    page = Page(
        5, blocks=(block, Block(0, BlockKind.PARAGRAPH)), tables=(table,), figures=(Figure(3),)
    )
    document = Document((page, Page(6)))
    assert page.text == "one\n\ntwo\n\nA\tB"
    assert document.text == "one\n\ntwo\n\nA\tB\f\f"
    assert document.table_view.tables == (table,)
    assert document.table_view.references[0].table is table
    assert document.table_view.references[0].page_number == 5


@pytest.mark.parametrize("owner", [Document, Figure, Table])
def test_metadata_is_a_deep_snapshot(owner: Any) -> None:
    metadata = {"nested": [{"tags": {"a", "b"}}]}
    record = owner(metadata=metadata) if owner is Document else owner(0, metadata=metadata)
    metadata["nested"][0]["tags"].add("later")
    metadata["nested"].append({"tags": {"new"}})
    assert record.metadata["nested"] == ({"tags": frozenset({"a", "b"})},)
    with pytest.raises(TypeError):
        record.metadata["nested"][0]["tags"] = set()


def test_annotation_destination_is_a_deep_snapshot() -> None:
    destination = [3, {"fit": ["XYZ", 0, 10]}]
    annotation = Annotation(destination=destination)
    destination.clear()
    assert annotation.destination == (3, {"fit": ("XYZ", 0, 10)})


@pytest.mark.parametrize("schema", ["", "4.0", "6.0"])
def test_document_rejects_unrecognized_schema(schema: str) -> None:
    with pytest.raises(ValueError, match="unsupported structured schema version"):
        Document(schema_version=schema)


@pytest.mark.parametrize(
    ("underline", "strikeout"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_invalid_span_text_is_discarded_before_style_projection(
    underline: bool, strikeout: bool
) -> None:
    line = TextLine(
        "correct", spans=(TextSpan("stale", bold=True),), underline=underline, strikeout=strikeout
    )
    assert line.spans == ()
    assert line.styled_spans() == (TextSpan("correct", underline=underline, strikeout=strikeout),)


def test_page_dimensions_convert_user_units_without_rotating_stored_geometry() -> None:
    page = Page(1, width=100, height=200, rotation=90, user_unit=2.5)
    assert (page.width_points, page.height_points) == (250, 500)
    assert (page.width, page.height) == (100, 200)


def test_page_serializers_emit_the_owned_text() -> None:
    page = Page(1, blocks=(Block(0, BlockKind.PARAGRAPH, (TextLine("A & <B>"),)),))
    assert "A &amp; &lt;B&gt;" in page.to_html()
    assert "A &" in page.to_markdown()
    assert "B" in page.to_markdown()


@pytest.mark.parametrize("underline", [False, True])
@pytest.mark.parametrize("strikeout", [False, True])
def test_line_decorations_combine_with_span_styles_without_mutation(
    underline: bool, strikeout: bool
) -> None:
    spans = (TextSpan("A", bold=True, underline=True), TextSpan("B", italic=True, strikeout=True))
    line = TextLine("AB", spans=spans, underline=underline, strikeout=strikeout)
    assert line.styled_spans() == (
        TextSpan("A", bold=True, underline=True, strikeout=strikeout),
        TextSpan("B", italic=True, underline=underline, strikeout=True),
    )
    assert line.spans is spans
    assert not spans[0].strikeout
    assert not spans[1].underline
