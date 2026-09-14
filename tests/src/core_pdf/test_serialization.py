import csv
import json
from dataclasses import replace
from io import StringIO
from typing import Any
from xml.etree import ElementTree

import pytest

from core_pdf.impl._impl.output.model import (
    Annotation,
    Block,
    BlockKind,
    Diagnostic,
    Document,
    Figure,
    FormField,
    Link,
    Page,
    Table,
    TableAssociatedText,
    TableCell,
    TableColumnBand,
    TableRowBand,
    TextLine,
    TextSpan,
)


def block(text: str, order: int = 0, **kwargs: Any) -> Block:
    return Block(order, BlockKind.PARAGRAPH, (TextLine(text),), **kwargs)


def test_json_preserves_shared_identity_without_merging_equal_objects() -> None:
    shared = TextLine('quoted "text" & <tag>', bbox=(1, 2, 3, 4))
    first = Block(0, BlockKind.PARAGRAPH, (shared,))
    second = Block(1, BlockKind.PARAGRAPH, (shared, replace(shared)))
    page = Page(7, blocks=(first, first, second))
    document = Document(
        (page, replace(page, page_number=9)), metadata={"nested": {"list": (True, None, 2.5)}}
    )
    result = json.loads(document.to_json())
    assert result == document.to_json_dict()
    assert len(result["blocks"]) == 4
    assert len(result["lines"]) == 4
    assert len(result["nodes"]) == 6
    assert result["blocks"][0]["line_ids"] == ["p7:line:0"]
    assert result["blocks"][1]["line_ids"] == ["p7:line:0", "p7:line:1"]
    assert result["nodes"][0]["target_id"] == result["nodes"][1]["target_id"]
    assert result["lines"][0]["text"] == shared.text
    assert result["lines"][0]["bbox"] == [1, 2, 3, 4]
    assert result["metadata"] == {"nested": {"list": [True, None, 2.5]}}


def test_json_exports_tables_and_nontext_records() -> None:
    table = Table(
        1,
        rows=((TableCell(0, 0, "A", row_span=2, column_span=3),),),
        title=TableAssociatedText("Title", kind="title"),
        caption=TableAssociatedText("Caption"),
        row_bands=(TableRowBand(0, kind="header"),),
        column_bands=(TableColumnBand(0),),
        metadata={"source": "test"},
    )
    page = Page(
        1,
        tables=(table, table),
        figures=(Figure(2, kind="chart"),),
        links=(Link(url='https://example.invalid/?q="<&'),),
        annotations=(Annotation(contents="note", destination={"page": 2}),),
        form_fields=(FormField(name="name", field_type="Tx", value_text="value"),),
        diagnostics=(Diagnostic("page", "warning"),),
    )
    result = Document((page,), diagnostics=(Diagnostic("document", "warning"),)).to_json_dict()
    # Decode the public JSON representation to narrow the heterogeneous payload for assertions.
    data = json.loads(json.dumps(result))
    assert len(data["tables"]) == 1
    assert data["tables"][0]["rows"][0][0]["row_span"] == 2
    assert data["tables"][0]["rows"][0][0]["column_span"] == 3
    assert data["tables"][0]["title"]["text"] == "Title"
    assert data["figures"][0]["kind"] == "chart"
    assert data["links"][0]["url"] == page.links[0].url
    assert data["annotations"][0]["destination"] == {"page": 2}
    assert data["form_fields"][0]["value_text"] == "value"
    assert data["pages"][0]["diagnostics"][0]["code"] == "page"
    assert data["diagnostics"][0]["code"] == "document"


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"bad": object()}, r"\$\.metadata.bad"),
        ({1: "bad"}, r"metadata key"),
        ({"nested": [object()]}, r"nested\[0\]"),
    ],
)
def test_json_reports_invalid_metadata_location(metadata: Any, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        Document((), metadata=metadata).to_json()


def test_json_rejects_duplicate_page_ids() -> None:
    with pytest.raises(ValueError, match="duplicate structured page id: p1"):
        Document((Page(1), Page(1))).to_json()


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (None, [8, 3, 5]),
        (2, [3]),
        ("3-1,2", [5, 3, 8]),
        (range(3, 0, -1), [5, 3, 8]),
        ([2, 2, 1], [3, 8]),
    ],
)
def test_csv_and_tei_preserve_selection_order_and_escape_text(
    selection: Any, expected: list[int]
) -> None:
    text = 'Comma, "quote"\n<tag>& café'
    document = Document(tuple(Page(number, blocks=(block(text),)) for number in (8, 3, 5)))
    rows = list(csv.DictReader(StringIO(document.to_csv(pages=selection))))
    assert [int(row["page_number"]) for row in rows] == expected
    assert all(row["text"] == text and row["line_index"] == "0" and row["x0"] == "" for row in rows)
    root = ElementTree.fromstring(document.to_tei(pages=selection))
    assert [int(pb.attrib["n"]) for pb in root.findall("./text/body/pb")] == expected
    assert [p.text for p in root.findall("./text/body/p")] == [text] * len(expected)


@pytest.mark.parametrize("selection", [0, 4, [], "bad", True])
@pytest.mark.parametrize("format_name", ["to_csv", "to_tei"])
def test_serializers_reject_invalid_selection(selection: Any, format_name: str) -> None:
    with pytest.raises((IndexError, ValueError, TypeError)):
        getattr(Document((Page(1),)), format_name)(pages=selection)


def test_empty_document_and_geometry_exports() -> None:
    document = Document(())
    assert len(list(csv.reader(StringIO(document.to_csv())))) == 1
    assert ElementTree.fromstring(document.to_tei()).findall("./text/body/pb") == []
    assert document.to_json_dict()["pages"] == []
    assert "<article" in document.to_html()
    line = TextLine("geometry", bbox=(1, 2, 3, 4))
    rows = list(
        csv.DictReader(
            StringIO(
                Document((Page(1, blocks=(Block(0, BlockKind.PARAGRAPH, (line,)),)),)).to_csv()
            )
        )
    )
    assert [float(rows[0][key]) for key in ("x0", "y0", "x1", "y1")] == [1, 2, 3, 4]


def test_html_escapes_text_and_preserves_styles_and_element_order() -> None:
    line = TextLine(
        '<>&"', bold=True, italic=True, underline=True, strikeout=True, mark=True, superscript=True
    )
    heading = Block(0, BlockKind.HEADING, (line,), level=3)
    list_line = TextLine(
        "1. <item>", spans=(TextSpan("1. ", bold=True), TextSpan("<item>", subscript=True))
    )
    listing = Block(2, BlockKind.LIST, (list_line, TextLine("plain")))
    paragraph = Block(3, BlockKind.PARAGRAPH, (TextLine("first"), TextLine("second")))
    figure = Figure(1, kind='chart" onload="bad')
    html = Document((Page(1, blocks=(paragraph, listing, heading), figures=(figure,)),)).to_html()
    root = ElementTree.fromstring(html)
    section = root.find("section")
    assert section is not None
    assert [item.tag for item in section] == ["h3", "figure", "ul", "p"]
    assert "".join(section[0].itertext()) == line.text
    assert section[0].find(".//sup") is not None
    assert section[1].attrib == {"data-kind": figure.kind}
    assert ["".join(item.itertext()) for item in section[2]] == ["<item>", "plain"]
    assert section[2].find(".//sub") is not None
    assert section[3].find("br") is not None


@pytest.mark.parametrize(
    "bands", [(), ("header", "body"), ("body", "body"), ("title", "caption"), ("header", "header")]
)
@pytest.mark.parametrize("associated", [False, True])
def test_html_and_markdown_tables_preserve_spans_and_associated_text(
    bands: tuple[str, ...], associated: bool
) -> None:
    table = Table(
        0,
        rows=((TableCell(0, 0, "<head>", column_span=2),), (TableCell(1, 0, "A&B", row_span=2),)),
        row_bands=tuple(TableRowBand(i, kind=kind) for i, kind in enumerate(bands)),
        title=TableAssociatedText("<title>", kind="title") if associated else None,
        caption=TableAssociatedText("A&B", kind="caption") if associated else None,
    )
    document = Document((Page(1, tables=(table,)),))
    html = document.to_html()
    markdown = document.to_markdown().rstrip("\f")
    assert markdown in html
    root = ElementTree.fromstring(f"<root>{markdown}</root>")
    if associated:
        assert [node.text for node in root.findall("div")] == ["<title>", "A&B"]
    if bands == ("title", "caption"):
        assert root.findall(".//tr") == []
    else:
        cells = root.findall(".//th") + root.findall(".//td")
        assert {node.text for node in cells} == {"<head>", "A&B"}
        assert any(node.get("colspan") == "2" for node in cells)
        assert any(node.get("rowspan") == "2" for node in cells)


def test_empty_table_and_mismatched_explicit_bands() -> None:
    assert "<table></table>" in Document((Page(1, tables=(Table(0),)),)).to_html()
    table = Table(0, rows=((TableCell(0, 0, "x"),),), row_bands=(TableRowBand(0), TableRowBand(1)))
    with pytest.raises(ValueError, match="zip"):
        Document((Page(1, tables=(table,)),)).to_html()
