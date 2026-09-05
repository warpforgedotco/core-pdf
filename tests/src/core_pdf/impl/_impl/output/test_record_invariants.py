# SPDX-License-Identifier: AGPL-3.0-only
"""Canonical text and identity must agree across output projections."""

from dataclasses import replace
from typing import Any, cast

import pytest

from core_pdf.impl._impl.extract.contracts import ParsedBlock, ParsedLine
from core_pdf.impl._impl.extract.emit import internal_normalized_blocks
from core_pdf.impl._impl.output.model import (
    Block,
    BlockKind,
    Document,
    Figure,
    Page,
    Table,
    TableCell,
    TextLine,
    TextSpan,
)
from core_pdf.impl.types import TextWord
from tests.helpers.pdf_bytes import one_page_pdf, open_pdf


@pytest.mark.parametrize("kind", [BlockKind.PARAGRAPH, BlockKind.HEADING, BlockKind.LIST])
def test_replacing_line_text_cannot_leave_old_text_in_styled_output(kind: BlockKind) -> None:
    line = TextLine("old", spans=(TextSpan("old", bold=True),), words=(TextWord("old"),))

    updated = replace(line, text="new")
    page = Page(1, blocks=(Block(0, kind, (updated,)),))
    document = Document((page,))

    assert updated.spans == ()
    assert tuple(word.text for word in updated.words) == ("new",)
    assert page.text == "new"
    for rendered in (document.to_html(), document.to_markdown()):
        assert "old" not in rendered
        assert "new" in rendered
    json_line = cast(Any, document.to_json_dict())["lines"][0]
    assert json_line["text"] == "new"
    assert "".join(span["text"] for span in json_line["spans"]) == "new"
    assert line.spans == (TextSpan("old", bold=True),)


def test_emission_preserves_line_end_hyphens_and_matching_styles() -> None:
    box = (0.0, 0.0, 20.0, 10.0)
    parsed = ParsedBlock(
        lines=(
            ParsedLine("multi-", box, "native", spans=(TextSpan("multi-", bold=True),)),
            ParsedLine("line", box, "native", spans=(TextSpan("line", italic=True),)),
        ),
        bbox=box,
    )

    blocks = internal_normalized_blocks((parsed,), ())
    page = Page(1, blocks=tuple(blocks))

    assert page.text == "multi-\nline"
    assert "<strong>multi-</strong>" in page.to_html()
    assert "multi-" in page.to_markdown()
    assert "<em>line</em>" in page.to_html()


def test_a_verified_styled_pdf_retains_matching_inline_spans() -> None:
    # Poppler 26.07.0 pdftotext -layout verified the exact "old" text before
    # this regression; the PDF's font resource explicitly selects Helvetica-Bold.
    pdf = one_page_pdf(
        b"BT /F1 12 Tf 20 250 Td (old) Tj ET",
        media_box=(0, 0, 200, 300),
        font=b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    )
    with open_pdf(pdf) as source:
        document = source.extract()

    line = document.pages[0].blocks[0].lines[0]
    assert line.text == "old"
    assert line.spans
    assert "".join(span.text for span in line.spans) == "old"
    assert "<strong>old</strong>" in document.to_html()


def test_shared_payloads_have_one_json_record_and_keep_every_ordered_occurrence() -> None:
    line = TextLine("word")
    block = Block(2, BlockKind.PARAGRAPH, (line,))
    table = Table(0, rows=((TableCell(0, 0, "cell"),),))
    figure = Figure(1)
    page = Page(1, blocks=(block, block), tables=(table, table), figures=(figure, figure))
    document = Document((page, replace(page, page_number=2)))

    payload = cast(Any, document.to_json_dict())

    for number in (1, 2):
        page_id = f"p{number}"
        for kind in ("block", "table", "figure", "line"):
            records = [record for record in payload[f"{kind}s"] if record["page_id"] == page_id]
            assert [record["id"] for record in records] == [f"{page_id}:{kind}:0"]
        nodes = [node for node in payload["nodes"] if node["page_id"] == page_id]
        assert [node["target_id"] for node in nodes] == [
            f"{page_id}:table:0",
            f"{page_id}:table:0",
            f"{page_id}:figure:0",
            f"{page_id}:figure:0",
            f"{page_id}:block:0",
            f"{page_id}:block:0",
        ]
        assert len({node["id"] for node in nodes}) == 6


def test_equal_but_distinct_payloads_remain_distinct_and_can_share_a_line() -> None:
    line = TextLine("word")
    first = Block(0, BlockKind.PARAGRAPH, (line,))
    second = replace(first)

    payload = cast(Any, Document((Page(1, blocks=(first, second)),)).to_json_dict())

    assert first == second
    assert first is not second
    assert [block["id"] for block in payload["blocks"]] == ["p1:block:0", "p1:block:1"]
    assert [block["line_ids"] for block in payload["blocks"]] == [["p1:line:0"], ["p1:line:0"]]
    assert len(payload["lines"]) == 1
