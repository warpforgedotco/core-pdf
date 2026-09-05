from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from core_pdf.impl._impl.output.model import Figure, Page, Table, TableCell
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing
from core_pdf_ocr.impl.extract.contracts import (
    PageRoute,
    ParsedBlock,
    ParsedLine,
)
from core_pdf_ocr.impl.extract.emit import assemble_page


def block(
    text: str,
    bbox: tuple[float, float, float, float],
    kind: str = "paragraph",
    source: str = "native",
) -> ParsedBlock:
    return ParsedBlock(
        lines=(ParsedLine(text=text, bbox=bbox, source=source),),
        bbox=bbox,
        kind=kind,
        level=1 if kind == "heading" else None,
    )


@dataclass(frozen=True, slots=True)
class PageInput:
    page_number: int
    width: float
    height: float
    rotation: int
    route: PageRoute
    blocks: tuple[ParsedBlock, ...]
    tables: tuple[Table, ...] = ()
    figures: tuple[Figure, ...] = ()
    diagnostics: tuple[str, ...] = ()
    full_page_image: bool = False


def page_of(
    *,
    route: PageRoute = PageRoute.NATIVE,
    blocks: tuple[ParsedBlock, ...] = (),
    width: float = 300.0,
    height: float = 400.0,
    **fields: Any,
) -> PageInput:
    return PageInput(
        page_number=1,
        width=width,
        height=height,
        rotation=0,
        route=route,
        blocks=blocks,
        **fields,
    )


def single_block_page(
    text: str,
    route: PageRoute = PageRoute.NATIVE,
    bbox: tuple[float, float, float, float] = (20.0, 120.0, 260.0, 150.0),
) -> PageInput:
    source = "ocr" if route is PageRoute.OCR else "native"
    return page_of(route=route, blocks=(block(text, bbox, source=source),))


def emit_page(parsed: PageInput, drawings: tuple[CapturedDrawing, ...] = ()) -> Page:
    return assemble_page(
        parsed.blocks,
        page_number=parsed.page_number,
        width=parsed.width,
        height=parsed.height,
        rotation=parsed.rotation,
        route=parsed.route,
        tables=parsed.tables,
        figures=parsed.figures,
        diagnostics=parsed.diagnostics,
        full_page_image=parsed.full_page_image,
        drawings=drawings,
    )


def test_emit_removes_punctuation_only_ocr_fragments() -> None:
    parsed = page_of(
        route=PageRoute.OCR,
        blocks=(
            ParsedBlock(
                lines=(ParsedLine(text="~", bbox=(20.0, 300.0, 40.0, 320.0), source="ocr"),),
                bbox=(20.0, 300.0, 40.0, 320.0),
            ),
            ParsedBlock(
                lines=(
                    ParsedLine(
                        text="Civil Division",
                        bbox=(20.0, 250.0, 260.0, 270.0),
                        source="ocr",
                    ),
                ),
                bbox=(20.0, 250.0, 260.0, 270.0),
            ),
        ),
    )

    page = emit_page(parsed)

    assert [block.text for block in page.blocks] == ["Civil Division"]


def test_emit_removes_tiny_ocr_fragments_duplicated_by_table_tokens() -> None:
    parsed = page_of(
        route=PageRoute.HYBRID,
        blocks=(
            ParsedBlock(
                lines=(ParsedLine(text="10\n23", bbox=(250.0, 20.0, 290.0, 40.0), source="ocr"),),
                bbox=(250.0, 20.0, 290.0, 40.0),
            ),
            block("IRB Statistics", (20.0, 300.0, 260.0, 320.0), "heading"),
        ),
        tables=(
            Table(
                order=0,
                bbox=(20.0, 120.0, 260.0, 220.0),
                rows=(
                    (TableCell(0, 0, "BCoIS"), TableCell(0, 1, "10")),
                    (TableCell(1, 0, "CCPS"), TableCell(1, 1, "23")),
                ),
            ),
        ),
    )

    page = emit_page(parsed)

    assert [block.text for block in page.blocks] == ["IRB Statistics"]


def test_emit_removes_stream_table_covered_by_synthetic_chart_table() -> None:
    synthetic = Table(
        order=0,
        bbox=(20.0, 140.0, 260.0, 180.0),
        rows=((TableCell(0, 0, "musculoskeletal diseases 182"),),),
        metadata={"source": "chart-ocr", "synthetic": True},
    )
    stream = Table(
        order=1,
        bbox=(20.0, 140.0, 260.0, 180.0),
        rows=((TableCell(0, 0, "musculoskeletal diseases 182 Public Health 2022"),),),
        metadata={"source": "stream"},
    )
    parsed = page_of(
        route=PageRoute.HYBRID,
        blocks=(block("Caption", (20.0, 300.0, 260.0, 320.0)),),
        tables=(synthetic, stream),
    )

    page = emit_page(parsed)

    assert len(page.tables) == 1
    assert page.tables[0].rows == synthetic.rows
    assert page.tables[0].order == 1


def test_emit_removes_tiny_synthetic_chart_table_covered_by_table() -> None:
    parsed = page_of(
        route=PageRoute.NATIVE,
        blocks=(block("IRB Statistics", (20.0, 300.0, 260.0, 320.0), "heading"),),
        tables=(
            Table(
                order=0,
                bbox=(20.0, 200.0, 260.0, 230.0),
                rows=((TableCell(0, 0, "1 11 14 13 17 23 12 10"),),),
                metadata={"source": "chart-ocr", "synthetic": True},
            ),
            Table(
                order=1,
                bbox=(20.0, 120.0, 260.0, 190.0),
                rows=(
                    (TableCell(0, 0, "Unit Determinations"), TableCell(0, 1, "FY18")),
                    (TableCell(1, 0, "CCPS"), TableCell(1, 1, "1 11 14 13")),
                    (TableCell(2, 0, "SCoB"), TableCell(2, 1, "17 23 12 10")),
                ),
            ),
        ),
    )

    page = emit_page(parsed)

    assert len(page.tables) == 1
    assert page.tables[0].metadata.get("source") != "chart-ocr"


def test_emit_keeps_chart_table_that_covers_tiny_synthetic_table() -> None:
    parsed = page_of(
        route=PageRoute.NATIVE,
        blocks=(
            block(
                "IRB Statistics Unit Determinations FY18 1 11 14 13 17 23 12 10",
                (20.0, 120.0, 260.0, 230.0),
                "heading",
            ),
        ),
        tables=(
            Table(
                order=0,
                bbox=(20.0, 200.0, 260.0, 230.0),
                rows=((TableCell(0, 0, "1 11 14 13 17 23 12 10"),),),
                metadata={"source": "chart-ocr", "synthetic": True},
            ),
            Table(
                order=1,
                bbox=(20.0, 120.0, 260.0, 190.0),
                rows=(
                    (TableCell(0, 0, "Unit Determinations"), TableCell(0, 1, "FY18")),
                    (TableCell(1, 0, "CCPS"), TableCell(1, 1, "1 11 14 13")),
                    (TableCell(2, 0, "SCoB"), TableCell(2, 1, "17 23 12 10")),
                ),
            ),
        ),
    )

    page = emit_page(parsed)

    assert len(page.tables) == 1
    assert "Unit Determinations" in page.tables[0].rows[0][0].text


@pytest.mark.parametrize(
    ("route", "text", "expected"),
    [
        pytest.param(
            PageRoute.OCR,
            "400 | 400 | 400\n137.0 | 128.1",
            "400 400 400\n137.0 128.1",
            id="removes-numeric-only-pipes",
        ),
        pytest.param(
            PageRoute.OCR,
            "total | 46 | 69\nR-21 | | 12",
            "total 46 69\nR-21 12",
            id="removes-sparse-pipes",
        ),
        pytest.param(PageRoute.OCR, "|\nvalid text", "valid text", id="removes-lone-pipe-lines"),
        pytest.param(
            PageRoute.OCR,
            "55 Cyril Magnin Street | San Francisco, CA | 94102",
            "55 Cyril Magnin Street | San Francisco, CA | 94102",
            id="keeps-pipes-in-prose",
        ),
        pytest.param(
            PageRoute.OCR,
            "> 1\n> quoted text remains",
            "1\n> quoted text remains",
            id="removes-numeric-angle-markers",
        ),
        pytest.param(
            PageRoute.OCR, "• 42 87\nvalid � text", "42 87\nvalid text", id="removes-sparse-symbols"
        ),
        pytest.param(
            PageRoute.OCR,
            "I agree [ to pay\n' Business Fax Number\n! Excess mark",
            "I agree to pay\nBusiness Fax Number\nExcess mark",
            id="removes-standalone-punctuation",
        ),
        pytest.param(
            PageRoute.OCR,
            "04\n07 U6.2\nModel 01 remains",
            "\nU6.2\nModel 01 remains",
            id="removes-isolated-leading-zero-tokens",
        ),
        pytest.param(
            PageRoute.OCR,
            "Warning! keep excited text",
            "Warning! keep excited text",
            id="keeps-embedded-punctuation",
        ),
        pytest.param(
            PageRoute.OCR,
            "• Complete application",
            "• Complete application",
            id="keeps-ocr-bullets-in-wordlike-lines",
        ),
        pytest.param(
            PageRoute.OCR,
            "ing groups include, for example",
            "groups include, for example",
            id="removes-line-initial-suffix-fragment",
        ),
        pytest.param(PageRoute.OCR, "ing 12", "ing 12", id="keeps-short-ocr-suffix-lines"),
    ],
)
def test_emit_normalizes_single_block_text(route: PageRoute, text: str, expected: str) -> None:
    assert emit_page(single_block_page(text, route)).text == expected


def test_emit_removes_corrupt_mixed_native_fragments() -> None:
    parsed = page_of(
        route=PageRoute.NATIVE,
        blocks=(
            ParsedBlock(
                lines=(
                    ParsedLine(
                        text="2 C5M 11 2/LHt O ExECTEo 5[8U[NTIAL HA55 ·RO¸ERTt[5",
                        bbox=(20.0, 300.0, 260.0, 320.0),
                        source="ocr",
                    ),
                    ParsedLine(
                        text="cOORDINAT£5 TARLE loi•21CONTfNUEOI",
                        bbox=(20.0, 320.0, 260.0, 340.0),
                        source="native",
                    ),
                ),
                bbox=(20.0, 300.0, 260.0, 340.0),
            ),
            block("Normal short text", (20.0, 250.0, 260.0, 270.0)),
        ),
    )

    page = emit_page(parsed)

    assert [block.text for block in page.blocks] == ["Normal short text"]
