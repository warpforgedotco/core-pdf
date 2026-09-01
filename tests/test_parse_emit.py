from __future__ import annotations

from typing import cast

from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.models import TextWord
from core_pdf.impl.parse import (
    ObservationBatch,
    ObservationSource,
    PageRoute,
    ParsedBlock,
    ParsedLine,
    ParsedPage,
)
from core_pdf.impl.parse.emit import assemble_page as emit_page
from core_pdf.impl.parse.emit import internal_line_decoration_flags
from core_pdf.impl.parse.layout import layout_blocks
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing
from core_pdf.impl.structured import BlockKind, Table, TableCell


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


def test_emit_preserves_distinct_word_boxes_computed_by_layout() -> None:
    first = TextRun("one", 10.0, 100.0, 40.0, 110.0, 0.0, 0.0, 10.0, 4.0, 0, 0, 0)
    second = TextRun("two", 50.0, 100.0, 80.0, 110.0, 0.0, 0.0, 10.0, 4.0, 1, 1, 0)
    observations = ObservationBatch.from_columns(
        (first.text, second.text),
        ((first.x0, first.y0, first.x1, first.y1), (second.x0, second.y0, second.x1, second.y1)),
        source=ObservationSource.NATIVE,
        line_break_before=(True, False),
        references=(first, second),
    )
    parsed = ParsedPage(
        page_number=1,
        width=200.0,
        height=200.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=layout_blocks(observations),
    )

    page = emit_page(parsed)

    assert [(word.text, word.bbox) for word in page.words] == [
        ("one", (10.0, 100.0, 40.0, 110.0)),
        ("two", (50.0, 100.0, 80.0, 110.0)),
    ]
    assert page.blocks[0].lines[0].bbox == (10.0, 100.0, 80.0, 110.0)


def test_emit_reconciles_word_geometry_with_normalized_text() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=200.0,
        height=200.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            ParsedBlock(
                lines=(
                    ParsedLine(
                        "ABΗCD",
                        (10.0, 100.0, 60.0, 110.0),
                        "native",
                        words=(TextWord("ABΗCD", (10.0, 100.0, 60.0, 110.0)),),
                    ),
                    ParsedLine(
                        'value "',
                        (10.0, 80.0, 70.0, 90.0),
                        "native",
                        words=(
                            TextWord("value", (10.0, 80.0, 50.0, 90.0)),
                            TextWord('"', (60.0, 80.0, 70.0, 90.0)),
                        ),
                    ),
                ),
                bbox=(10.0, 80.0, 70.0, 110.0),
            ),
        ),
    )

    page = emit_page(parsed)

    first, second = page.blocks[0].lines
    assert [(word.text, word.bbox) for word in first.words] == [
        ("ABHCD", (10.0, 100.0, 60.0, 110.0))
    ]
    assert [(word.text, word.bbox) for word in second.words] == [("value", None)]


def test_emit_attaches_caption_and_section_to_table() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=200.0,
        height=300.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block("Revenue", (10.0, 250.0, 100.0, 270.0), "heading"),
            block("Table 1: Revenue by year", (10.0, 215.0, 150.0, 225.0), "caption"),
        ),
        tables=(
            Table(
                order=0,
                bbox=(10.0, 150.0, 150.0, 205.0),
                rows=((TableCell(0, 0, "Year"),),),
            ),
        ),
    )

    page = emit_page(parsed)

    table = page.tables[0]
    assert table.metadata["caption"] == "Table 1: Revenue by year"
    assert table.metadata["section"] == "Revenue"
    assert table.metadata["section_level"] == 1
    assert any(block.kind is BlockKind.HEADING for block in page.blocks)


def test_emit_preserves_structured_reading_order_diagnostic() -> None:
    parsed = ParsedPage(
        page_number=2,
        width=200.0,
        height=300.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(block("Mixed direction", (10.0, 250.0, 100.0, 270.0)),),
        diagnostics=("reading-order-ambiguous",),
    )

    page = emit_page(parsed)

    assert len(page.diagnostics) == 1
    diagnostic = page.diagnostics[0]
    assert diagnostic.code == "reading-order-ambiguous"
    assert diagnostic.page_number == 2
    assert "differently rotated text" in diagnostic.message


def test_line_decoration_flags_support_partial_underlines() -> None:
    line = ParsedLine("prefix B suffix", (10.0, 100.0, 110.0, 110.0), "native")
    drawing = type("Drawing", (), {"kind": "fill", "bbox": (50.0, 99.5, 56.0, 100.5)})()

    assert internal_line_decoration_flags(line, (drawing,))["underline"]


def test_emit_materializes_line_decoration_bbox_once() -> None:
    calls = 0

    class Path:
        def bbox(self) -> tuple[float, float, float, float]:
            nonlocal calls
            calls += 1
            return (50.0, 99.5, 56.0, 100.5)

    drawing = type("Drawing", (), {"kind": "fill", "bbox": None, "path": Path()})()
    parsed = ParsedPage(
        page_number=1,
        width=200.0,
        height=300.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block("first", (10.0, 100.0, 110.0, 110.0)),
            block("second", (10.0, 100.0, 110.0, 110.0)),
        ),
    )

    page = emit_page(parsed, (cast(CapturedDrawing, drawing),))

    assert calls == 1
    assert all(line.underline for block in page.blocks for line in block.lines)


def test_emit_removes_blocks_duplicated_by_table_cells() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=200.0,
        height=300.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block("Introduction text", (10.0, 250.0, 100.0, 270.0)),
            block("Year Value 2024 10", (10.0, 150.0, 150.0, 205.0)),
            block("Afterword text", (10.0, 100.0, 100.0, 120.0)),
        ),
        tables=(
            Table(
                order=0,
                bbox=(10.0, 150.0, 150.0, 205.0),
                rows=(
                    (TableCell(0, 0, "Year"), TableCell(0, 1, "Value")),
                    (TableCell(1, 0, "2024"), TableCell(1, 1, "10")),
                ),
                metadata={"source": "stream"},
            ),
        ),
    )

    page = emit_page(parsed)

    assert {block.text for block in page.blocks} == {"Introduction text", "Afterword text"}
    assert len(page.tables) == 1
    assert page.tables[0].rows[0][0].text == "Year"


def test_emit_keeps_line_that_contains_but_does_not_duplicate_a_table() -> None:
    """A small table inside an oversized line box must not delete the line.

    Some valid font descriptors use extreme font-wide ascent and descent
    values. Their line boxes can contain a nearby table even when their text
    does not belong to it, so containment has to be measured in the direction
    of line-inside-table rather than by the smaller of the two areas.
    """
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block("Normative body text after the table", (20.0, 150.0, 280.0, 260.0)),
            block("Bit position Meaning", (20.0, 240.0, 280.0, 260.0)),
        ),
        tables=(
            Table(
                order=0,
                bbox=(20.0, 240.0, 280.0, 260.0),
                rows=((TableCell(0, 0, "Bit position"), TableCell(0, 1, "Meaning")),),
            ),
        ),
    )

    page = emit_page(parsed)

    assert "Normative body text after the table" in page.text
    assert len(page.tables) == 1


def test_emit_preserves_structured_stream_table_with_sparse_numeric_values() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=200.0,
        height=300.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block(
                "Supporting Policy Form: ICC18 ENT-06 1805 Data Page", (10.0, 150.0, 150.0, 205.0)
            ),
        ),
        tables=(
            Table(
                order=0,
                bbox=(10.0, 150.0, 150.0, 205.0),
                rows=(
                    (TableCell(0, 0, "Supporting Policy Form:"),),
                    (TableCell(1, 0, "ICC18 ENT-06 1805"), TableCell(1, 1, "Data Page")),
                ),
                metadata={"source": "stream", "numeric_cells": 2},
            ),
        ),
    )

    page = emit_page(parsed)

    assert len(page.tables) == 1


def test_emit_removes_character_spaced_stream_table_duplicated_by_block() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=200.0,
        height=300.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(block("NGL Pipelines & Services $ 10 $ 9", (10.0, 150.0, 150.0, 205.0)),),
        tables=(
            Table(
                order=0,
                bbox=(10.0, 150.0, 150.0, 205.0),
                rows=(
                    (
                        TableCell(0, 0, "N G L P i p e l i n e s"),
                        TableCell(0, 1, "S e r v i c e s"),
                    ),
                    (TableCell(1, 0, "$ 1 0"), TableCell(1, 1, "$ 9")),
                ),
                metadata={"source": "stream"},
            ),
        ),
    )

    page = emit_page(parsed)

    assert [block.text for block in page.blocks] == ["NGL Pipelines & Services $ 10 $ 9"]
    assert page.tables == ()


def test_emit_removes_stream_table_covered_by_overlapping_blocks() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block(
                "A. Loading Temperature Fuel Oxidizer Tank Volume Line Volume Pressure",
                (20.0, 200.0, 260.0, 260.0),
            ),
            block(
                "Final Weight Bleed Unit Resulting Load Specification Nominal Load",
                (20.0, 145.0, 260.0, 205.0),
            ),
        ),
        tables=(
            Table(
                order=0,
                bbox=(20.0, 140.0, 260.0, 265.0),
                rows=(
                    (
                        TableCell(0, 0, "A. Loading Temperature"),
                        TableCell(0, 1, "Fuel Oxidizer"),
                    ),
                    (
                        TableCell(1, 0, "Tank Volume Line Volume"),
                        TableCell(1, 1, "Pressure"),
                    ),
                    (
                        TableCell(2, 0, "Final Weight Bleed Unit"),
                        TableCell(2, 1, "Resulting Load Specification Nominal Load"),
                    ),
                ),
                metadata={"source": "stream"},
            ),
        ),
    )

    page = emit_page(parsed)

    assert len(page.blocks) == 2
    assert page.tables == ()


def test_emit_keeps_a_grid_shaped_table_and_drops_the_duplicate_block() -> None:
    """A table and blocks describing one region both hold the same glyphs.

    One of them has to go, and dropping the table is the wrong way round: the
    text survives either way, but only the table carries the rows and columns.
    Emission used to resolve this in favour of blocks in every case, which
    discarded the structure of genuine tables -- the dominant reason a page
    with a table in the ground truth came back with none.
    """
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block(
                "Marketplace assigned policy number Policy issuer name Recipient SSN "
                "Recipient spouse SSN Policy termination date Street address State "
                "province Country ZIP foreign postal code",
                (20.0, 140.0, 260.0, 265.0),
            ),
        ),
        tables=(
            Table(
                order=0,
                bbox=(20.0, 140.0, 260.0, 265.0),
                rows=(
                    (
                        TableCell(0, 0, "Marketplace assigned policy number"),
                        TableCell(0, 1, "Policy issuer name"),
                    ),
                    (
                        TableCell(1, 0, "Recipient SSN"),
                        TableCell(1, 1, "Recipient spouse SSN"),
                    ),
                    (
                        TableCell(2, 0, "Policy termination date Street address"),
                        TableCell(2, 1, "State province Country ZIP foreign postal code"),
                    ),
                ),
                metadata={"section": "Part I Recipient Information"},
            ),
        ),
    )

    page = emit_page(parsed)

    # The rows use their columns, so this is a table and it is kept.
    assert len(page.tables) == 1
    assert [[cell.text for cell in row] for row in page.tables[0].rows][0] == [
        "Marketplace assigned policy number",
        "Policy issuer name",
    ]
    # ...and the block repeating it is removed, so the text appears once.
    assert page.blocks == ()


def test_emit_removes_small_table_duplicated_by_page_text() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block(
                "A. Monthly enrollment premiums B. Monthly second lowest cost silver plan",
                (20.0, 320.0, 260.0, 340.0),
            ),
        ),
        tables=(
            Table(
                order=0,
                bbox=(20.0, 120.0, 260.0, 190.0),
                rows=(
                    (
                        TableCell(0, 0, "A. Monthly enrollment premiums"),
                        TableCell(0, 1, "B. Monthly second lowest cost silver plan"),
                    ),
                ),
            ),
        ),
    )

    page = emit_page(parsed)

    assert len(page.blocks) == 1
    assert page.tables == ()


def test_emit_removes_corrupt_native_blocks() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block(
                "76391*11 IOIIlo6 I * 9 2*0 118)96 '1'1322) '1'19)20 IZZO "
                '1911*2.1,z,z CSM/l":OST L*O*Io',
                (20.0, 200.0, 260.0, 260.0),
            ),
            block(
                "Service module loading parameters remain available as ordinary text",
                (20.0, 145.0, 260.0, 165.0),
            ),
        ),
    )

    page = emit_page(parsed)

    assert [block.text for block in page.blocks] == [
        "Service module loading parameters remain available as ordinary text"
    ]


def test_emit_removes_fragmented_stream_table() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(block("Scheduled maturities of debt", (20.0, 300.0, 260.0, 320.0)),),
        tables=(
            Table(
                order=0,
                bbox=(20.0, 140.0, 260.0, 265.0),
                rows=tuple(
                    (
                        TableCell(row, 0, "T o t a l o f 2 0 2 3"),
                        TableCell(row, 1, "S e n i o r N o t e s 2 6 2 7 5"),
                    )
                    for row in range(8)
                ),
                metadata={"source": "stream"},
            ),
        ),
    )

    page = emit_page(parsed)

    assert page.tables == ()


def test_emit_removes_noisy_stream_table() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(block("Mass properties table", (20.0, 300.0, 260.0, 320.0)),),
        tables=(
            Table(
                order=0,
                bbox=(20.0, 140.0, 260.0, 265.0),
                rows=tuple(
                    (
                        TableCell(row, 0, "I 0 ll 13 8 7 o o"),
                        TableCell(row, 1, "531o6 10llfo2 relDc c t2 l"),
                    )
                    for row in range(9)
                ),
                metadata={"source": "stream"},
            ),
        ),
    )

    page = emit_page(parsed)

    assert page.tables == ()


def test_emit_removes_short_corrupt_native_fragments() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block("r • r-", (20.0, 300.0, 40.0, 320.0)),
            block("Normal short text", (20.0, 250.0, 260.0, 270.0)),
        ),
    )

    page = emit_page(parsed)

    assert [block.text for block in page.blocks] == ["Normal short text"]


def test_emit_keeps_short_non_latin_native_text() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block("日本語かなカナ漢字", (20.0, 300.0, 260.0, 320.0)),
            block("你好", (20.0, 250.0, 260.0, 270.0)),
            block("（）", (20.0, 200.0, 260.0, 220.0)),
        ),
    )

    page = emit_page(parsed)

    assert {emitted_block.text for emitted_block in page.blocks} == {
        "日本語かなカナ漢字",
        "你好",
        "（）",
    }


def test_emit_removes_symbol_only_native_blocks() -> None:
    # Isolated Braille glyphs or stray symbols with no alphanumeric content
    # carry no semantic text (cf. the OCR route's `internal_corrupt_ocr_block`).
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block("⠭ ⠬", (20.0, 300.0, 260.0, 320.0)),
            block("Real sentence describing the payload", (20.0, 250.0, 260.0, 270.0)),
            # CJK fullwidth punctuation is preserved (letter-like content).
            block("（）", (20.0, 200.0, 260.0, 220.0)),
        ),
    )

    page = emit_page(parsed)

    assert [emitted_block.text for emitted_block in page.blocks] == [
        "Real sentence describing the payload",
        "（）",
    ]


def test_emit_removes_corrupt_mixed_native_fragments() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
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


def test_emit_removes_punctuation_only_ocr_fragments() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
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
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
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


def test_emit_normalizes_latin_context_confusable_characters() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block("Fax: 699-3395 35149؛", (20.0, 300.0, 260.0, 320.0)),
            block("Model 1Η64-061 and count ١ ٦", (20.0, 250.0, 260.0, 270.0)),
            block("□ Reading Order Detection", (20.0, 200.0, 260.0, 220.0)),
            block("☐ Check one ☒ selected ❖ note", (20.0, 150.0, 260.0, 170.0)),
            block("GREEN OLED |;", (20.0, 100.0, 260.0, 120.0)),
            block("Total ] amount _ due", (20.0, 50.0, 260.0, 70.0)),
            block('Footer " continued', (20.0, 20.0, 260.0, 40.0)),
        ),
    )

    page = emit_page(parsed)

    assert {block.text for block in page.blocks} == {
        "Fax: 699-3395 35149",
        "Model 1H64-061 and count 1 6",
        "Reading Order Detection",
        "Check one selected note",
        "GREEN OLED",
        "Total amount due",
        "Footer continued",
    }


def test_emit_removes_numeric_only_ocr_pipe_artifacts() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.OCR,
        blocks=(
            block(
                "400 | 400 | 400\n137.0 | 128.1",
                (20.0, 120.0, 260.0, 150.0),
                source="ocr",
            ),
        ),
    )

    page = emit_page(parsed)

    assert page.text == "400 400 400\n137.0 128.1"


def test_emit_removes_sparse_ocr_pipe_artifacts() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.OCR,
        blocks=(
            block(
                "total | 46 | 69\nR-21 | | 12",
                (20.0, 120.0, 260.0, 150.0),
                source="ocr",
            ),
        ),
    )

    page = emit_page(parsed)

    assert page.text == "total 46 69\nR-21 12"


def test_emit_removes_lone_ocr_pipe_artifact_lines() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.OCR,
        blocks=(block("|\nvalid text", (20.0, 120.0, 260.0, 150.0), source="ocr"),),
    )

    page = emit_page(parsed)

    assert page.text == "valid text"


def test_emit_keeps_ocr_pipes_in_prose_lines() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.OCR,
        blocks=(
            block(
                "55 Cyril Magnin Street | San Francisco, CA | 94102",
                (20.0, 120.0, 260.0, 150.0),
                source="ocr",
            ),
        ),
    )

    page = emit_page(parsed)

    assert page.text == "55 Cyril Magnin Street | San Francisco, CA | 94102"


def test_emit_removes_numeric_ocr_angle_marker_artifacts() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.OCR,
        blocks=(
            block(
                "> 1\n> quoted text remains",
                (20.0, 120.0, 260.0, 150.0),
                source="ocr",
            ),
        ),
    )

    page = emit_page(parsed)

    assert page.text == "1\n> quoted text remains"


def test_emit_removes_sparse_ocr_symbol_artifacts() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.OCR,
        blocks=(block("• 42 87\nvalid � text", (20.0, 120.0, 260.0, 150.0), source="ocr"),),
    )

    page = emit_page(parsed)

    assert page.text == "42 87\nvalid text"


def test_emit_removes_ocr_standalone_punctuation_artifacts() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.OCR,
        blocks=(
            block(
                "I agree [ to pay\n' Business Fax Number\n! Excess mark",
                (20.0, 120.0, 260.0, 150.0),
                source="ocr",
            ),
        ),
    )

    page = emit_page(parsed)

    assert page.text == "I agree to pay\nBusiness Fax Number\nExcess mark"


def test_emit_removes_isolated_leading_zero_ocr_artifact_tokens() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.OCR,
        blocks=(
            block(
                "04\n07 U6.2\nModel 01 remains",
                (20.0, 120.0, 260.0, 150.0),
                source="ocr",
            ),
        ),
    )

    page = emit_page(parsed)

    assert page.text == "\nU6.2\nModel 01 remains"


def test_emit_keeps_embedded_ocr_punctuation() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.OCR,
        blocks=(block("Warning! keep excited text", (20.0, 120.0, 260.0, 150.0), source="ocr"),),
    )

    page = emit_page(parsed)

    assert page.text == "Warning! keep excited text"


def test_emit_normalizes_intrusive_punctuation_inside_tokens() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block(
                "7!19/71 T[34 G! (%! Warning!",
                (20.0, 120.0, 260.0, 150.0),
                source="native",
            ),
        ),
    )

    page = emit_page(parsed)

    assert page.text == "7!19/71 T34 G! (% Warning!"


def test_emit_removes_soft_line_end_hyphens_before_lowercase_continuations() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            ParsedBlock(
                lines=(
                    ParsedLine(
                        text="gover-",
                        bbox=(20.0, 140.0, 260.0, 150.0),
                        source="native",
                    ),
                    ParsedLine(
                        text="nance responds",
                        bbox=(20.0, 120.0, 260.0, 130.0),
                        source="native",
                    ),
                ),
                bbox=(20.0, 120.0, 260.0, 150.0),
            ),
        ),
    )

    page = emit_page(parsed)

    assert page.text == "gover\nnance responds"


def test_emit_keeps_non_continuation_line_end_hyphens() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            ParsedBlock(
                lines=(
                    ParsedLine(
                        text="Part A-",
                        bbox=(20.0, 140.0, 260.0, 150.0),
                        source="native",
                    ),
                    ParsedLine(
                        text="Next item",
                        bbox=(20.0, 120.0, 260.0, 130.0),
                        source="native",
                    ),
                ),
                bbox=(20.0, 120.0, 260.0, 150.0),
            ),
        ),
    )

    page = emit_page(parsed)

    assert page.text == "Part A-\nNext item"


def test_emit_keeps_ocr_bullets_in_wordlike_lines() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.OCR,
        blocks=(block("• Complete application", (20.0, 120.0, 260.0, 150.0), source="ocr"),),
    )

    page = emit_page(parsed)

    assert page.text == "• Complete application"


def test_emit_removes_nonword_bullets_for_native_lines() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(block("• 42 87", (20.0, 120.0, 260.0, 150.0), source="native"),),
    )

    page = emit_page(parsed)

    assert page.text == "42 87"


def test_emit_removes_line_initial_ocr_suffix_fragments() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.OCR,
        blocks=(
            block(
                "ing groups include, for example",
                (20.0, 120.0, 260.0, 150.0),
                source="ocr",
            ),
        ),
    )

    page = emit_page(parsed)

    assert page.text == "groups include, for example"


def test_emit_removes_expanded_line_initial_suffix_fragments() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block(
                (
                    "ence on global dynamics\n"
                    "tions within states\n"
                    "ating shifts continue\n"
                    "ducted studies"
                ),
                (20.0, 120.0, 260.0, 150.0),
                source="native",
            ),
        ),
    )

    page = emit_page(parsed)

    assert page.text == "on global dynamics\nwithin states\nshifts continue\nstudies"


def test_emit_removes_native_replacement_character_tokens() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(block("valid � text", (20.0, 120.0, 260.0, 150.0), source="native"),),
    )

    page = emit_page(parsed)

    assert page.text == "valid text"


def test_emit_keeps_short_ocr_suffix_lines() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.OCR,
        blocks=(block("ing 12", (20.0, 120.0, 260.0, 150.0), source="ocr"),),
    )

    page = emit_page(parsed)

    assert page.text == "ing 12"


def test_emit_removes_blocks_outside_page_bounds() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(
            block("visible", (20.0, 300.0, 80.0, 320.0)),
            block("off page", (20.0, 430.0, 80.0, 450.0)),
        ),
    )

    page = emit_page(parsed)

    assert [block.text for block in page.blocks] == ["visible"]


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
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.HYBRID,
        blocks=(block("Caption", (20.0, 300.0, 260.0, 320.0)),),
        tables=(synthetic, stream),
    )

    page = emit_page(parsed)

    assert len(page.tables) == 1
    assert page.tables[0].rows == synthetic.rows
    assert page.tables[0].order == 1


def test_emit_removes_tiny_synthetic_chart_table_covered_by_table() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
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
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
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


def test_emit_removes_tiny_bi_artifact_table() -> None:
    parsed = ParsedPage(
        page_number=1,
        width=300.0,
        height=400.0,
        rotation=0,
        route=PageRoute.NATIVE,
        blocks=(block("Figure caption", (20.0, 300.0, 260.0, 320.0), "caption"),),
        tables=(
            Table(
                order=0,
                bbox=(20.0, 200.0, 260.0, 230.0),
                rows=(
                    (TableCell(0, 0, "B I"),),
                    (TableCell(1, 0, ""),),
                ),
            ),
        ),
    )

    page = emit_page(parsed)

    assert page.tables == ()
