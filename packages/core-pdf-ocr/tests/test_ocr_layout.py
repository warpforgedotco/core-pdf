from __future__ import annotations

from core_pdf.impl._impl.extract.block_layout import (
    internal_column_major_prose,
    internal_reading_order_evidence,
)
from core_pdf_ocr.impl.extract.block_layout import (
    layout_blocks,
    layout_blocks_with_evidence,
)
from core_pdf_ocr.impl.extract.contracts import (
    ObservationBatch,
    ObservationSource,
    ParsedBlock,
    ParsedLine,
)


def test_rotated_ocr_words_group_on_vertical_baseline() -> None:
    batch = ObservationBatch.from_columns(
        ("top", "bottom"),
        ((10.0, 30.0, 20.0, 40.0), (10.0, 10.0, 20.0, 20.0)),
        source=ObservationSource.OCR,
        confidence=(90.0, 90.0),
        rotation=(90, 90),
    )

    blocks, evidence = layout_blocks_with_evidence(batch)

    assert [line.text for block in blocks for line in block.lines] == ["bottom top"]
    assert evidence.rotation_count == 1


def test_rtl_ocr_words_read_right_to_left_within_line() -> None:
    batch = ObservationBatch.from_columns(
        ("עולם", "שלום"),
        ((10.0, 10.0, 30.0, 20.0), (40.0, 10.0, 60.0, 20.0)),
        source=ObservationSource.OCR,
        confidence=(90.0, 90.0),
    )

    blocks = layout_blocks(batch)

    assert [line.text for block in blocks for line in block.lines] == ["שלום עולם"]


def test_mixed_rotations_in_one_block_are_reported_as_ambiguous() -> None:
    block = ParsedBlock(
        lines=(
            ParsedLine("body", (0.0, 20.0, 40.0, 30.0), "native", sequence=0),
            ParsedLine(
                "margin note",
                (50.0, 0.0, 60.0, 40.0),
                "native",
                sequence=1,
                rotation=90,
            ),
        ),
        bbox=(0.0, 0.0, 60.0, 40.0),
    )

    evidence = internal_reading_order_evidence((block,))

    assert evidence.ambiguous
    assert evidence.confidence == 0.5
    assert evidence.rotation_count == 2
    assert evidence.column_count == 1


def test_column_major_prose_recovers_two_columns_with_a_header_cluster() -> None:
    lines = [
        ParsedLine(
            text=f"{column} line {index}",
            bbox=(left, 800.0 - index * 8.0, left + 80.0, 806.0 - index * 8.0),
            source="ocr",
        )
        for index in range(80)
        for column, left in (("left", 40.0), ("right", 240.0))
    ]
    # A centered heading creates a third x-start cluster without being a column.
    lines.insert(
        0,
        ParsedLine(
            text="Centered heading",
            bbox=(140.0, 820.0, 220.0, 826.0),
            source="ocr",
        ),
    )
    block = ParsedBlock(
        lines=tuple(lines),
        bbox=(40.0, 0.0, 320.0, 826.0),
    )

    ordered = internal_column_major_prose([block])[0].lines

    assert [line.text for line in ordered] == [
        *(f"left line {index}" for index in range(80)),
        "Centered heading",
        *(f"right line {index}" for index in range(80)),
    ]
