from __future__ import annotations

import numpy

from core_pdf.impl.engine.parse import ObservationBatch, ObservationSource, layout_blocks
from core_pdf.impl.engine.parse.layout import (
    internal_column_major_prose,
    internal_interval_crossing_counts,
    internal_reading_order_evidence,
    internal_topological_block_order,
    internal_topological_block_order_quadratic,
    layout_blocks_with_evidence,
)
from core_pdf.impl.engine.parse.model import ParsedBlock, ParsedLine


def observations(
    items: tuple[tuple[str, tuple[float, float, float, float]], ...],
) -> ObservationBatch:
    return ObservationBatch.from_columns(
        (text for text, internal_box in items),
        (box for internal_text, box in items),
        source=ObservationSource.OCR,
        confidence=(90.0 for internal_item in items),
        line_break_before=(True for internal_item in items),
    )


def test_interval_crossing_counts_match_strict_broadcast_oracle() -> None:
    random = numpy.random.default_rng(20260821)
    starts = random.uniform(-50.0, 500.0, 1_000)
    ends = starts + random.uniform(0.0, 120.0, len(starts))
    boxes = numpy.column_stack((starts, starts, ends, ends))
    positions = numpy.concatenate((random.uniform(-100.0, 650.0, 300), starts, ends))

    expected = ((starts[None, :] < positions[:, None]) & (ends[None, :] > positions[:, None])).sum(
        axis=1
    )

    numpy.testing.assert_array_equal(internal_interval_crossing_counts(boxes, positions), expected)


def test_xy_cut_reads_columns_before_moving_right() -> None:
    batch = observations(
        (
            ("header", (0.0, 80.0, 100.0, 90.0)),
            ("left one", (0.0, 60.0, 40.0, 70.0)),
            ("right one", (60.0, 60.0, 100.0, 70.0)),
            ("left two", (0.0, 40.0, 40.0, 50.0)),
            ("right two", (60.0, 40.0, 100.0, 50.0)),
        )
    )

    blocks = layout_blocks(batch)

    assert [line.text for block in blocks for line in block.lines] == [
        "header",
        "left one",
        "left two",
        "right one",
        "right two",
    ]
    assert blocks[0].column_index is None
    assert {block.column_index for block in blocks[1:]} == {0, 1}


def test_reading_order_evidence_records_geometric_column_repair() -> None:
    batch = observations(
        (
            ("left one", (0.0, 60.0, 40.0, 70.0)),
            ("right one", (60.0, 60.0, 100.0, 70.0)),
            ("left two", (0.0, 40.0, 40.0, 50.0)),
            ("right two", (60.0, 40.0, 100.0, 50.0)),
        )
    )

    blocks, evidence = layout_blocks_with_evidence(batch)

    assert [line.text for block in blocks for line in block.lines] == [
        "left one",
        "left two",
        "right one",
        "right two",
    ]
    assert evidence.repaired
    assert evidence.source_inversions == 1
    assert evidence.source_inversion_ratio == 1 / 6
    assert evidence.column_count == 2
    assert evidence.confidence == 1.0
    assert not evidence.ambiguous


def test_reading_order_evidence_preserves_authored_observation_sequence() -> None:
    batch = ObservationBatch.from_columns(
        ("left one", "right one", "left two", "right two"),
        (
            (0.0, 60.0, 40.0, 70.0),
            (60.0, 60.0, 100.0, 70.0),
            (0.0, 40.0, 40.0, 50.0),
            (60.0, 40.0, 100.0, 50.0),
        ),
        source=ObservationSource.OCR,
        sequence=(10, 30, 20, 40),
        confidence=(90.0, 90.0, 90.0, 90.0),
        line_break_before=(True, True, True, True),
    )

    blocks, evidence = layout_blocks_with_evidence(batch)

    assert [(line.text, line.sequence) for block in blocks for line in block.lines] == [
        ("left one", 10),
        ("left two", 20),
        ("right one", 30),
        ("right two", 40),
    ]
    assert evidence.source_inversions == 0
    assert not evidence.repaired
    assert evidence.strategy == "source-stable"


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

    assert ordered[0].text == "left line 0"
    assert ordered[81].text == "right line 0"


def test_full_width_obstacle_keeps_surrounding_regions_separate() -> None:
    batch = observations(
        (
            ("above", (0.0, 60.0, 100.0, 70.0)),
            ("inside", (10.0, 30.0, 90.0, 40.0)),
            ("below", (0.0, 0.0, 100.0, 10.0)),
        )
    )

    blocks = layout_blocks(batch, obstacles=((0.0, 20.0, 100.0, 50.0),))

    assert [[line.text for line in block.lines] for block in blocks] == [
        ["above"],
        ["inside"],
        ["below"],
    ]


def test_native_layout_drops_symbol_heavy_artifact_lines() -> None:
    batch = ObservationBatch.from_columns(
        ("heading", "2Y--> -- a IIIISCI...........................", "R1 +5V"),
        (
            (0.0, 40.0, 60.0, 50.0),
            (0.0, 20.0, 120.0, 30.0),
            (0.0, 0.0, 45.0, 10.0),
        ),
        source=ObservationSource.NATIVE,
        confidence=(0.84, 0.84, 0.84),
        line_break_before=(True, True, True),
    )

    blocks = layout_blocks(batch)

    assert [line.text for block in blocks for line in block.lines] == ["heading", "R1 +5V"]


def test_native_layout_drops_repeated_single_letter_artifacts() -> None:
    batch = ObservationBatch.from_columns(
        ("heading", "I", "B", "I", "B", "I", "B", "I", "B", "body"),
        tuple(
            (0.0, float((9 - index) * 20), 60.0, float((9 - index) * 20 + 10))
            for index in range(10)
        ),
        source=ObservationSource.NATIVE,
        confidence=(0.84 for internal_item in range(10)),
        line_break_before=(True for internal_item in range(10)),
    )

    blocks = layout_blocks(batch)

    assert [line.text for block in blocks for line in block.lines] == ["heading", "body"]


def test_layout_assigns_conservative_semantic_block_roles() -> None:
    batch = ObservationBatch.from_columns(
        ("Annual report", "Revenue increased", "Figure 1: Results", "- first item"),
        (
            (0.0, 72.0, 100.0, 86.0),
            (0.0, 48.0, 100.0, 58.0),
            (0.0, 24.0, 140.0, 34.0),
            (0.0, 0.0, 100.0, 10.0),
        ),
        source=ObservationSource.NATIVE,
        font_size=(18.0, 10.0, 10.0, 10.0),
        line_break_before=(True, True, True, True),
    )

    blocks = layout_blocks(batch, use_xy_cut=False)

    assert [block.kind for block in blocks] == ["heading", "paragraph", "caption", "list"]


def test_sparse_block_order_matches_quadratic_oracle() -> None:
    random = numpy.random.default_rng(27032026)
    for sample in range(40):
        count = int(random.integers(64, 160))
        x0 = random.uniform(0.0, 500.0, count)
        y0 = random.uniform(0.0, 800.0, count)
        widths = random.uniform(5.0, 180.0, count)
        heights = random.uniform(4.0, 80.0, count)
        blocks = [
            ParsedBlock(
                lines=(
                    ParsedLine(
                        text=f"{sample}:{index}",
                        bbox=(
                            float(x0[index]),
                            float(y0[index]),
                            float(x0[index] + widths[index]),
                            float(y0[index] + heights[index]),
                        ),
                        source="native",
                    ),
                ),
                bbox=(
                    float(x0[index]),
                    float(y0[index]),
                    float(x0[index] + widths[index]),
                    float(y0[index] + heights[index]),
                ),
            )
            for index in range(count)
        ]

        expected = internal_topological_block_order_quadratic(blocks)
        actual = internal_topological_block_order(blocks)

        assert [block.lines[0].text for block in actual] == [
            block.lines[0].text for block in expected
        ]
