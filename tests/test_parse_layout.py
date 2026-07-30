from __future__ import annotations

from core_pdf.impl.engine.parse import ObservationBatch, ObservationSource, layout_blocks


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
