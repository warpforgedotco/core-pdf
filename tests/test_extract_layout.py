from __future__ import annotations

import numpy
import pytest

from core_pdf.impl._impl.extract.block_layout import (
    internal_group_text_and_words,
    layout_blocks,
    layout_blocks_with_evidence,
)
from core_pdf.impl._impl.extract.contracts import (
    ObservationBatch,
    ObservationSource,
)
from core_pdf.impl._impl.extract.regions import (
    internal_best_projection_gap,
    internal_best_region_projection_gap,
    internal_interval_crossing_counts,
    internal_LayoutGeometry,
    internal_row_order_indexes,
    internal_row_order_region,
)
from core_pdf.impl._impl.layout import reconstruction
from core_pdf.impl._impl.model.runs import LayoutLineText, LayoutLineTextSegment
from tests.helpers import extract_fakes


def test_native_line_reuses_one_reconstruction_for_text_and_word_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = LayoutLineText(
        " left\tright ",
        (
            LayoutLineTextSegment("left", "", (0.0, 0.0, 4.0, 2.0), 0),
            LayoutLineTextSegment("right", "\t", (10.0, 0.0, 15.0, 2.0), 0),
        ),
    )
    calls = 0

    def reconstruct(*args: object, **kwargs: object) -> LayoutLineText:
        nonlocal calls
        calls += 1
        return result

    monkeypatch.setattr(reconstruction, "reconstruct_layout_line_text", reconstruct)
    batch = extract_fakes.observations(
        (("left right", (0.0, 0.0, 15.0, 2.0)),),
        references=(extract_fakes.text_run("left right"),),
    )

    text, words = internal_group_text_and_words(batch, numpy.array([0]))

    assert calls == 1
    assert text == "left\tright"
    assert tuple(word.text for word in words) == ("left", "right")
    assert tuple(word.bbox for word in words) == ((0.0, 0.0, 4.0, 2.0), (10.0, 0.0, 15.0, 2.0))


def observations(
    items: tuple[tuple[str, tuple[float, float, float, float]], ...],
) -> ObservationBatch:
    """Confident text observations that each start a line."""
    return extract_fakes.observations(
        items,
        source=ObservationSource.NATIVE,
        confidence=90.0,
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


def test_reusable_region_orders_match_independent_sorting() -> None:
    random = numpy.random.default_rng(21082026)
    starts = random.uniform(0.0, 500.0, (300, 2))
    sizes = random.uniform(1.0, 80.0, (300, 2))
    boxes = numpy.column_stack((starts, starts + sizes)).reshape(300, 4)[:, (0, 1, 2, 3)]
    # column_stack above produces x0, y0, x1, y1 for the two-column operands.
    geometry = internal_LayoutGeometry.create(boxes)
    root_indexes = numpy.arange(len(boxes))
    root = geometry.region(root_indexes)

    for _sample in range(50):
        indexes = numpy.sort(random.choice(root_indexes, size=100, replace=False))
        region = geometry.region(indexes, root)
        for axis in (0, 1):
            assert internal_best_region_projection_gap(geometry, region, axis, 0.0) == (
                internal_best_projection_gap(boxes[indexes], axis, 0.0)
            )
        numpy.testing.assert_array_equal(
            internal_row_order_region(geometry, region),
            internal_row_order_indexes(indexes, boxes),
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
        source=ObservationSource.NATIVE,
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


def test_native_layout_preserves_symbol_heavy_decoded_lines() -> None:
    # pdftotext -layout and pdftoppm 26.07.0 preserve and render this exact
    # line, including its arrow, dashes, and dots, in a WinAnsi PDF.
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

    assert [line.text for block in blocks for line in block.lines] == [
        "heading",
        "2Y--> -- a IIIISCI...........................",
        "R1 +5V",
    ]


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
