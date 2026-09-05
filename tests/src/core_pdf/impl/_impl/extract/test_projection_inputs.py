from __future__ import annotations

import numpy

from core_pdf.impl._impl.extract.block_layout import layout_blocks_with_evidence
from core_pdf.impl._impl.extract.contracts import ObservationBatch, ParsedBlock, ParsedLine
from core_pdf.impl._impl.extract.emit import internal_compose_page, internal_normalized_blocks
from core_pdf.impl._impl.extract.table_reconcile import (
    internal_profile_tables,
    internal_remove_block_duplicate_tables,
)
from core_pdf.impl._impl.output.model import Block, BlockKind, Table, TableCell, TextLine
from core_pdf.impl.types import TextWord


def test_layout_uses_supplied_source_labels_and_group_order_without_mutating_observations() -> None:
    batch = ObservationBatch.from_columns(
        ("right", "left"),
        ((50.0, 10.0, 80.0, 20.0), (10.0, 10.0, 40.0, 20.0)),
        source=7,
    )

    def order_group(observations: ObservationBatch, indexes: numpy.ndarray) -> numpy.ndarray:
        return indexes[numpy.argsort(observations.bbox[indexes, 0])]

    blocks, evidence = layout_blocks_with_evidence(
        batch,
        source_labels={7: "external"},
        group_order=order_group,
    )

    line = blocks[0].lines[0]
    assert line.text == "left right"
    assert line.source == "external"
    assert [(word.text, word.bbox, word.source) for word in line.words] == [
        ("left", (10.0, 10.0, 40.0, 20.0), "external"),
        ("right", (50.0, 10.0, 80.0, 20.0), "external"),
    ]
    assert batch.text == ("right", "left")
    assert batch.bbox.tolist() == [[50.0, 10.0, 80.0, 20.0], [10.0, 10.0, 40.0, 20.0]]
    assert evidence.rotation_count == 1


def test_block_normalizer_reconciles_words_before_composition() -> None:
    bbox = (10.0, 10.0, 40.0, 20.0)
    parsed = ParsedBlock(
        lines=(
            ParsedLine(
                "bad",
                bbox,
                "external",
                words=(TextWord("bad", bbox),),
            ),
        ),
        bbox=bbox,
    )
    calls: list[tuple[str, str]] = []

    def normalize(text: str, source: str) -> str:
        calls.append((text, source))
        return "good value"

    normalized = internal_normalized_blocks((parsed,), (), normalize_text=normalize)
    page = internal_compose_page(
        (parsed,),
        normalized,
        (),
        page_number=1,
        width=100.0,
        height=100.0,
        rotation=0,
        route="external",
    )

    assert calls == [("bad", "external")]
    assert page.blocks[0].text == "good value"
    assert [(word.text, word.bbox) for word in page.blocks[0].lines[0].words] == [
        ("good", None),
        ("value", None),
    ]
    assert parsed.lines[0].text == "bad"
    assert page.base_route == "external"


def test_table_projection_accepts_explicit_protections_and_rejections() -> None:
    bbox = (10.0, 10.0, 80.0, 20.0)
    text = "one two three four"
    blocks = [
        Block(order=0, kind=BlockKind.PARAGRAPH, lines=(TextLine(text, bbox=bbox),), bbox=bbox)
    ]
    duplicate = Table(order=0, rows=((TableCell(0, 0, text),),), bbox=bbox)
    rejected = Table(order=1, rows=((TableCell(0, 0, "unique"),),), bbox=bbox)

    tables = internal_profile_tables((duplicate, rejected))
    assert internal_remove_block_duplicate_tables(blocks, tables) == tables[1:]
    assert (
        internal_remove_block_duplicate_tables(
            blocks,
            tables,
            protected_table_indexes=frozenset({0}),
            rejected_table_indexes=frozenset({1}),
        )
        == tables[:1]
    )
