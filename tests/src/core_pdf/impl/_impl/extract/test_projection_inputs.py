from __future__ import annotations

import numpy

from core_pdf.impl._impl.extract.block_layout import layout_blocks_with_evidence
from core_pdf.impl._impl.extract.contracts import ObservationBatch, ParsedBlock, ParsedLine
from core_pdf.impl._impl.extract.emit import internal_compose_page, internal_normalized_blocks
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


def test_block_normalizer_preserves_prepared_text_and_word_geometry() -> None:
    bbox = (10.0, 10.0, 40.0, 20.0)
    word = TextWord("value", bbox)
    parsed = ParsedBlock(
        lines=(ParsedLine("value", bbox, "external", words=(word,)),),
        bbox=bbox,
    )
    normalized = internal_normalized_blocks((parsed,), ())
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
    assert page.blocks[0].text == "value"
    assert page.blocks[0].lines[0].words == (word,)
    assert page.base_route == "external"
