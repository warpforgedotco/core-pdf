import numpy
import pytest

from core_pdf.impl import extract_block_layout as native_layout
from core_pdf.impl.extract_contracts import ObservationBatch
from core_pdf_ocr.impl.extract import block_layout


def make_batch(texts=("first", "second", "third"), *, rotation=0, source=1):
    return ObservationBatch.from_columns(
        texts,
        ((0, 0, 10, 10), (20, 20, 30, 30), (40, 40, 50, 50)),
        source=source,
        rotation=(rotation,) * 3,
        confidence=(99,) * 3,
    )


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [(0, [0, 1, 2]), (90, [0, 1, 2]), (180, [2, 1, 0]), (270, [2, 1, 0]), (450, [0, 1, 2])],
)
def test_ocr_order_uses_the_rotated_writing_axis(rotation, expected) -> None:
    indexes = numpy.array([2, 0, 1])
    assert block_layout.group_order(make_batch(rotation=rotation), indexes).tolist() == expected
    assert indexes.tolist() == [2, 0, 1]


def test_native_group_preserves_original_index_identity() -> None:
    indexes = numpy.array([2, 0, 1])
    assert block_layout.group_order(make_batch(source=0), indexes) is indexes


@pytest.mark.parametrize(
    ("texts", "expected"),
    [(("א", "ب", "١"), [2, 1, 0]), (("é!", "!", "漢"), [0, 1, 2]), (("א", "A", "1"), [0, 1, 2])],
)
def test_unicode_direction_votes_reverse_only_strong_rtl_majority(texts, expected) -> None:
    assert block_layout.group_order(make_batch(texts), numpy.arange(3)).tolist() == expected


def test_equal_positions_keep_input_order() -> None:
    observations = ObservationBatch.from_columns(("A", "B"), ((0, 0, 10, 10),) * 2, source=1)
    assert block_layout.group_order(observations, numpy.array([1, 0])).tolist() == [1, 0]


def test_ocr_group_order_orders_group_text_and_words() -> None:
    observations = ObservationBatch.from_columns(
        ("left", "right"), ((0, 0, 20, 10), (30, 0, 60, 10)), source=1
    )
    indexes = block_layout.group_order(observations, numpy.array([1, 0]))
    text, words = native_layout.group_text_and_words(observations, indexes)
    assert text == "left right"
    assert tuple(word.text for word in words) == ("left", "right")


def test_layout_preserves_ocr_source_labels_and_evidence() -> None:
    observations = ObservationBatch.from_columns(
        ("Hello", "world"), ((10, 10, 35, 20), (40, 10, 70, 20)), source=1, confidence=(99, 99)
    )
    blocks, evidence = block_layout.layout_blocks_with_evidence(
        observations, page_width=100, page_height=100
    )
    assert len(blocks) == 1
    assert blocks[0].lines[0].line.source == "ocr"
    assert blocks[0].lines[0].line.text == "Hello world"
    assert evidence.line_count == 1
    assert not evidence.ambiguous
