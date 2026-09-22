import numpy
import pytest

from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf_ocr.impl.extract import block_layout


def internal_batch(texts=("first", "second", "third"), *, rotation=0, source=1):
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
    assert (
        block_layout.internal_group_order(internal_batch(rotation=rotation), indexes).tolist()
        == expected
    )
    assert indexes.tolist() == [2, 0, 1]


def test_native_group_preserves_original_index_identity() -> None:
    indexes = numpy.array([2, 0, 1])
    assert block_layout.internal_group_order(internal_batch(source=0), indexes) is indexes


@pytest.mark.parametrize(
    ("texts", "expected"),
    [(("א", "ب", "١"), [2, 1, 0]), (("é!", "!", "漢"), [0, 1, 2]), (("א", "A", "1"), [0, 1, 2])],
)
def test_unicode_direction_votes_reverse_only_strong_rtl_majority(texts, expected) -> None:
    assert (
        block_layout.internal_group_order(internal_batch(texts), numpy.arange(3)).tolist()
        == expected
    )


def test_equal_positions_keep_input_order() -> None:
    observations = ObservationBatch.from_columns(("A", "B"), ((0, 0, 10, 10),) * 2, source=1)
    assert block_layout.internal_group_order(observations, numpy.array([1, 0])).tolist() == [1, 0]


@pytest.mark.parametrize("may_contain_ocr", [False, True])
def test_group_text_words_use_requested_ocr_order(may_contain_ocr) -> None:
    observations = ObservationBatch.from_columns(
        ("left", "right"), ((0, 0, 20, 10), (30, 0, 60, 10)), source=1
    )
    text, words = block_layout.internal_group_text_and_words(
        observations, numpy.array([1, 0]), may_contain_ocr=may_contain_ocr
    )
    assert text == ("left right" if may_contain_ocr else "right left")
    assert tuple(word.text for word in words) == (
        ("left", "right") if may_contain_ocr else ("right", "left")
    )


def test_layout_wrappers_preserve_ocr_source_labels_and_evidence() -> None:
    observations = ObservationBatch.from_columns(
        ("Hello", "world"), ((10, 10, 35, 20), (40, 10, 70, 20)), source=1, confidence=(99, 99)
    )
    lines = block_layout.internal_build_lines(observations)
    assert len(lines.lines) == 1
    assert lines.lines[0].line.source == "ocr"
    plain = block_layout.layout_blocks(observations, page_width=100, page_height=100)
    enriched, evidence = block_layout.layout_blocks_with_evidence(
        observations, page_width=100, page_height=100
    )
    assert plain == enriched
    assert len(plain) == 1
    assert plain[0].lines[0].line.text == "Hello world"
    assert evidence.line_count == 1
    assert not evidence.ambiguous
