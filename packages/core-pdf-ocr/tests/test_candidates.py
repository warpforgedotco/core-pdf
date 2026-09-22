import pytest

from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf_ocr.impl.extract.ocr import candidates
from core_pdf_ocr.impl.extract.quality import Candidate, internal_candidate


def internal_batch(
    texts: tuple[str, ...],
    boxes: tuple[tuple[float, float, float, float], ...] | None = None,
    confidence: tuple[float, ...] | None = None,
) -> ObservationBatch:
    return ObservationBatch.from_columns(
        texts,
        boxes
        if boxes is not None
        else tuple((i * 30, 0, i * 30 + 20, 10) for i in range(len(texts))),
        source=1,
        confidence=confidence if confidence is not None else (90,) * len(texts),
    )


def internal_result(
    text: str,
    box: tuple[float, float, float, float] = (0, 0, 100, 10),
    *,
    confidence: float = 90,
    mode: int = 6,
    height: float = 0,
) -> Candidate:
    return internal_candidate(
        mode, internal_batch((text,), (box,), (confidence,)), median_text_height=height
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [("ＦＯＯ", "foo"), ("Straße", "strasse"), ("‘A’—“B”−", "'a'-\"b\"-"), ("ﬁ", "fi")],
)
def test_token_keys_normalize_typography_without_losing_punctuation(
    text: str, expected: str
) -> None:
    assert candidates.normalized_ocr_token_key(text) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ((), ("word",), False),
        (("abc",), ("abc", "def"), False),
        (("word",), ("prefix", "word", "suffix"), True),
        (("prefix", "word"), ("word",), True),
        (("alpha", "beta"), ("alpha", "extra", "beta"), False),
        (("alpha",), ("alphabet",), False),
    ],
)
def test_containment_requires_contiguous_whole_tokens_and_enough_text(
    left, right, expected
) -> None:
    assert candidates.candidate_text_containment(left, right) is expected


@pytest.mark.parametrize(
    ("matched", "preview_count", "spatial", "expected"),
    [
        (0, 0, 0, False),
        (23, 23, 23, False),
        (24, 24, 24, True),
        (24, 33, 24, True),
        (24, 34, 24, False),
        (40, 40, 22, True),
        (40, 40, 21, False),
    ],
)
def test_hidden_text_requires_minimum_count_token_ratio_and_spatial_ratio(
    matched: int, preview_count: int, spatial: int, expected: bool
) -> None:
    hidden = internal_batch(tuple(f"word{i}" for i in range(matched)))
    preview = internal_batch(
        tuple(f"word{i}" for i in range(preview_count)),
        tuple(
            (i * 30, 0 if i < spatial else 100, i * 30 + 20, 10 if i < spatial else 110)
            for i in range(preview_count)
        ),
    )
    assert candidates.hidden_text_verification(hidden, preview) is expected


def test_hidden_repeated_tokens_match_nearest_unused_occurrence() -> None:
    hidden = internal_batch(("word",) * 24)
    reverse = internal_batch(
        ("word",) * 24, tuple((i * 30, 0, i * 30 + 20, 10) for i in reversed(range(24)))
    )
    assert candidates.hidden_text_verification(hidden, reverse)
    one = internal_batch(("word",))
    assert not candidates.hidden_text_verification(one, reverse)


def test_empty_and_single_candidate_merges_preserve_identity() -> None:
    empty = candidates.merge_candidate_batches(())
    assert empty.mode == -1
    assert not len(empty.observations)
    result = internal_result("hello")
    assert candidates.merge_candidate_batches((result,)) is result


@pytest.mark.parametrize(
    ("first", "second", "first_conf", "second_conf", "expected"),
    [
        ("hello", "hello world", 99, 70, "hello world"),
        ("hello world", "hello", 70, 99, "hello world"),
        ("hello", "hello", 70, 99, "hello"),
        ("hello", "hello", 99, 70, "hello"),
        ("hello", "hallo", 70, 99, "hallo"),
        ("hello", "hallo", 99, 70, "hello"),
    ],
)
def test_overlapping_tiles_prefer_containing_text_then_utility(
    first: str, second: str, first_conf: float, second_conf: float, expected: str
) -> None:
    merged = candidates.merge_candidate_batches(
        (
            internal_result(first, confidence=first_conf, height=10),
            internal_result(second, confidence=second_conf, height=20),
        )
    )
    assert merged.observations.text == (expected,)
    assert merged.metrics.median_text_height == 15


def test_mode_selection_keeps_best_complete_mode_and_its_symbols() -> None:
    symbols = internal_batch(("H",))
    first = internal_candidate(6, internal_batch(("hello",)), symbols=symbols)
    second = internal_result("a", mode=11)
    merged = candidates.merge_candidate_batches((second, first))
    assert merged.mode == 6
    assert merged.observations.text == ("hello",)
    assert merged.symbols.text == ("H",)


def test_single_mode_batch_keeps_distinct_overlapping_text_and_sorts_page_order() -> None:
    observations = internal_batch(
        ("alpha", "bravo", "above", "beside"),
        ((0, 0, 20, 10), (0, 0, 20, 10), (0, 30, 20, 40), (40, 0, 60, 10)),
    )
    merged = candidates.merge_candidate_batches(
        (internal_candidate(6, observations), internal_candidate(11, ObservationBatch.empty()))
    )
    assert merged.observations.text == ("above", "alpha", "bravo", "beside")


def test_replaced_and_discarded_tiles_do_not_leave_stale_deduplication_entries() -> None:
    merged = candidates.merge_candidate_batches(
        tuple(
            internal_result(text, confidence=confidence)
            for text, confidence in (
                ("hello", 70),
                ("hello world", 80),
                ("hello", 99),
                ("hello world", 99),
            )
        )
    )
    assert merged.observations.text == ("hello world",)
    assert merged.observations.confidence.tolist() == [99]


def test_augmentation_preserves_primary_identity_when_no_usable_additions() -> None:
    primary = internal_result("primary")
    empty = internal_candidate(6, ObservationBatch.empty())
    result, added = candidates.augment_candidate(primary, empty, minimum_confidence=70)
    assert result is primary
    assert added == 0


@pytest.mark.parametrize(
    ("text", "confidence", "minimum", "box", "added"),
    [
        ("word", 69, 0, (200, 0, 300, 10), 0),
        ("word", 70, 0, (200, 0, 300, 10), 1),
        ("word", 80, 85, (200, 0, 300, 10), 0),
        ("!!!", 99, 70, (200, 0, 300, 10), 0),
        ("a", 84, 70, (200, 0, 300, 10), 0),
        ("a", 85, 70, (200, 0, 300, 10), 1),
        ("word", 99, 70, (70, 0, 170, 10), 0),
        ("word", 99, 70, (71, 0, 171, 10), 1),
    ],
)
def test_augmentation_requires_confidence_information_and_uncovered_space(
    text, confidence, minimum, box, added
) -> None:
    primary = internal_result("primary")
    result, count = candidates.augment_candidate(
        primary, internal_result(text, box, confidence=confidence), minimum_confidence=minimum
    )
    assert count == added
    assert result.observations.text == (("primary", text) if added else ("primary",))
    assert result.symbols is primary.symbols
    if not added:
        assert result is primary
