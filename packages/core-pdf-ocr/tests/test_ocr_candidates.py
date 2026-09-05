# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf_ocr.impl.extract import quality as ocr_quality
from core_pdf_ocr.impl.extract.contracts import ObservationBatch, ObservationSource
from core_pdf_ocr.impl.extract.ocr import candidates as ocr_candidates


@pytest.mark.parametrize("intervening_count", [0, 25, 50])
def test_candidate_merge_finds_overlapping_duplicates_among_unrelated_observations(
    intervening_count: int,
) -> None:
    fillers = tuple(f"unrelated text {index}" for index in range(intervening_count))
    first = ObservationBatch.from_columns(
        ("duplicate label", *fillers),
        (
            (0.0, 100.0, 100.0, 200.0),
            *(
                (
                    200.0 + index * 20.0,
                    91.0 + index / max(1, intervening_count),
                    210.0 + index * 20.0,
                    101.0 + index / max(1, intervening_count),
                )
                for index in range(intervening_count)
            ),
        ),
        source=ObservationSource.OCR,
        confidence=(95.0,) * (intervening_count + 1),
    )
    second = ObservationBatch.from_columns(
        ("duplicate label",),
        ((0.0, 90.0, 100.0, 190.0),),
        source=ObservationSource.OCR,
        confidence=(95.0,),
    )

    merged = ocr_candidates.internal_merge_candidate_batches(
        (ocr_quality.internal_candidate(3, first), ocr_quality.internal_candidate(3, second))
    )

    assert merged.observations.text.count("duplicate label") == 1
    assert set(merged.observations.text) == {"duplicate label", *fillers}
    assert len(merged.observations) == intervening_count + 1
    assert first.text == ("duplicate label", *fillers)
    assert second.text == ("duplicate label",)


def test_candidate_merge_searches_replacement_geometry_for_later_duplicates() -> None:
    candidates = tuple(
        ocr_quality.internal_candidate(
            3,
            ObservationBatch.from_columns(
                (text,), (box,), source=ObservationSource.OCR, confidence=(confidence,)
            ),
        )
        for text, box, confidence in (
            ("transformation", (0.0, 100.0, 100.0, 200.0), 95.0),
            ("column transformation", (0.0, 50.0, 100.0, 150.0), 90.0),
            ("column transformation", (0.0, -10.0, 100.0, 90.0), 99.0),
        )
    )

    merged = ocr_candidates.internal_merge_candidate_batches(candidates)

    assert merged.observations.text == ("column transformation",)
    assert merged.observations.bbox.tolist() == [[0.0, -10.0, 100.0, 90.0]]


@pytest.mark.parametrize("offset", [(0.0, 20.0), (200.0, 0.0)])
def test_candidate_merge_preserves_repeated_text_in_separate_locations(
    offset: tuple[float, float],
) -> None:
    x, y = offset
    candidates = tuple(
        ocr_quality.internal_candidate(
            3,
            ObservationBatch.from_columns(
                ("repeated label",), (box,), source=ObservationSource.OCR, confidence=(95.0,)
            ),
        )
        for box in ((0.0, 0.0, 100.0, 10.0), (x, y, x + 100.0, y + 10.0))
    )

    merged = ocr_candidates.internal_merge_candidate_batches(candidates)

    assert merged.observations.text == ("repeated label", "repeated label")
