from core_ocr.impl.candidates import OcrCandidate
from core_ocr.impl.selection import (
    ocr_candidate_score,
    rendered_sparse_ocr_candidate_is_usable_without_resolution_retry,
    select_ocr_candidate,
)
from core_ocr.impl.types import OcrTextResult


def test_confident_sparse_render_can_skip_resolution_retry() -> None:
    text = " ".join(
        "Name Affiliation Moderator Conference Date Company Corporation" for _ in range(30)
    )
    candidate = OcrCandidate(
        "rendered_page_300dpi_psm11",
        OcrTextResult(text, 82),
    )

    assert rendered_sparse_ocr_candidate_is_usable_without_resolution_retry(candidate)


def test_sparse_render_with_low_confidence_keeps_resolution_retry() -> None:
    text = " ".join(
        "Name Affiliation Moderator Conference Date Company Corporation" for _ in range(30)
    )
    candidate = OcrCandidate(
        "rendered_page_300dpi_psm11",
        OcrTextResult(text, 79),
    )

    assert not rendered_sparse_ocr_candidate_is_usable_without_resolution_retry(candidate)


def test_selection_prefers_cleaner_lower_resolution_page_candidate() -> None:
    candidate_300 = OcrCandidate(
        "rendered_page_300dpi",
        OcrTextResult("A clean label " * 30, 80),
    )
    candidate_400 = OcrCandidate(
        "rendered_page_400dpi",
        OcrTextResult("A clean label " * 31 + " ! ! ! !", 82),
    )

    selected = select_ocr_candidate([candidate_400, candidate_300])

    assert selected is candidate_300


def test_sparse_two_column_split_does_not_get_full_bonus() -> None:
    split = OcrCandidate(
        "rendered_page_two_columns",
        OcrTextResult("table header 12 14 16", 26),
        region_count=2,
    )
    page = OcrCandidate(
        "rendered_page_300dpi_psm11",
        OcrTextResult("table header 12 14 16 " * 30, 58),
    )

    assert ocr_candidate_score(split) < ocr_candidate_score(page)


def test_selection_prefers_fuller_high_resolution_psm4_pass() -> None:
    low_resolution = OcrCandidate(
        "rendered_page_300dpi_psm4",
        OcrTextResult("label " * 40, 73),
    )
    high_resolution = OcrCandidate(
        "rendered_page_400dpi_psm4",
        OcrTextResult("label " * 70, 55),
    )

    assert select_ocr_candidate([low_resolution, high_resolution]) is high_resolution
