import pytest

from core_pdf.impl.engine.extraction.ocr.postprocess import (
    normalize_precision_first_prize_line_text,
    repair_document_local_identifier_text,
)


@pytest.mark.parametrize(
    "text",
    [
        "3. You fail to certify under penalties June 3, 2004, in Pub. 519.",
        "Schedule D (Form 1040) on line 14.",
        "credit from Form 2439, 4136, or 8885.",
        "Enter $0 on line 12. The $9,000",
        "Discover® Card, MasterCard® card, or loan or credit card payment.",
        "An ITIN is for tax use only.",
        "An HFD is a distribution made.",
        "Amount You Owe payment system (EFTPS).",
    ],
)
def test_native_identifier_repair_does_not_apply_ocr_prize_noise_rules(text: str) -> None:
    assert repair_document_local_identifier_text(text, normalize_ocr_noise=False) == text


def test_ocr_identifier_repair_retains_generic_noise_normalization() -> None:
    assert (
        repair_document_local_identifier_text("alpha|beta", normalize_ocr_noise=True) == "alphabeta"
    )


@pytest.mark.parametrize(
    "text",
    [
        "An ITIN is for tax use only.",
        "An HFD is a distribution made.",
        "Amount You Owe payment system (EFTPS).",
    ],
)
def test_ocr_identifier_repair_preserves_grammatical_lead_words(text: str) -> None:
    assert repair_document_local_identifier_text(text, normalize_ocr_noise=True) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ab QX-1 signal", "AbQX-1 signal"),
        ("Foo Net signal", "FooNet signal"),
    ],
)
def test_ocr_identifier_repair_still_compacts_positive_technical_patterns(
    text: str,
    expected: str,
) -> None:
    assert repair_document_local_identifier_text(text, normalize_ocr_noise=True) == expected


def test_native_identifier_repair_uses_explicit_document_support() -> None:
    assert (
        repair_document_local_identifier_text(
            "Pay with Master Card",
            support_texts=("MasterCard",),
            normalize_ocr_noise=False,
        )
        == "Pay with MasterCard"
    )


@pytest.mark.parametrize(
    "text",
    [
        "3. You fail to certify under penalties June 3, 2004, in Pub. 519.",
        "Schedule D (Form 1040) on line 14.",
        "credit from Form 2439, 4136, or 8885.",
        "Enter $0 on line 12. The $9,000",
    ],
)
def test_ocr_prize_cleanup_rejects_numbered_prose_without_multiple_rank_signals(
    text: str,
) -> None:
    assert normalize_precision_first_prize_line_text(text) == text


def test_ocr_prize_cleanup_accepts_multiple_rank_signals_and_explicit_amounts() -> None:
    assert normalize_precision_first_prize_line_text("IL $500 x 2.) $300") == ("1. $500 2.) $300")
