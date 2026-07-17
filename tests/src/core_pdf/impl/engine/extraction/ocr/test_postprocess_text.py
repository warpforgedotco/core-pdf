import pytest

from core_pdf.impl.engine.extraction.ocr.postprocess import (
    repair_document_local_identifier_text,
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
def test_native_identifier_repair_does_not_apply_ocr_prize_noise_rules(text: str) -> None:
    assert repair_document_local_identifier_text(text, normalize_ocr_noise=False) == text


def test_ocr_identifier_repair_retains_generic_noise_normalization() -> None:
    assert (
        repair_document_local_identifier_text("alpha|beta", normalize_ocr_noise=True) == "alphabeta"
    )
