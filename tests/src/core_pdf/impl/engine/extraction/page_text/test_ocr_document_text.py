from pathlib import Path
from typing import Any, cast

import pytest

from core_pdf import PdfDocument

SCORE_BENCH_SRC = Path(__file__).parents[7] / "tests" / "fixtures" / "SCORE-Bench" / "src"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "2023-gmi-lab-call_p4-15-p003.pdf",
            "Topic Area 1",
        ),
        (
            "EPA_AirQualityLetter_Table-p001.pdf",
            "UNITED STATES ENVIRONMENTAL PROTECTION AGENCY",
        ),
    ],
)
def test_document_extract_preserves_ocr_text_without_resolved_geometry(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    expected: str,
) -> None:
    monkeypatch.setenv("CORE_PDF_OCR", "1")

    with PdfDocument.open(SCORE_BENCH_SRC / filename) as document:
        page = cast(Any, document.pages[0])
        page_text = page.extract_text()
        resolved_lines = page.extract_resolved_lines()
        document_text = document.extract().text

    assert expected in page_text
    assert expected in document_text
    assert resolved_lines
