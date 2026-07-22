from pathlib import Path
from typing import Any, cast

import pytest
from core_ocr.impl import coordinator, schematic

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


def test_dense_sparse_text_schematic_uses_tiled_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_PDF_OCR", "1")

    fixture = SCORE_BENCH_SRC / "VCAs_REV2_SCHEMATIC-p002.pdf"
    with PdfDocument.open(fixture) as document:
        page = cast(Any, document.pages[0])
        text = page.extract_text()
        diagnostics = page.extraction_cache["ocr_candidate_diagnostics"]

    assert text
    assert any(candidate["name"].endswith("_tiled") for candidate in diagnostics)


@pytest.mark.parametrize(
    ("token_type", "evidence_count", "confidence", "expected"),
    [
        ("rail", 2, 60, True),
        ("reference", 1, 90, True),
        ("rail", 1, 89, False),
        (None, 3, 95, False),
    ],
)
def test_rendered_schematic_supplement_requires_typed_evidence(
    token_type: str | None,
    evidence_count: int,
    confidence: int,
    expected: bool,
) -> None:
    entry = schematic.SchematicSupplementEntry(
        token="gnd",
        key="gnd",
        token_type=token_type,
        evidence_count=evidence_count,
        confidence=confidence,
    )

    assert schematic.rendered_schematic_addition_is_safe(entry) is expected


def test_dense_table_cleanup_removes_form_mark_artifacts() -> None:
    text = "Label ~ | *\nValue 123"

    assert coordinator.remove_dense_table_ocr_artifact_tokens(text) == "Label\nValue 123"
