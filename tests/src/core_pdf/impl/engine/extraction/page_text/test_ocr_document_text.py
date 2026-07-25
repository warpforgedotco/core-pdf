from pathlib import Path
from typing import Any, cast

import pytest
from core_ocr.impl import coordinator, schematic, text_analysis
from core_ocr.impl.types import OcrTextChoice

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


def test_schematic_pin_choice_confidence_recovers_tiny_labels() -> None:
    row = {
        "text": "2",
        "conf": 19,
        "choices": (OcrTextChoice("2", 96),),
    }

    assert schematic.schematic_pin_choice_confidence(row, "2") == 96


def test_dense_table_cleanup_removes_form_mark_artifacts() -> None:
    text = "APPEARANCES....... 2\nEXHIBITS....... 6\nCERTIFICATE....... 115\nLabel ~ | *"

    assert coordinator.remove_dense_table_ocr_artifact_tokens(text) == (
        "APPEARANCES 2\nEXHIBITS 6\nCERTIFICATE 115\nLabel"
    )


def test_dense_table_cleanup_preserves_dot_runs_outside_transcript_indexes() -> None:
    text = "MASS PROPERTIES\n....................\n0.001 0.002"

    assert coordinator.remove_dense_table_ocr_artifact_tokens(text) == text


def test_dense_table_cleanup_prunes_systematic_contained_fragments() -> None:
    complete = [f"entry {index} code {index}:4 value" for index in range(10)]
    fragments = [f"code {index}:4" for index in range(10)]

    cleaned = coordinator.precision_prune_redundant_dense_table_text(
        "\n".join([*complete, *fragments])
    )

    assert cleaned.splitlines() == complete


def test_dense_table_cleanup_preserves_isolated_contained_line() -> None:
    lines = [f"entry {index} code {index}:4 value" for index in range(12)]
    lines.append("code 1:4")

    assert coordinator.precision_prune_redundant_dense_table_text("\n".join(lines)) == "\n".join(
        lines
    )


def test_dense_table_cleanup_preserves_repeated_numeric_measurement_rows() -> None:
    complete = [f"station {index} 100 200 300" for index in range(10)]
    fragments = [f"100 200 {index}" for index in range(10)]

    text = "\n".join([*complete, *fragments])

    assert coordinator.precision_prune_redundant_dense_table_text(text) == text


def test_archival_letter_list_repair_recovers_sequence_markers() -> None:
    text = "\n".join(
        [
            "A letter from Alpha accepting membership.",
            "A letter from Beta accepting membership.",
            "& letter from Gamma accepting membership.",
            "4 letter from Delta accepting membership.",
        ]
    )

    repaired = coordinator.repair_repeated_archival_letter_list_markers(text)

    assert repaired.splitlines() == [
        "(a) A letter from Alpha accepting membership.",
        "(b) A letter from Beta accepting membership.",
        "(c) A letter from Gamma accepting membership.",
        "(d) A letter from Delta accepting membership.",
    ]


def test_archival_letter_list_repair_ignores_isolated_phrase() -> None:
    text = "A letter from Alpha accepting membership."

    assert coordinator.repair_repeated_archival_letter_list_markers(text) == text


def test_group_insurance_coverage_election_repair_recovers_checkboxes() -> None:
    text = "LifefAD&D Yes No Dependent Life § Yes {1 NoLTO il Yes NoSTD 2: Yes i. No"

    assert coordinator.repair_group_insurance_coverage_election_line(text) == (
        "Life/AD&D [] Yes [] No Dependent Life [] Yes [] No LTD [] Yes [] No STD [] Yes [] No"
    )


def test_chart_row_repair_joins_split_first_numeric_value() -> None:
    text = "2003 3.007 741 60,420\n2004 935 592 12,526\nHeader 3.007 741"

    assert text_analysis.repair_year_prefixed_chart_numeric_rows(text) == (
        "2003 3,007,741 60,420\n2004 935,592 12,526\nHeader 3.007 741"
    )
