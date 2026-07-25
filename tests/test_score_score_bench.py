from pathlib import Path
from runpy import run_path

import pytest

score_bench = run_path(
    str(Path(__file__).parents[1] / "scripts" / "score_score_bench.py"),
    run_name="score_score_bench_tests",
)
tokenize = score_bench["tokenize"]


def test_tokenize_normalizes_compatible_unicode_forms() -> None:
    assert tokenize("–12V in³ ‘quoted’") == tokenize("-12V in3 'quoted'")


def test_ordered_errors_detect_reordered_text_hidden_by_cct() -> None:
    score_tokens = score_bench["score_tokens"]
    score_ordered_errors = score_bench["score_ordered_errors"]

    assert score_tokens("alpha beta", "beta alpha")[0] == 1.0
    cer, wer = score_ordered_errors("alpha beta", "beta alpha")

    assert cer > 0.0
    assert wer == 1.0


def test_table_scores_separate_topology_from_cell_content() -> None:
    truth = [
        {
            "type": "Table",
            "text": [
                {"x": 0, "y": 0, "w": 1, "h": 1, "content": "alpha"},
                {"x": 1, "y": 0, "w": 1, "h": 1, "content": "beta"},
            ],
        }
    ]
    predicted = [{"rows": [["alpha", "wrong"]], "spans": [[{}, {}]]}]

    structure, content = score_bench["score_tables"](truth, predicted)

    assert structure == 1.0
    assert content == 0.5


def test_score_case_reports_native_and_ocr_tracks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = score_bench["ScoreBenchCase"](
        stem="missing.pdf",
        pdf=tmp_path / "missing.pdf",
        content_gt=tmp_path / "content.txt",
        table_gt=tmp_path / "table.json",
    )
    score_case = score_bench["score_case"]

    monkeypatch.delenv("CORE_PDF_OCR", raising=False)
    assert score_case(case).track == "ocr"
    monkeypatch.setenv("CORE_PDF_OCR", "0")
    assert score_case(case).track == "native"
    monkeypatch.setenv("CORE_PDF_OCR", "1")
    assert score_case(case).track == "ocr"
