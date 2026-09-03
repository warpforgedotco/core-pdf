import json
from pathlib import Path
from random import Random
from runpy import run_path

import pytest

from core_pdf.impl.extract.ocr import tesseract as ocr_tesseract
from tests.helpers.paths import REPO_ROOT

score_bench = run_path(
    str(REPO_ROOT / "scripts" / "score_unstructured_bench.py"),
    run_name="score_score_bench_tests",
)
tokenize = score_bench["tokenize"]


def test_tokenize_normalizes_compatible_unicode_forms() -> None:
    assert tokenize("\u201312V in³ \u2018quoted\u2019") == ["-", "12v", "in3", "'", "quoted", "'"]


def test_ordered_errors_detect_reordered_text_hidden_by_cct() -> None:
    score_tokens = score_bench["score_tokens"]
    score_cct = score_bench["score_cct"]
    content_order_gap = score_bench["content_order_gap"]
    score_ordered_errors = score_bench["score_ordered_errors"]

    assert score_tokens("alpha beta", "beta alpha")[0] == 1.0
    assert score_cct("alpha beta", "beta alpha") < 1.0
    assert content_order_gap(1.0, score_cct("alpha beta", "beta alpha")) > 0.0
    assert content_order_gap(0.5, 0.75) == 0.0
    cer, wer = score_ordered_errors("alpha beta", "beta alpha")

    assert cer > 0.0
    assert wer == 1.0


def test_bit_vector_edit_distance_matches_dynamic_programming() -> None:
    edit_distance = score_bench["edit_distance"]
    random = Random(0)
    alphabet = ["a", "b", "é", "alpha", "beta"]

    def dynamic_programming(reference: list[str], predicted: list[str]) -> int:
        previous = list(range(len(predicted) + 1))
        for row, reference_item in enumerate(reference, start=1):
            current = [row]
            for column, predicted_item in enumerate(predicted, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[column] + 1,
                        previous[column - 1] + (reference_item != predicted_item),
                    )
                )
            previous = current
        return previous[-1]

    for _ in range(500):
        reference = [random.choice(alphabet) for _ in range(random.randrange(25))]
        predicted = [random.choice(alphabet) for _ in range(random.randrange(25))]
        assert edit_distance(reference, predicted) == dynamic_programming(reference, predicted)


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


def test_table_scores_match_reordered_tables_by_content() -> None:
    truth = [
        {
            "type": "Table",
            "text": [{"x": 0, "y": 0, "w": 1, "h": 1, "content": "first"}],
        },
        {
            "type": "Table",
            "text": [{"x": 0, "y": 0, "w": 1, "h": 1, "content": "second"}],
        },
    ]
    predicted = [
        {"rows": [["second"]], "spans": [[{}]]},
        {"rows": [["first"]], "spans": [[{}]]},
    ]

    structure, content = score_bench["score_tables"](truth, predicted)

    assert structure == 1.0
    assert content == 1.0


def test_table_scores_tolerate_one_cell_grid_offset() -> None:
    truth = [
        {
            "type": "Table",
            "text": [{"x": 1, "y": 1, "w": 1, "h": 1, "content": "alpha"}],
        }
    ]
    predicted = [{"rows": [["alpha"]], "spans": [[{}]]}]

    structure, content = score_bench["score_tables"](truth, predicted)

    assert structure == 1.0
    assert content == 1.0


def test_metric_report_distinguishes_text_and_table_timings() -> None:
    score = score_bench["CaseScore"](
        stem="timing.pdf",
        status="ok",
        cct=1.0,
        percent_tokens_found=1.0,
        percent_tokens_added=0.0,
        precision=1.0,
        gt_tokens=1,
        predicted_tokens=1,
        matched_tokens=1,
        elapsed_seconds=3.0,
        cer=0.0,
        wer=0.0,
        text_elapsed_seconds=2.0,
        table_elapsed_seconds=0.5,
    )

    rows = score_bench["score_metric_rows"]([score])
    values = {title: metrics for title, metrics, _suffix in rows}

    assert values["Text extraction seconds"] == [2.0]
    assert values["Table extraction seconds"] == [0.5]
    assert values["CER"] == [0.0]
    assert values["WER"] == [0.0]


def test_bootstrap_intervals_are_deterministic_and_data_dependent() -> None:
    interval = score_bench["bootstrap_interval"]([0.0, 1.0, 1.0], samples=100)

    assert interval == (0.0, 1.0)
    assert interval == score_bench["bootstrap_interval"]([0.0, 1.0, 1.0], samples=100)
    assert score_bench["bootstrap_interval"]([0.25], samples=100) == (0.25, 0.25)


def test_json_scores_record_the_scoring_schema_version(tmp_path: Path) -> None:
    score = score_bench["CaseScore"](
        stem="versioned.pdf",
        status="ok",
        cct=1.0,
        percent_tokens_found=1.0,
        percent_tokens_added=0.0,
        precision=1.0,
        gt_tokens=1,
        predicted_tokens=1,
        matched_tokens=1,
        elapsed_seconds=0.0,
    )
    output_path = tmp_path / "score.json"
    benchmark = score_bench["ScoreBench"](json_output=output_path)

    benchmark.internal_write_json([score_bench["NumberedCaseScore"](1, score)])

    assert json.loads(output_path.read_text(encoding="utf-8"))[0]["scoring_schema_version"] == "2"


BASE_CASE_SCORE = {
    "stem": "case.pdf",
    "status": "ok",
    "cct": 0.0,
    "percent_tokens_found": 0.0,
    "percent_tokens_added": 0.0,
    "precision": 0.0,
    "gt_tokens": 10,
    "predicted_tokens": 0,
    "matched_tokens": 0,
    "elapsed_seconds": 0.0,
}


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, "empty-output", id="empty-output"),
        pytest.param(
            {
                "percent_tokens_found": 0.9,
                "percent_tokens_added": 0.5,
                "precision": 0.4,
                "predicted_tokens": 22,
                "matched_tokens": 9,
            },
            "low-precision",
            id="low-precision",
        ),
        pytest.param(
            {
                "cct": 0.3,
                "percent_tokens_found": 1.0,
                "precision": 1.0,
                "predicted_tokens": 10,
                "matched_tokens": 10,
                "content_f1": 0.9,
                "order_gap": 0.6,
            },
            "order-gap",
            id="order-gap",
        ),
    ],
)
def test_score_failure_bucket_identifies_actionable_modes(
    overrides: dict[str, object], expected: str
) -> None:
    case_score = score_bench["CaseScore"]
    bucket = score_bench["score_failure_bucket"]

    assert bucket(case_score(**{**BASE_CASE_SCORE, **overrides})) == expected


def test_score_hints_formats_missing_and_extra_tokens() -> None:
    score = score_bench["CaseScore"](
        stem="example.pdf",
        status="ok",
        cct=0.5,
        percent_tokens_found=0.5,
        percent_tokens_added=0.5,
        precision=0.5,
        gt_tokens=4,
        predicted_tokens=4,
        matched_tokens=2,
        elapsed_seconds=0.0,
        missing_top=[("alpha", 3), ("beta", 2), ("gamma", 1), ("delta", 1)],
        extra_top=[("noise", 4)],
    )

    assert score_bench["score_hints"](score) == "missing=alpha:3,beta:2,gamma:1 extra=noise:4"


def test_result_review_fields_survive_long_stems() -> None:
    score = score_bench["CaseScore"](
        stem="very-long-file-name-that-would-otherwise-make-the-table-hard-to-review.pdf",
        status="ok",
        cct=0.5,
        percent_tokens_found=0.5,
        percent_tokens_added=0.5,
        precision=0.5,
        gt_tokens=4,
        predicted_tokens=4,
        matched_tokens=2,
        elapsed_seconds=0.0,
        content_f1=0.5,
        text_elapsed_seconds=1.25,
        missing_top=[("alpha", 3), ("beta", 2), ("gamma", 1)],
        extra_top=[("verylongtoken", 4), ("noise", 1)],
    )
    result = score_bench["NumberedCaseScore"](3, score)
    benchmark = score_bench["ScoreBench"]()
    benchmark.total_cases = 1
    benchmark.started_at = 0.0

    html_text = benchmark.internal_render_html([result], 1.0)

    # A long stem and both hint columns must survive into the rendered report.
    assert "very-long-file-name-that-would-otherwise-make-the-table-hard-to-review.pdf" in html_text
    assert "missing=alpha:3,beta:2,gamma:1" in html_text
    assert "extra=verylongtoken:4,noise:1" in html_text


def sample_numbered_scores() -> list[object]:
    case_score = score_bench["CaseScore"]
    numbered_score = score_bench["NumberedCaseScore"]
    return [
        numbered_score(
            1,
            case_score(
                stem="alpha.pdf",
                status="ok",
                cct=0.25,
                percent_tokens_found=0.5,
                percent_tokens_added=0.0,
                precision=1.0,
                gt_tokens=10,
                predicted_tokens=5,
                matched_tokens=5,
                elapsed_seconds=1.0,
                content_f1=0.6667,
                order_gap=0.4167,
                text_elapsed_seconds=0.5,
                table_structure_f1=0.25,
                table_content_f1=0.5,
            ),
        ),
        numbered_score(
            2,
            case_score(
                stem="beta.pdf",
                status="ok",
                cct=0.75,
                percent_tokens_found=1.0,
                percent_tokens_added=0.5,
                precision=0.5,
                gt_tokens=10,
                predicted_tokens=20,
                matched_tokens=10,
                elapsed_seconds=2.0,
                content_f1=0.6667,
                order_gap=0.0,
                text_elapsed_seconds=1.5,
                table_structure_f1=0.75,
                table_content_f1=1.0,
            ),
        ),
    ]


def test_report_summarizes_score_distributions(capsys: pytest.CaptureFixture[str]) -> None:
    benchmark = score_bench["ScoreBench"](report_limit=1)
    benchmark.total_cases = 2
    benchmark.started_at = 0.0
    results = sample_numbered_scores()

    benchmark.internal_print_report(results)
    output = capsys.readouterr().out

    assert "Metrics" in output
    assert "| CCT | 0.5000 | 0.5000 | 0.2500 | 0.7500 |" in output
    assert "| Recall | 0.7500 | 0.7500 | 0.5000 | 1.0000 |" in output
    assert "| Precision | 0.7500 | 0.7500 | 0.5000 | 1.0000 |" in output
    assert "E2E seconds" in output
    assert "| Table structure | 0.5000 | 0.5000 | 0.2500 | 0.7500 |" in output
    assert "Bucket Breakdown" in output


def test_html_report_renders_score_summary(tmp_path: Path) -> None:
    output_path = tmp_path / "score.html"
    benchmark = score_bench["ScoreBench"](html_output=output_path, report_limit=1)
    benchmark.total_cases = 2
    benchmark.started_at = 0.0

    benchmark.internal_write_html(sample_numbered_scores())
    html_text = output_path.read_text(encoding="utf-8")

    assert "<title>SCORE-Bench Report</title>" in html_text
    assert "Score Summary" in html_text
    assert '<dl class="metric-stats">' in html_text
    assert "Precision" in html_text
    assert "E2E seconds" in html_text
    assert "<dt>p95</dt>" in html_text
    assert "<dt>p99</dt>" in html_text
    assert "Low Precision / Extra Output" in html_text
    assert "alpha.pdf" in html_text


def test_cli_has_no_extraction_track_flags() -> None:
    benchmark = score_bench["ScoreBench"].from_cli([])

    assert benchmark.html_output == score_bench["DEFAULT_HTML_OUTPUT"]
    with pytest.raises(SystemExit):
        score_bench["ScoreBench"].from_cli(["--ocr"])
    with pytest.raises(SystemExit):
        score_bench["ScoreBench"].from_cli(["--native"])
    with pytest.raises(SystemExit):
        score_bench["ScoreBench"].from_cli(["--candidate-analysis"])
    with pytest.raises(SystemExit):
        score_bench["ScoreBench"].from_cli(["--extraction-analysis"])


def test_cli_can_disable_default_html_report() -> None:
    benchmark = score_bench["ScoreBench"].from_cli(["--no-html-output"])

    assert benchmark.html_output is None


def test_backend_is_created_for_each_session(monkeypatch: pytest.MonkeyPatch) -> None:
    import tesserocr

    class FakeApi:
        def __init__(self, **internal_kwargs: object) -> None:
            pass

        def SetVariable(self, *internal_args: object) -> None:
            pass

        def SetPageSegMode(self, internal_mode: int) -> None:
            pass

    monkeypatch.setattr(tesserocr, "PyTessBaseAPI", FakeApi)
    # Resolving real tessdata is not part of this lifecycle test.
    monkeypatch.setattr(ocr_tesseract, "internal_tessdata_path", lambda: "")
    first = ocr_tesseract.internal_api(3)
    second = ocr_tesseract.internal_api(3)

    assert first is not second
