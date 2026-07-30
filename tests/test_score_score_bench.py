from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import Random
from runpy import run_path
from threading import Barrier
from types import SimpleNamespace

import pytest

score_bench = run_path(
    str(Path(__file__).parents[1] / "scripts" / "score_unstructured_bench.py"),
    run_name="score_score_bench_tests",
)
tokenize = score_bench["tokenize"]


def test_tokenize_normalizes_compatible_unicode_forms() -> None:
    assert tokenize("–12V in³ ‘quoted’") == tokenize("-12V in3 'quoted'")


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


def test_score_document_candidates_identifies_oracle_gap() -> None:
    page = SimpleNamespace(
        extraction_cache={
            "ocr_candidate_analysis": (
                {"name": "selected", "selected": True, "text": "alpha noise"},
                {"name": "oracle", "selected": False, "text": "alpha beta"},
            )
        }
    )
    document = SimpleNamespace(pages=[page])

    candidates = score_bench["score_document_candidates"](document, "alpha beta")

    assert candidates[0]["content_f1"] == pytest.approx(0.5)
    assert candidates[0]["cct"] < 1.0
    assert candidates[1]["cct"] == 1.0
    assert candidates[1]["wer"] == 0.0


def test_collect_document_extraction_analysis_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collect = score_bench["collect_document_extraction_analysis"]
    page = SimpleNamespace(
        extraction_cache={
            "ocr_requests": ({"source": "page", "elapsed_seconds": 1.25},),
            "unrelated": object(),
        }
    )
    document = SimpleNamespace(pages=[page])

    monkeypatch.delenv("CORE_PDF_EXTRACTION_ANALYSIS", raising=False)
    assert collect(document) == []

    monkeypatch.setenv("CORE_PDF_EXTRACTION_ANALYSIS", "1")
    assert collect(document) == [
        {
            "page": 1,
            "ocr_requests": ({"source": "page", "elapsed_seconds": 1.25},),
        }
    ]


def test_case_progress_uses_plain_text(capsys: pytest.CaptureFixture[str]) -> None:
    score = score_bench["CaseScore"](
        stem="example.pdf",
        status="ok",
        cct=0.75,
        percent_tokens_found=0.8,
        percent_tokens_added=0.1,
        precision=0.9,
        gt_tokens=10,
        predicted_tokens=9,
        matched_tokens=8,
        elapsed_seconds=1.25,
        content_f1=0.75,
        text_elapsed_seconds=0.5,
    )
    benchmark = score_bench["ScoreBench"]()
    benchmark.total_cases = 4

    benchmark.internal_print_progress(2, score)

    assert capsys.readouterr().out == (
        "[  2/4  ] example.pdf                                                ok          "
        "cct 0.7500  f1 0.7500  rec 0.8000  prec 0.9000   0.50s\n"
    )


def test_score_failure_bucket_identifies_actionable_modes() -> None:
    case_score = score_bench["CaseScore"]
    bucket = score_bench["score_failure_bucket"]

    assert (
        bucket(
            case_score(
                stem="empty.pdf",
                status="ok",
                cct=0.0,
                percent_tokens_found=0.0,
                percent_tokens_added=0.0,
                precision=0.0,
                gt_tokens=10,
                predicted_tokens=0,
                matched_tokens=0,
                elapsed_seconds=0.0,
            )
        )
        == "empty-output"
    )
    assert (
        bucket(
            case_score(
                stem="extra.pdf",
                status="ok",
                cct=0.0,
                percent_tokens_found=0.9,
                percent_tokens_added=0.5,
                precision=0.4,
                gt_tokens=10,
                predicted_tokens=22,
                matched_tokens=9,
                elapsed_seconds=0.0,
            )
        )
        == "low-precision"
    )
    assert (
        bucket(
            case_score(
                stem="order.pdf",
                status="ok",
                cct=0.3,
                percent_tokens_found=1.0,
                percent_tokens_added=0.0,
                precision=1.0,
                gt_tokens=10,
                predicted_tokens=10,
                matched_tokens=10,
                elapsed_seconds=0.0,
                content_f1=0.9,
                order_gap=0.6,
            )
        )
        == "order-gap"
    )


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


def test_result_table_wraps_long_review_fields(capsys: pytest.CaptureFixture[str]) -> None:
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

    score_bench["ScoreBench"].internal_print_results("Sample", [result])
    output = capsys.readouterr().out

    assert "Sample" in output
    assert "very-long-file-name-that-would-" in output
    assert "otherwise-make-the-table-hard-to-" in output
    assert "missing=alpha:3,beta:2,gamma:1" in output


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
    assert "CCT                mean  0.5000" in output
    assert "Recall             mean  0.7500" in output
    assert "Precision          mean  0.7500" in output
    assert "E2E seconds" in output
    assert "p95" in output
    assert "p99" in output
    assert "Table structure    mean  0.5000" in output


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


def test_cli_can_disable_default_html_report() -> None:
    benchmark = score_bench["ScoreBench"].from_cli(["--no-html-output"])

    assert benchmark.html_output is None


def test_cli_accepts_report_limit() -> None:
    benchmark = score_bench["ScoreBench"].from_cli(["--report-limit", "7"])

    assert benchmark.report_limit == 7


def test_cli_accepts_deterministic_partition() -> None:
    benchmark = score_bench["ScoreBench"].from_cli(["--partition", "holdout"])

    assert benchmark.partition == "holdout"
    assert score_bench["score_bench_partition"]("example.pdf") in {"train", "holdout"}


def test_score_cases_preserve_case_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_type = score_bench["ScoreBench"]
    score_case_globals = benchmark_type.internal_score_cases.__globals__
    cases = [SimpleNamespace(stem=f"case-{index}") for index in range(4)]

    def fake_score_case(case: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(stem=case.stem)

    monkeypatch.setitem(score_case_globals, "score_case", fake_score_case)
    benchmark = benchmark_type()
    benchmark.total_cases = len(cases)
    monkeypatch.setattr(benchmark, "internal_print_progress", lambda *internal_args: None)
    scores = benchmark.internal_score_cases(cases)

    assert [score.stem for score in scores] == [case.stem for case in cases]


def test_backend_is_thread_local(monkeypatch: pytest.MonkeyPatch) -> None:
    import tesserocr

    from core_pdf.impl.engine import parse as ocr

    class FakeApi:
        def __init__(self, **internal_kwargs: object) -> None:
            pass

        def SetVariable(self, *internal_args: object) -> None:
            pass

        def SetPageSegMode(self, internal_mode: int) -> None:
            pass

    ocr.internal_OCR_LOCAL.__dict__.clear()
    monkeypatch.setattr(tesserocr, "PyTessBaseAPI", FakeApi)
    barrier = Barrier(2)

    def worker() -> tuple[int, int]:
        first = ocr.internal_api(3)
        second = ocr.internal_api(3)
        barrier.wait(timeout=5)
        return (id(first), id(second))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda internal_index: worker(), range(2)))

    assert results[0][0] == results[0][1]
    assert results[1][0] == results[1][1]
    assert results[0][0] != results[1][0]
