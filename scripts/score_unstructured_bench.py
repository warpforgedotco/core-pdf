from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import textwrap
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any, Iterable, cast

from core_pdf import PdfDocument
from core_pdf.impl.extract.contracts import ParseReport

ROOT = Path(__file__).resolve().parents[1]
SCORE_BENCH_ROOT = ROOT / "tests" / "fixtures" / "SCORE-Bench"
CORRECT_TRUTH_SUFFIX = "__correct_truth"
DEFAULT_HTML_OUTPUT = ROOT / "parsebench-output" / "scorebench-report.html"
PARTITION_SALT = "core-pdf-precision-v1\0"
SCORING_SCHEMA_VERSION = "2"


@dataclass(frozen=True)
class ScoreBenchCase:
    stem: str
    pdf: Path
    content_gt: Path
    table_gt: Path


@dataclass(frozen=True)
class CaseScore:
    stem: str
    status: str
    cct: float
    percent_tokens_found: float
    percent_tokens_added: float
    precision: float
    gt_tokens: int
    predicted_tokens: int
    matched_tokens: int
    elapsed_seconds: float
    scoring_schema_version: str = SCORING_SCHEMA_VERSION
    content_f1: float = 0.0
    order_gap: float = 0.0
    cer: float | None = None
    wer: float | None = None
    table_structure_f1: float | None = None
    table_content_f1: float | None = None
    table_expected: int = 0
    table_predicted: int = 0
    table_matched: int = 0
    open_elapsed_seconds: float = 0.0
    text_elapsed_seconds: float = 0.0
    table_elapsed_seconds: float = 0.0
    evaluation_elapsed_seconds: float = 0.0
    error: str | None = None
    missing_top: list[tuple[str, int]] | None = None
    extra_top: list[tuple[str, int]] | None = None
    candidate_analysis: list[dict[str, Any]] | None = None
    selected_candidate_cct: float | None = None
    best_candidate_name: str | None = None
    best_candidate_cct: float | None = None
    candidate_oracle_gap: float | None = None
    selected_to_final_cct: float | None = None
    best_to_final_cct: float | None = None
    extraction_analysis: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class NumberedCaseScore:
    case_number: int
    score: CaseScore


def score_failure_bucket(score: CaseScore) -> str:
    if score.status != "ok":
        return score.status
    if score.gt_tokens > 0 and score.predicted_tokens == 0:
        return "empty-output"
    if score.percent_tokens_found < 0.2:
        return "missing-most"
    if score.percent_tokens_found < 0.6 and score.precision >= 0.8:
        return "low-recall"
    if score.precision < 0.6 and score.percent_tokens_found >= 0.8:
        return "low-precision"
    if score.table_structure_f1 is not None and score.table_structure_f1 < 0.5:
        return "table-structure"
    if score.content_f1 >= 0.8 and score.order_gap >= 0.45:
        return "order-gap"
    if score.content_f1 < 0.5:
        return "low-f1"
    return "mixed"


def format_token_counts(items: list[tuple[str, int]] | None, limit: int = 3) -> str:
    if not items:
        return "-"
    return ",".join(f"{token}:{count}" for token, count in items[:limit])


def score_hints(score: CaseScore) -> str:
    parts = []
    if score.missing_top:
        parts.append(f"missing={format_token_counts(score.missing_top)}")
    if score.extra_top:
        parts.append(f"extra={format_token_counts(score.extra_top)}")
    return " ".join(parts) if parts else "-"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_interval(
    values: list[float], *, samples: int = 2_000, seed: int = 0
) -> tuple[float, float]:
    """Return a deterministic percentile-bootstrap 95% confidence interval."""
    if len(values) < 2:
        value = values[0] if values else 0.0
        return value, value
    random = Random(seed)
    means = [sum(random.choice(values) for _ in values) / len(values) for _ in range(samples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def shorten_middle(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    prefix_width = (width - 3) // 2
    suffix_width = width - 3 - prefix_width
    return f"{value[:prefix_width]}...{value[-suffix_width:]}"


def wrap_cell(value: str, width: int) -> list[str]:
    if len(value) <= width:
        return [value]
    wrapped = textwrap.wrap(
        value,
        width=width,
        break_long_words=True,
        break_on_hyphens=True,
    )
    if wrapped:
        return wrapped
    return [value[:width]]


def format_table_cell(value: str, width: int, align: str) -> str:
    if len(value) > width:
        value = shorten_middle(value, width)
    if align == "right":
        return value.rjust(width)
    return value.ljust(width)


def print_metric_summary(title: str, values: list[float], suffix: str = "") -> None:
    if not values:
        return
    print(
        f"  {title:<18} "
        f"mean {mean(values):>7.4f}{suffix}  "
        f"p10 {percentile(values, 0.1):>7.4f}{suffix}  "
        f"p50 {percentile(values, 0.5):>7.4f}{suffix}  "
        f"p90 {percentile(values, 0.9):>7.4f}{suffix}  "
        f"p95 {percentile(values, 0.95):>7.4f}{suffix}  "
        f"p99 {percentile(values, 0.99):>7.4f}{suffix}  "
        f"min {min(values):>7.4f}{suffix}  "
        f"max {max(values):>7.4f}{suffix}"
    )


def metric_value(value: float, suffix: str = "") -> str:
    return f"{value:.4f}{suffix}"


def render_meta_card(label: str, value: str) -> str:
    return f"""
    <div class="meta-card">
      <span>{html.escape(label)}</span>
      <strong>{html.escape(value)}</strong>
    </div>
    """


def render_metric_card(title: str, values: list[float], suffix: str = "") -> str:
    low, high = bootstrap_interval(values)
    return f"""
    <div class="metric-card">
      <span>{html.escape(title)}</span>
      <strong>{metric_value(mean(values), suffix)}</strong>
      <dl class="metric-stats">
        <div><dt>p10</dt><dd>{metric_value(percentile(values, 0.1), suffix)}</dd></div>
        <div><dt>p50</dt><dd>{metric_value(percentile(values, 0.5), suffix)}</dd></div>
        <div><dt>p90</dt><dd>{metric_value(percentile(values, 0.9), suffix)}</dd></div>
        <div><dt>p95</dt><dd>{metric_value(percentile(values, 0.95), suffix)}</dd></div>
        <div><dt>p99</dt><dd>{metric_value(percentile(values, 0.99), suffix)}</dd></div>
        <div><dt>95% CI</dt><dd>{metric_value(low, suffix)}–{metric_value(high, suffix)}</dd></div>
        <div><dt>min</dt><dd>{metric_value(min(values), suffix)}</dd></div>
        <div><dt>max</dt><dd>{metric_value(max(values), suffix)}</dd></div>
      </dl>
    </div>
    """


def render_html_section(title: str, results: list[NumberedCaseScore]) -> str:
    if not results:
        return ""
    rows = "\n".join(render_html_result_row(rank, result) for rank, result in enumerate(results, 1))
    return f"""
    <section>
      <h2>{html.escape(title)}</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Case</th>
              <th class="bucket">Bucket</th>
              <th>CCT</th>
              <th>F1</th>
              <th>Gap</th>
              <th>Recall</th>
              <th>Precision</th>
              <th>Time</th>
              <th class="stem">Stem</th>
              <th class="hints">Hints</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
    </section>
    """


def render_html_result_row(rank: int, result: NumberedCaseScore) -> str:
    score = result.score
    bucket = score_failure_bucket(score)
    return f"""
    <tr>
      <td>{rank}</td>
      <td>{result.case_number}</td>
      <td class="bucket">
        <span class="bucket-pill {html.escape(bucket)}">{html.escape(bucket)}</span>
      </td>
      <td>{score.cct:.4f}</td>
      <td>{score.content_f1:.4f}</td>
      <td>{score.order_gap:.4f}</td>
      <td>{score.percent_tokens_found:.4f}</td>
      <td>{score.precision:.4f}</td>
      <td>{score.text_elapsed_seconds:.2f}s</td>
      <td class="stem">{html.escape(score.stem)}</td>
      <td class="hints"><code>{html.escape(score_hints(score))}</code></td>
    </tr>
    """


def score_metric_rows(scores: list[CaseScore]) -> list[tuple[str, list[float], str]]:
    table_structure = [
        score.table_structure_f1 for score in scores if score.table_structure_f1 is not None
    ]
    table_content = [
        score.table_content_f1 for score in scores if score.table_content_f1 is not None
    ]
    return [
        ("CCT", [score.cct for score in scores], ""),
        ("Content F1", [score.content_f1 for score in scores], ""),
        ("Recall", [score.percent_tokens_found for score in scores], ""),
        ("Precision", [score.precision for score in scores], ""),
        ("Order gap (F1-CCT)", [score.order_gap for score in scores], ""),
        ("CER", [score.cer for score in scores if score.cer is not None], ""),
        ("WER", [score.wer for score in scores if score.wer is not None], ""),
        ("E2E seconds", [score.elapsed_seconds for score in scores], "s"),
        ("Text extraction seconds", [score.text_elapsed_seconds for score in scores], "s"),
        ("Table extraction seconds", [score.table_elapsed_seconds for score in scores], "s"),
        ("Table structure", table_structure, ""),
        ("Table content", table_content, ""),
    ]


def score_result_sections(
    results: list[NumberedCaseScore],
) -> list[tuple[str, list[NumberedCaseScore]]]:
    successful = [result for result in results if result.score.status == "ok"]
    sections = [
        ("Results by CCT", sorted(results, key=lambda item: item.score.cct)),
    ]
    if not successful:
        return sections
    sections.extend(
        [
            (
                "Low Precision / Extra Output",
                sorted(
                    successful,
                    key=lambda item: (
                        item.score.precision,
                        -item.score.percent_tokens_found,
                        item.case_number,
                    ),
                ),
            ),
            (
                "Low Recall / Missing Output",
                sorted(
                    successful,
                    key=lambda item: (
                        item.score.percent_tokens_found,
                        -item.score.precision,
                        item.case_number,
                    ),
                ),
            ),
            (
                "Largest Order Gap",
                sorted(
                    successful,
                    key=lambda item: (-item.score.order_gap, item.case_number),
                ),
            ),
            (
                "Slowest Extraction",
                sorted(
                    successful,
                    key=lambda item: (-item.score.text_elapsed_seconds, item.case_number),
                ),
            ),
        ]
    )
    table_results = [
        result
        for result in successful
        if result.score.table_structure_f1 is not None or result.score.table_content_f1 is not None
    ]
    if table_results:
        sections.append(
            (
                "Weakest Table Scores",
                sorted(
                    table_results,
                    key=lambda item: (
                        item.score.table_structure_f1
                        if item.score.table_structure_f1 is not None
                        else 1.0,
                        item.score.table_content_f1
                        if item.score.table_content_f1 is not None
                        else 1.0,
                        item.case_number,
                    ),
                ),
            )
        )
    candidate_results = [
        result for result in successful if result.score.candidate_oracle_gap is not None
    ]
    if candidate_results:
        sections.append(
            (
                "Largest Candidate Oracle Gap",
                sorted(
                    candidate_results,
                    key=lambda item: (
                        -(item.score.candidate_oracle_gap or 0.0),
                        item.case_number,
                    ),
                ),
            )
        )
    return sections


def iter_score_bench_cases(root: Path = SCORE_BENCH_ROOT) -> list[ScoreBenchCase]:
    content_dir = root / "content-gt"
    table_dir = root / "table-gt"
    src_dir = root / "src"
    if not content_dir.exists() or not table_dir.exists() or not src_dir.exists():
        return []

    content_by_stem = ground_truth_by_stem(content_dir, "*.txt")
    correct_truth_root = root.with_name(f"{root.name}-correct-truth")
    if correct_truth_root.exists():
        content_by_stem.update(
            {
                stem: path
                for stem, path in ground_truth_by_stem(correct_truth_root, "*.txt").items()
                if is_correct_truth_path(path)
            }
        )
    table_by_stem = ground_truth_by_stem(table_dir, "*.json")
    return [
        ScoreBenchCase(stem, src_dir / stem, content_by_stem[stem], table_by_stem[stem])
        for stem in sorted(content_by_stem.keys() & table_by_stem.keys())
    ]


def score_bench_partition(stem: str) -> str:
    digest = hashlib.sha256(f"{PARTITION_SALT}{stem}".encode()).digest()
    return "holdout" if int.from_bytes(digest[:8], "big") % 5 == 0 else "train"


def source_pdf_stem(name: str) -> str:
    marker = ".pdf__"
    return name.split(marker, 1)[0] + ".pdf" if marker in name else name


def ground_truth_by_stem(directory: Path, pattern: str) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in sorted(directory.glob(pattern)):
        stem = source_pdf_stem(path.name)
        if stem not in selected or is_correct_truth_path(path):
            selected[stem] = path
    return selected


def is_correct_truth_path(path: Path) -> bool:
    return path.stem.endswith(CORRECT_TRUTH_SUFFIX)


def score_case(case: ScoreBenchCase) -> CaseScore:
    started = perf_counter()
    if not is_materialized_pdf(case.pdf):
        return CaseScore(
            stem=case.stem,
            status="missing_pdf",
            cct=0.0,
            percent_tokens_found=0.0,
            percent_tokens_added=0.0,
            precision=0.0,
            gt_tokens=0,
            predicted_tokens=0,
            matched_tokens=0,
            elapsed_seconds=perf_counter() - started,
            error=f"{case.pdf} is not a materialized PDF. Run git lfs pull in SCORE-Bench.",
        )

    gt_text = case.content_gt.read_text(encoding="utf-8")
    try:
        open_started = perf_counter()
        document = PdfDocument(case.pdf)
        open_elapsed = perf_counter() - open_started
        with document:
            text_started = perf_counter()
            predicted_text = extract_document_text(document)
            text_elapsed = perf_counter() - text_started
            table_started = perf_counter()
            predicted_tables = document.extract().table_view.tables
            table_elapsed = perf_counter() - table_started
            evaluation_started = perf_counter()
            candidate_analysis = score_document_candidates(document, gt_text)
            extraction_analysis = collect_document_extraction_analysis(document)
        content_f1, found, added, precision, gt_count, predicted_count, matched = score_tokens(
            gt_text, predicted_text
        )
        cct = score_cct(gt_text, predicted_text)
        token_diff = score_token_diff_summary(gt_text, predicted_text)
        cer, wer = score_ordered_errors(gt_text, predicted_text)
        table_truth = json.loads(case.table_gt.read_text(encoding="utf-8"))
        table_structure_f1, table_content_f1 = score_tables(table_truth, predicted_tables)
        truth_cells = ground_truth_table_cells(table_truth)
        predicted_cells = predicted_table_cells(predicted_tables)
        table_pairs = match_table_indexes(truth_cells, predicted_cells)
        selected_candidate = next(
            (candidate for candidate in candidate_analysis if candidate["selected"]),
            None,
        )
        best_candidate = max(
            candidate_analysis,
            key=lambda candidate: (candidate["cct"], candidate["precision"]),
            default=None,
        )
        selected_candidate_cct = (
            cast(float, selected_candidate["cct"]) if selected_candidate is not None else None
        )
        best_candidate_cct = (
            cast(float, best_candidate["cct"]) if best_candidate is not None else None
        )
        evaluation_elapsed = perf_counter() - evaluation_started
        score = CaseScore(
            stem=case.stem,
            status="ok",
            cct=cct,
            percent_tokens_found=found,
            percent_tokens_added=added,
            precision=precision,
            gt_tokens=gt_count,
            predicted_tokens=predicted_count,
            matched_tokens=matched,
            elapsed_seconds=perf_counter() - started,
            content_f1=content_f1,
            order_gap=content_order_gap(content_f1, cct),
            cer=cer,
            wer=wer,
            table_structure_f1=table_structure_f1,
            table_content_f1=table_content_f1,
            table_expected=len(group_table_cells(truth_cells)),
            table_predicted=len(group_table_cells(predicted_cells)),
            table_matched=len(table_pairs),
            open_elapsed_seconds=open_elapsed,
            text_elapsed_seconds=text_elapsed,
            table_elapsed_seconds=table_elapsed,
            evaluation_elapsed_seconds=evaluation_elapsed,
            missing_top=token_diff["missing_top"][:5],
            extra_top=token_diff["extra_top"][:5],
            candidate_analysis=candidate_analysis or None,
            selected_candidate_cct=selected_candidate_cct,
            best_candidate_name=(
                cast(str, best_candidate["name"]) if best_candidate is not None else None
            ),
            best_candidate_cct=best_candidate_cct,
            candidate_oracle_gap=(
                best_candidate_cct - selected_candidate_cct
                if best_candidate_cct is not None and selected_candidate_cct is not None
                else None
            ),
            selected_to_final_cct=(
                cct - selected_candidate_cct if selected_candidate_cct is not None else None
            ),
            best_to_final_cct=(
                best_candidate_cct - cct if best_candidate_cct is not None else None
            ),
            extraction_analysis=extraction_analysis or None,
        )
        return score
    except Exception as exc:
        return CaseScore(
            stem=case.stem,
            status="error",
            cct=0.0,
            percent_tokens_found=0.0,
            percent_tokens_added=0.0,
            precision=0.0,
            gt_tokens=len(tokenize(clean_score_bench_text(gt_text))),
            predicted_tokens=0,
            matched_tokens=0,
            elapsed_seconds=perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def extract_document_text(document: PdfDocument) -> str:
    """Return the canonical core-document text used for benchmark scoring."""
    return document.extract().text


def score_document_candidates(document: PdfDocument, gt_text: str) -> list[dict[str, Any]]:
    """Score opt-in raw OCR candidates against the case's content ground truth."""
    scored: list[dict[str, Any]] = []
    for page_number, page in enumerate(document.pages, start=1):
        report = getattr(page, "parse_report", None)
        records = report.recognition.candidate_analysis if isinstance(report, ParseReport) else ()
        for record in records:
            if not isinstance(record, dict):
                continue
            text = record.get("text")
            if not isinstance(text, str):
                continue
            content_f1, found, added, precision, gt_count, predicted_count, matched = score_tokens(
                gt_text, text
            )
            cct = score_cct(gt_text, text)
            cer, wer = score_ordered_errors(gt_text, text)
            scored.append(
                {key: value for key, value in record.items() if key != "text"}
                | {
                    "page": page_number,
                    "cct": cct,
                    "content_f1": content_f1,
                    "order_gap": content_order_gap(content_f1, cct),
                    "recall": found,
                    "added": added,
                    "precision": precision,
                    "gt_tokens": gt_count,
                    "predicted_tokens": predicted_count,
                    "matched_tokens": matched,
                    "cer": cer,
                    "wer": wer,
                }
            )
    return scored


def collect_document_extraction_analysis(document: PdfDocument) -> list[dict[str, Any]]:
    """Collect opt-in extraction diagnostics in a JSON-friendly page envelope."""
    if os.environ.get("CORE_PDF_EXTRACTION_ANALYSIS", "").casefold() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return []
    records: list[dict[str, Any]] = []
    for page_number, page in enumerate(document.pages, start=1):
        report = getattr(page, "parse_report", None)
        if isinstance(report, ParseReport):
            records.append({"page": page_number, **report.as_record()})
    return records


def configure_native_thread_budget() -> None:
    """Bound native worker pools so case threads do not oversubscribe the CPU."""
    for variable in (
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"


def score_token_diff_summary(gt_text: str, predicted_text: str) -> dict[str, Any]:
    gt_counter = Counter(tokenize(clean_score_bench_text(gt_text)))
    predicted_counter = Counter(tokenize(predicted_text))
    matched_counter = gt_counter & predicted_counter
    missing = gt_counter - predicted_counter
    extra = predicted_counter - gt_counter
    return {
        "gt_tokens": sum(gt_counter.values()),
        "predicted_tokens": sum(predicted_counter.values()),
        "matched_tokens": sum(matched_counter.values()),
        "missing_top": missing.most_common(100),
        "extra_top": extra.most_common(100),
    }


def is_materialized_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 5:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def score_tokens(
    gt_text: str, predicted_text: str
) -> tuple[float, float, float, float, int, int, int]:
    gt_tokens = tokenize(clean_score_bench_text(gt_text))
    predicted_tokens = tokenize(predicted_text)
    gt_counter = Counter(gt_tokens)
    predicted_counter = Counter(predicted_tokens)
    matched = sum((gt_counter & predicted_counter).values())
    gt_count = len(gt_tokens)
    predicted_count = len(predicted_tokens)
    return (
        (2 * matched / (gt_count + predicted_count) if gt_count + predicted_count else 1.0),
        matched / gt_count if gt_count else 1.0,
        (max(predicted_count - matched, 0) / predicted_count if predicted_count else 0.0),
        (matched / predicted_count if predicted_count else (1.0 if gt_count == 0 else 0.0)),
        gt_count,
        predicted_count,
        matched,
    )


def edit_distance(reference: list[str], predicted: list[str]) -> int:
    """Return exact Levenshtein distance with Myers' bit-vector algorithm."""
    if reference == predicted:
        return 0
    if not reference:
        return len(predicted)
    if not predicted:
        return len(reference)
    if len(reference) > len(predicted):
        reference, predicted = predicted, reference

    pattern_length = len(reference)
    pattern_mask = (1 << pattern_length) - 1
    high_bit = 1 << (pattern_length - 1)
    equality_masks: dict[str, int] = {}
    for index, item in enumerate(reference):
        equality_masks[item] = equality_masks.get(item, 0) | (1 << index)

    positive = pattern_mask
    negative = 0
    distance = pattern_length
    for item in predicted:
        equality = equality_masks.get(item, 0)
        vertical = equality | negative
        horizontal = (((equality & positive) + positive) ^ positive) | equality
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        if positive_horizontal & high_bit:
            distance += 1
        elif negative_horizontal & high_bit:
            distance -= 1
        positive_horizontal = ((positive_horizontal << 1) | 1) & pattern_mask
        negative_horizontal = (negative_horizontal << 1) & pattern_mask
        positive = (negative_horizontal | ~(vertical | positive_horizontal)) & pattern_mask
        negative = positive_horizontal & vertical
    return distance


def score_ordered_errors(gt_text: str, predicted_text: str) -> tuple[float, float]:
    cleaned_gt = clean_score_bench_text(gt_text)
    gt_characters = list(normalize_score_text(cleaned_gt))
    predicted_characters = list(normalize_score_text(predicted_text))
    gt_tokens = tokenize(cleaned_gt)
    predicted_tokens = tokenize(predicted_text)
    cer = edit_distance(gt_characters, predicted_characters) / max(1, len(gt_characters))
    wer = edit_distance(gt_tokens, predicted_tokens) / max(1, len(gt_tokens))
    return cer, wer


def score_cct(gt_text: str, predicted_text: str) -> float:
    """Return character-level normalized edit similarity for extracted content."""
    reference = list(normalize_score_text(clean_score_bench_text(gt_text)))
    predicted = list(normalize_score_text(predicted_text))
    return max(0.0, 1.0 - edit_distance(reference, predicted) / max(1, len(reference)))


def content_order_gap(content_f1: float, cct: float) -> float:
    """Estimate the quality lost to sequence differences after token matching."""
    return max(0.0, content_f1 - cct)


def normalize_score_text(text: str) -> str:
    """Normalize characters while retaining whitespace and reading order for CER."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def score_tables(
    ground_truth: object, predicted_records: object
) -> tuple[float | None, float | None]:
    """Score table cell topology and coordinate-aware cell content independently."""
    truth_cells = ground_truth_table_cells(ground_truth)
    predicted_cells = predicted_table_cells(predicted_records)
    if not truth_cells and not predicted_cells:
        return None, None

    table_pairs = match_table_indexes(truth_cells, predicted_cells)
    truth_to_match = {truth: match for match, (truth, _predicted) in enumerate(table_pairs)}
    predicted_to_match = {predicted: match for match, (_truth, predicted) in enumerate(table_pairs)}

    def remap(
        cells: list[tuple[int, int, int, int, int, str]], mapping: dict[int, int], offset: int
    ) -> list[tuple[int, int, int, int, int, str]]:
        return [
            (mapping.get(table, offset + table), x, y, width, height, content)
            for table, x, y, width, height, content in cells
        ]

    matched_count = len(table_pairs)
    truth_cells = remap(truth_cells, truth_to_match, matched_count)
    predicted_cells = remap(
        predicted_cells,
        predicted_to_match,
        matched_count + len(group_table_cells(truth_cells)),
    )

    truth_tables = group_table_cells(truth_cells)
    predicted_tables = group_table_cells(predicted_cells)
    structure_total = len(truth_cells) + len(predicted_cells)
    structure_matches = 0
    truth_content_count = 0
    predicted_content_count = 0
    content_matches = 0
    for table in set(truth_tables) | set(predicted_tables):
        truth = truth_tables.get(table, [])
        predicted = predicted_tables.get(table, [])
        matches = match_table_cells(truth, predicted)
        structure_matches += len(matches)
        truth_content_count += sum(len(tokenize(cell[5])) for cell in truth)
        predicted_content_count += sum(len(tokenize(cell[5])) for cell in predicted)
        content_matches += sum(
            sum((Counter(tokenize(truth_cell[5])) & Counter(tokenize(predicted_cell[5]))).values())
            for truth_cell, predicted_cell in matches
        )
    structure_f1 = 2 * structure_matches / structure_total if structure_total else 1.0
    content_total = truth_content_count + predicted_content_count
    content_f1 = 2 * content_matches / content_total if content_total else 1.0
    return structure_f1, content_f1


def match_table_indexes(
    truth_cells: list[tuple[int, int, int, int, int, str]],
    predicted_cells: list[tuple[int, int, int, int, int, str]],
) -> tuple[tuple[int, int], ...]:
    """Match tables by content overlap so list ordering is not table identity."""
    truth_tables = group_table_cells(truth_cells)
    predicted_tables = group_table_cells(predicted_cells)
    candidates: list[tuple[float, int, int]] = []
    for truth_index, truth in truth_tables.items():
        truth_tokens = Counter(token for cell in truth for token in tokenize(cell[5]))
        for predicted_index, predicted in predicted_tables.items():
            predicted_tokens = Counter(token for cell in predicted for token in tokenize(cell[5]))
            overlap = sum((truth_tokens & predicted_tokens).values())
            denominator = max(1, sum(truth_tokens.values()) + sum(predicted_tokens.values()))
            if overlap:
                candidates.append((2 * overlap / denominator, truth_index, predicted_index))
    pairs: list[tuple[int, int]] = []
    used_truth: set[int] = set()
    used_predicted: set[int] = set()
    for _score, truth_index, predicted_index in sorted(candidates, reverse=True):
        if truth_index in used_truth or predicted_index in used_predicted:
            continue
        pairs.append((truth_index, predicted_index))
        used_truth.add(truth_index)
        used_predicted.add(predicted_index)
    return tuple(pairs)


def group_table_cells(
    cells: list[tuple[int, int, int, int, int, str]],
) -> dict[int, list[tuple[int, int, int, int, int, str]]]:
    grouped: dict[int, list[tuple[int, int, int, int, int, str]]] = {}
    for cell in cells:
        grouped.setdefault(cell[0], []).append(cell)
    return grouped


def match_table_cells(
    truth: list[tuple[int, int, int, int, int, str]],
    predicted: list[tuple[int, int, int, int, int, str]],
    tolerance: int = 1,
) -> tuple[tuple[tuple[int, int, int, int, int, str], tuple[int, int, int, int, int, str]], ...]:
    """Match cells by span and nearby grid coordinates, at most once each."""
    candidates: list[tuple[int, int, int]] = []
    for truth_index, truth_cell in enumerate(truth):
        for predicted_index, predicted_cell in enumerate(predicted):
            if truth_cell[3:5] != predicted_cell[3:5]:
                continue
            distance = abs(truth_cell[1] - predicted_cell[1]) + abs(
                truth_cell[2] - predicted_cell[2]
            )
            if distance <= tolerance * 2:
                candidates.append((distance, truth_index, predicted_index))
    matches: list[
        tuple[tuple[int, int, int, int, int, str], tuple[int, int, int, int, int, str]]
    ] = []
    used_truth: set[int] = set()
    used_predicted: set[int] = set()
    for _distance, truth_index, predicted_index in sorted(candidates):
        if truth_index in used_truth or predicted_index in used_predicted:
            continue
        matches.append((truth[truth_index], predicted[predicted_index]))
        used_truth.add(truth_index)
        used_predicted.add(predicted_index)
    return tuple(matches)


def ground_truth_table_cells(value: object) -> list[tuple[int, int, int, int, int, str]]:
    if not isinstance(value, list):
        return []
    cells = []
    for table_index, table in enumerate(value):
        if not isinstance(table, dict):
            continue
        table_cells = table.get("text")
        if not isinstance(table_cells, list):
            continue
        for cell in table_cells:
            if not isinstance(cell, dict):
                continue
            cells.append(
                (
                    table_index,
                    int(cast(Any, cell.get("x", 0))),
                    int(cast(Any, cell.get("y", 0))),
                    max(1, int(cast(Any, cell.get("w", 1)))),
                    max(1, int(cast(Any, cell.get("h", 1)))),
                    str(cell.get("content", "")),
                )
            )
    return cells


def internal_table_spans(record: object, y: int, x: int) -> tuple[int, int]:
    """Return (col_span, row_span) for one cell of a predicted table.

    Dictionary payloads store spans as (row_span, col_span) pairs; named keys
    are also accepted for fixture and diagnostic data.
    """
    spans = record.get("spans") if isinstance(record, dict) else getattr(record, "spans", None)
    if not isinstance(spans, (list, tuple)) or y >= len(spans):
        return 1, 1
    span_row = spans[y]
    if not isinstance(span_row, (list, tuple)) or x >= len(span_row):
        return 1, 1
    span = span_row[x]
    if isinstance(span, dict):
        return (
            max(1, int(cast(Any, span.get("col_span", 1)))),
            max(1, int(cast(Any, span.get("row_span", 1)))),
        )
    if isinstance(span, (list, tuple)) and len(span) >= 2:
        return (
            max(1, int(cast(Any, span[1]))),
            max(1, int(cast(Any, span[0]))),
        )
    return 1, 1


def predicted_table_cells(value: object) -> list[tuple[int, int, int, int, int, str]]:
    # The graph API returns structured Table objects directly.
    if not isinstance(value, (list, tuple)):
        return []
    cells = []
    for table_index, entry in enumerate(value):
        record = getattr(entry, "record", entry)
        rows = record.get("rows") if isinstance(record, dict) else getattr(record, "rows", None)
        if not isinstance(rows, (list, tuple)):
            continue
        for y, row in enumerate(rows):
            if not isinstance(row, (list, tuple)):
                continue
            for x, content in enumerate(row):
                if hasattr(content, "text"):
                    cells.append(
                        (
                            table_index,
                            x,
                            y,
                            max(1, int(getattr(content, "column_span", 1))),
                            max(1, int(getattr(content, "row_span", 1))),
                            str(getattr(content, "text", "") or ""),
                        )
                    )
                    continue
                col_span, row_span = internal_table_spans(record, y, x)
                cells.append((table_index, x, y, col_span, row_span, str(content or "")))
    return cells


def counter_f1(reference: Counter[Any], predicted: Counter[Any]) -> float:
    matched = sum((reference & predicted).values())
    total = sum(reference.values()) + sum(predicted.values())
    return 2 * matched / total if total else 1.0


def clean_score_bench_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        normalized = line.strip().lstrip("\ufeff")
        if not is_unstructured_marker(normalized):
            lines.append(line)
    return "\n".join(lines).replace("\ufeff", "")


def is_unstructured_marker(line: str) -> bool:
    if not line.startswith("-" * 10):
        return False
    prefix_len = len(line) - len(line.lstrip("-"))
    return prefix_len >= 10 and line[prefix_len:].lstrip().startswith("Unstructured ")


PUNCTUATION_TRANS = str.maketrans(
    {
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).translate(PUNCTUATION_TRANS)
    tokens: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current).casefold())
                current.clear()
            if not ch.isspace():
                tokens.append(ch.casefold())
    if current:
        tokens.append("".join(current).casefold())
    return tokens


@dataclass
class ScoreBench:
    root: Path = SCORE_BENCH_ROOT
    limit: int | None = None
    case_filters: tuple[str, ...] = ()
    json_output: Path | None = None
    html_output: Path | None = DEFAULT_HTML_OUTPUT
    candidate_analysis: bool = False
    extraction_analysis: bool = False
    full_results: bool = False
    report_limit: int = 25
    partition: str = "all"
    fail_on_errors: bool = False
    total_cases: int = field(init=False, default=0)
    started_at: float = field(init=False, default=0.0)

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> ScoreBench:
        parser = argparse.ArgumentParser(
            description="Score core_pdf extraction against SCORE-Bench ground truth."
        )
        parser.add_argument("--root", type=Path, default=SCORE_BENCH_ROOT)
        parser.add_argument("--limit", type=int)
        parser.add_argument("--case", action="append", default=[])
        parser.add_argument("--json-output", type=Path)
        parser.add_argument(
            "--html-output",
            type=Path,
            default=DEFAULT_HTML_OUTPUT,
            help="HTML report path. Defaults to parsebench-output/scorebench-report.html.",
        )
        parser.add_argument(
            "--no-html-output",
            action="store_true",
            help="Disable the default HTML report.",
        )
        parser.add_argument("--candidate-analysis", action="store_true")
        parser.add_argument("--extraction-analysis", action="store_true")
        parser.add_argument("--full-results", action="store_true")
        parser.add_argument(
            "--report-limit",
            type=int,
            default=25,
            help="Maximum rows per summary table unless --full-results is set.",
        )
        parser.add_argument("--fail-on-errors", action="store_true")
        parser.add_argument(
            "--partition",
            choices=("all", "train", "holdout"),
            default="all",
            help="Run all cases or the deterministic precision train/holdout partition.",
        )
        args = parser.parse_args(argv)
        return cls(
            root=args.root,
            limit=args.limit,
            case_filters=tuple(value.casefold() for value in args.case),
            json_output=args.json_output,
            html_output=None if args.no_html_output else args.html_output,
            candidate_analysis=args.candidate_analysis,
            extraction_analysis=args.extraction_analysis,
            full_results=args.full_results,
            report_limit=args.report_limit,
            fail_on_errors=args.fail_on_errors,
            partition=args.partition,
        )

    def run(self) -> int:
        cases = self.internal_selected_cases()
        if not cases:
            print(f"No SCORE-Bench cases found under {self.root}.", file=sys.stderr)
            return 2

        self.total_cases = len(cases)
        self.started_at = perf_counter()
        try:
            scores = self.internal_score_cases(cases)
        except ValueError as exc:
            print(f"Invalid arguments: {exc}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)
            return 130

        results = sorted(
            (
                NumberedCaseScore(case_number, score)
                for case_number, score in enumerate(scores, start=1)
            ),
            key=lambda item: (item.score.cct, item.case_number),
        )
        self.internal_write_json(results)
        self.internal_write_html(results)
        self.internal_print_report(results)
        if self.fail_on_errors and any(score.status != "ok" for score in scores):
            return 1
        return 0

    def internal_selected_cases(self) -> list[ScoreBenchCase]:
        cases = iter_score_bench_cases(self.root)
        if self.partition != "all":
            cases = [case for case in cases if score_bench_partition(case.stem) == self.partition]
        if self.case_filters:
            cases = [
                case
                for case in cases
                if any(value in case.stem.casefold() for value in self.case_filters)
            ]
        return cases if self.limit is None else cases[: self.limit]

    def internal_score_cases(self, cases: list[ScoreBenchCase]) -> list[CaseScore]:
        configure_native_thread_budget()
        os.environ["CORE_PDF_CANDIDATE_ANALYSIS"] = "1" if self.candidate_analysis else "0"
        os.environ["CORE_PDF_EXTRACTION_ANALYSIS"] = "1" if self.extraction_analysis else "0"
        os.environ["CORE_PDF_TRACE"] = "1" if self.extraction_analysis else "0"
        workers = min(os.cpu_count() or 4, len(cases)) if len(cases) > 1 else 1
        is_picklable = (
            workers > 1
            and getattr(score_case, "__qualname__", "").count(".") == 0
            and getattr(score_case, "__module__", None) is not None
        )
        if is_picklable:
            try:
                chunksize = max(1, len(cases) // (workers * 4))
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    return self.internal_collect_scores(
                        executor.map(score_case, cases, chunksize=chunksize)
                    )
            except Exception:
                pass
        return self.internal_collect_scores(map(score_case, cases))

    def internal_collect_scores(self, scored: Iterable[CaseScore]) -> list[CaseScore]:
        scores = list(scored)
        for case_number, score in enumerate(scores, start=1):
            self.internal_print_progress(case_number, score)
        return scores

    def internal_print_progress(self, case_number: int, score: CaseScore) -> None:
        # Progress suppressed; clean Markdown summary is printed at completion.
        pass

    def internal_print_report(self, results: list[NumberedCaseScore]) -> None:
        statuses = Counter(result.score.status for result in results)
        elapsed = perf_counter() - self.started_at
        successful = [result.score for result in results if result.score.status == "ok"]
        lines: list[str] = []
        lines.append("## SCORE-Bench Results\n")
        lines.append("| Cases | OK | Errors | Missing | Elapsed |")
        lines.append("|------:|---:|-------:|--------:|--------:|")
        lines.append(
            f"| {len(results)} "
            f"| {statuses['ok']} "
            f"| {statuses['error']} "
            f"| {statuses['missing_pdf']} "
            f"| {elapsed:.1f}s |"
        )
        if successful:
            lines.append("")
            lines.append("### Overall Metrics\n")
            lines.append("| Metric | Mean | Median | Min | Max | 95% CI |")
            lines.append("|:-------|-----:|-------:|----:|----:|:--------|")
            import statistics

            for title, values, suffix in score_metric_rows(successful):
                if not values:
                    continue
                mean = statistics.mean(values)
                median = statistics.median(values)
                mn = min(values)
                mx = max(values)
                low, high = bootstrap_interval(values)
                s = suffix
                lines.append(
                    f"| {title} | {mean:.4f}{s} | {median:.4f}{s} | {mn:.4f}{s} "
                    f"| {mx:.4f}{s} | {low:.4f}{s}–{high:.4f}{s} |"
                )
            expected_tables = sum(score.table_expected for score in successful)
            predicted_tables = sum(score.table_predicted for score in successful)
            matched_tables = sum(score.table_matched for score in successful)
            lines.append("")
            lines.append("### Table Coverage\n")
            lines.append("| Expected | Predicted | Matched | Recall | Precision |")
            lines.append("|---------:|----------:|--------:|-------:|----------:|")
            table_recall = matched_tables / expected_tables if expected_tables else 1.0
            table_precision = matched_tables / predicted_tables if predicted_tables else 0.0
            lines.append(
                f"| {expected_tables} | {predicted_tables} | {matched_tables} "
                f"| {table_recall:.4f} | {table_precision:.4f} |"
            )
            # Bucket breakdown
            bucket_counts = Counter(score_failure_bucket(score) for score in successful)
            lines.append("")
            lines.append("### Bucket Breakdown\n")
            lines.append("| Bucket | Count |")
            lines.append("|:-------|------:|")
            for bucket, count in sorted(bucket_counts.items()):
                lines.append(f"| {bucket} | {count} |")
            # Weakest 25 by CCT
            weakest = sorted(successful, key=lambda s: s.cct)[:25]
            lines.append("")
            lines.append("### Weakest 25 Cases (by CCT)\n")
            lines.append("| # | Stem | Bucket | CCT | F1 | Recall | Prec | Time |")
            lines.append("|--:|:-----|:-------|----:|---:|-------:|-----:|-----:|")
            for rank, score in enumerate(weakest, 1):
                hints = score_hints(score)
                lines.append(
                    f"| {rank} "
                    f"| {shorten_middle(score.stem, 42)} "
                    f"| {score_failure_bucket(score)} "
                    f"| {score.cct:.4f} "
                    f"| {score.content_f1:.4f} "
                    f"| {score.percent_tokens_found:.4f} "
                    f"| {score.precision:.4f} "
                    f"| {score.text_elapsed_seconds:.2f}s |"
                )
                if hints:
                    lines.append(f"|   | *{hints}* |  |  |  |  |  |  |")
        errors = [result for result in results if result.score.error]
        if errors:
            lines.append("")
            lines.append("### Errors\n")
            for result in errors:
                lines.append(
                    f"- **{result.case_number}. {result.score.stem}**: {result.score.error}"
                )
        if self.html_output is not None:
            lines.append(f"\n> HTML report: `{self.html_output}`")
        print("\n".join(lines))

    def internal_print_bucket_summary(self, scores: list[CaseScore]) -> None:
        pass  # Integrated into internal_print_report Markdown output.

    def internal_limit_results(self, results: list[NumberedCaseScore]) -> list[NumberedCaseScore]:
        return results if self.full_results else results[: self.report_limit]

    def internal_write_json(self, results: list[NumberedCaseScore]) -> None:
        if self.json_output is None:
            return
        self.json_output.parent.mkdir(parents=True, exist_ok=True)
        self.json_output.write_text(
            json.dumps([asdict(result.score) for result in results], indent=2) + "\n",
            encoding="utf-8",
        )

    def internal_write_html(self, results: list[NumberedCaseScore]) -> None:
        if self.html_output is None:
            return
        self.html_output.parent.mkdir(parents=True, exist_ok=True)
        self.html_output.write_text(
            self.internal_render_html(results, perf_counter() - self.started_at),
            encoding="utf-8",
        )

    def internal_render_html(self, results: list[NumberedCaseScore], elapsed_seconds: float) -> str:
        statuses = Counter(result.score.status for result in results)
        successful = [result.score for result in results if result.score.status == "ok"]
        metric_cards = "\n".join(
            render_metric_card(title, values, suffix)
            for title, values, suffix in score_metric_rows(successful)
            if values
        )
        bucket_cards = "\n".join(
            f"""
            <div class="bucket-card">
              <span>{html.escape(bucket)}</span>
              <strong>{count}</strong>
            </div>
            """
            for bucket, count in sorted(
                Counter(score_failure_bucket(score) for score in successful).items()
            )
        )
        expected_tables = sum(score.table_expected for score in successful)
        predicted_tables = sum(score.table_predicted for score in successful)
        matched_tables = sum(score.table_matched for score in successful)
        table_recall = matched_tables / expected_tables if expected_tables else 1.0
        table_precision = matched_tables / predicted_tables if predicted_tables else 0.0
        table_coverage = f"""
  <section>
    <h2>Table Coverage</h2>
    <div class="run-meta">
      {render_meta_card("Expected", str(expected_tables))}
      {render_meta_card("Predicted", str(predicted_tables))}
      {render_meta_card("Matched", str(matched_tables))}
      {render_meta_card("Recall", f"{table_recall:.4f}")}
      {render_meta_card("Precision", f"{table_precision:.4f}")}
    </div>
  </section>
"""
        sections = "\n".join(
            render_html_section(title, self.internal_limit_results(section_results))
            for title, section_results in score_result_sections(results)
            if section_results
        )
        errors = "\n".join(
            f"<li><strong>{result.case_number}. {html.escape(result.score.stem)}</strong>: "
            f"{html.escape(result.score.error or '')}</li>"
            for result in results
            if result.score.error
        )
        errors_block = f"<section><h2>Errors</h2><ul>{errors}</ul></section>" if errors else ""
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SCORE-Bench Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #18202a;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #0f766e;
      --bad: #b42318;
      --warn: #b54708;
      --good: #027a48;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 28px; }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-end;
      margin-bottom: 20px;
    }}
    h1 {{ margin: 0; font-size: 28px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin: 18px 0;
    }}
    .run-meta, .bucket-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      align-items: stretch;
    }}
    .meta-card, .metric-card, .bucket-card {{
      background: #fbfcfe;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .meta-card span, .metric-card span, .bucket-card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .meta-card strong, .metric-card strong, .bucket-card strong {{
      display: block;
      font-size: 20px;
    }}
    .metric-stats {{
      display: grid;
      grid-template-columns: repeat(5, minmax(42px, 1fr));
      gap: 8px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .metric-stats div {{
      min-width: 0;
    }}
    .metric-stats dt {{
      margin: 0 0 2px;
      font-weight: 600;
    }}
    .metric-stats dd {{
      margin: 0;
      color: var(--text);
      font-variant-numeric: tabular-nums;
      overflow-wrap: anywhere;
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: right;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 600; }}
    td.stem, td.hints, th.stem, th.hints {{
      text-align: left;
      white-space: normal;
      min-width: 220px;
    }}
    td.bucket, th.bucket {{ text-align: left; }}
    .bucket-pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      background: #eef4ff;
      color: #175cd3;
      font-size: 12px;
      white-space: nowrap;
    }}
    .bucket-pill.empty-output, .bucket-pill.missing-most, .bucket-pill.low-f1 {{
      background: #fef3f2;
      color: var(--bad);
    }}
    .bucket-pill.low-precision, .bucket-pill.low-recall,
    .bucket-pill.order-gap, .bucket-pill.table-structure {{
      background: #fffaeb;
      color: var(--warn);
    }}
    .score-good {{ color: var(--good); }}
    .score-bad {{ color: var(--bad); }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>SCORE-Bench Report</h1>
      <div class="muted">Generated by scripts/score_unstructured_bench.py</div>
      <div class="muted">Scoring schema {SCORING_SCHEMA_VERSION}</div>
    </div>
  </header>
  <section>
    <h2>Run Summary</h2>
    <div class="run-meta">
      {render_meta_card("Elapsed", f"{elapsed_seconds:.2f}s")}
      {render_meta_card("Completed", f"{len(results)}/{self.total_cases}")}
      {render_meta_card("OK", str(statuses["ok"]))}
      {render_meta_card("Errors", str(statuses["error"]))}
      {render_meta_card("Missing PDFs", str(statuses["missing_pdf"]))}
    </div>
  </section>
  <section>
    <h2>Score Summary</h2>
    <div class="metric-grid">
      {metric_cards}
    </div>
  </section>
  {table_coverage}
  <section>
    <h2>Buckets</h2>
    <div class="bucket-grid">
      {bucket_cards}
    </div>
  </section>
  {sections}
  {errors_block}
</main>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    return ScoreBench.from_cli(argv).run()


if __name__ == "__main__":
    raise SystemExit(main())
