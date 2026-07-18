from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from core_pdf import PdfDocument

ROOT = Path(__file__).resolve().parents[1]
SCORE_BENCH_ROOT = ROOT / "tests" / "fixtures" / "SCORE-Bench"
CORRECT_TRUTH_SUFFIX = "__correct_truth"
CONSOLE = Console()


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
    error: str | None = None
    missing_top: list[tuple[str, int]] | None = None
    extra_top: list[tuple[str, int]] | None = None


@dataclass(frozen=True)
class NumberedCaseScore:
    case_number: int
    score: CaseScore


@dataclass(slots=True)
class ActiveCase:
    case_number: int
    stem: str
    started_at: float


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
        with PdfDocument(case.pdf) as document:
            predicted_text = extract_document_text(document)
        cct, found, added, precision, gt_count, predicted_count, matched = score_tokens(
            gt_text, predicted_text
        )
        token_diff = score_token_diff_summary(gt_text, predicted_text)
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
            missing_top=token_diff["missing_top"][:5],
            extra_top=token_diff["extra_top"][:5],
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


def score_cases(
    cases: list[ScoreBenchCase],
    *,
    on_score: Callable[[int, CaseScore], None] | None = None,
    on_case_started: Callable[[int, ScoreBenchCase], None] | None = None,
    jobs: int = 1,
) -> list[CaseScore]:
    if jobs < 1:
        raise ValueError("--jobs must be at least 1")
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    if jobs == 1 or len(cases) <= 1:
        scores = []
        for case_number, case in enumerate(cases, start=1):
            if on_case_started is not None:
                on_case_started(case_number, case)
            score = score_case(case)
            scores.append(score)
            if on_score is not None:
                on_score(case_number, score)
        return scores

    scores_by_number: dict[int, CaseScore] = {}
    max_workers = min(jobs, len(cases))
    numbered_cases = list(enumerate(cases, start=1))
    queued_case_numbers = deque(case_number for case_number, _case in numbered_cases)
    running_case_numbers: set[int] = set()

    def promote_cases() -> None:
        while queued_case_numbers and len(running_case_numbers) < max_workers:
            case_number = queued_case_numbers.popleft()
            running_case_numbers.add(case_number)
            if on_case_started is not None:
                on_case_started(case_number, cases[case_number - 1])

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(score_numbered_case, case_number, case): case_number
            for case_number, case in numbered_cases
        }
        promote_cases()
        for future in concurrent.futures.as_completed(futures):
            case_number, score = future.result()
            scores_by_number[case_number] = score
            running_case_numbers.discard(case_number)
            promote_cases()
            if on_score is not None:
                on_score(case_number, score)
    return [scores_by_number[index] for index in range(1, len(cases) + 1)]


def score_numbered_case(
    case_number: int,
    case: ScoreBenchCase,
) -> tuple[int, CaseScore]:
    return case_number, score_case(case)


def is_full_precision_score(score: CaseScore) -> bool:
    return (
        score.status == "ok"
        and score.cct >= 1.0
        and score.percent_tokens_found >= 1.0
        and score.precision >= 1.0
        and score.gt_tokens == score.predicted_tokens == score.matched_tokens
    )


class ScoreBenchUI:
    def __init__(
        self,
        *,
        total_cases: int,
        jobs: int,
        filters: list[str],
        limit: int | None,
    ) -> None:
        self.total_cases = total_cases
        self.jobs = jobs
        self.filters = filters
        self.limit = limit
        self.started_at = perf_counter()
        self.completed_cases = 0
        self.status_counts: Counter[str] = Counter()
        self.recent_results: deque[NumberedCaseScore] = deque(
            maxlen=recent_results_history_limit(total_cases)
        )
        self.completed_results: list[NumberedCaseScore] = []
        self.queued_cases: deque[tuple[int, str]] = deque()
        self.running_cases: dict[int, ActiveCase] = {}
        self.progress = Progress(
            TextColumn("[bold blue]Scoring"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total} cases"),
            TimeElapsedColumn(),
            console=CONSOLE,
        )
        self.progress_task_id: TaskID = self.progress.add_task("score-bench", total=total_cases)

    def set_case_queue(self, cases: list[ScoreBenchCase]) -> None:
        self.queued_cases = deque(
            (case_number, case.stem) for case_number, case in enumerate(cases, start=1)
        )
        self.running_cases.clear()

    def on_case_started(self, case_number: int, case: ScoreBenchCase) -> None:
        queued_case = self.queued_cases[0] if self.queued_cases else None
        if queued_case is not None and queued_case[0] == case_number:
            self.queued_cases.popleft()
        else:
            self.queued_cases = deque(item for item in self.queued_cases if item[0] != case_number)
        self.running_cases[case_number] = ActiveCase(
            case_number=case_number,
            stem=case.stem,
            started_at=perf_counter(),
        )

    def on_score(self, case_number: int, score: CaseScore) -> None:
        numbered_score = NumberedCaseScore(case_number=case_number, score=score)
        self.running_cases.pop(case_number, None)
        self.completed_cases += 1
        self.status_counts[score.status] += 1
        self.recent_results.appendleft(numbered_score)
        self.completed_results.append(numbered_score)
        self.progress.update(self.progress_task_id, advance=1)

    def render_live(self) -> Layout:
        layout = Layout(name="scorebench")
        layout.split_column(
            Layout(self._header_panel(), name="header", size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(name="main", ratio=3),
            Layout(name="side", ratio=2),
        )
        layout["main"].split_column(
            Layout(self._recent_results_panel(), name="recent", ratio=2),
            Layout(self._worst_cases_panel(), name="worst", ratio=1),
            Layout(self._progress_panel(), name="progress", size=5),
        )
        layout["side"].split_column(
            Layout(self._summary_panel(), name="summary", size=9),
            Layout(self._active_cases_panel(), name="active", size=14),
            Layout(self._score_distribution_panel(), name="distribution", size=11),
            Layout(self._focus_case_panel(), name="focus", ratio=1),
        )
        return layout

    def render_final(self, *, full_results: bool = False) -> list[Panel]:
        panels: list[Panel] = [
            Panel(
                self._summary_table(),
                title="SCORE-Bench Summary",
                subtitle=self._subtitle(),
                border_style="green",
            ),
            Panel(
                self._score_distribution_table(),
                title="Score Bands",
                border_style="cyan",
            ),
            Panel(
                self._ranked_results_table(
                    ranked_results=self._sorted_results_by_precision()[:25],
                    title="Worst 25 Cases by Precision",
                ),
                border_style="yellow",
            ),
            Panel(
                self._ranked_results_table(
                    ranked_results=self._sorted_results_by_runtime()[:25],
                    title="Worst 25 Cases by Runtime",
                ),
                border_style="yellow",
            ),
            Panel(
                self._ranked_results_table(
                    ranked_results=list(reversed(self._sorted_results_by_cct()[-25:])),
                    title="Best 25 Cases",
                ),
                border_style="green",
            ),
            self._focus_case_panel(),
        ]
        if full_results:
            panels.append(
                Panel(
                    self._ranked_results_table(
                        ranked_results=self._sorted_results_by_cct(),
                        title="All Results by CCT",
                    ),
                    border_style="white",
                )
            )
        return panels

    def render_error_panel(self) -> Panel | None:
        error_results = [
            result for result in self.completed_results if result.score.error is not None
        ]
        if not error_results:
            return None
        table = Table(expand=True, box=None, pad_edge=False)
        table.add_column("Case", style="bold red", no_wrap=True)
        table.add_column("Error", overflow="fold")
        for result in error_results:
            table.add_row(
                f"{result.case_number}. {result.score.stem}",
                result.score.error or "",
            )
        return Panel(table, title="Errors", border_style="red")

    def render_output_panel(self, *, json_output: Path | None) -> Panel | None:
        rows: list[tuple[str, str]] = []
        if json_output is not None:
            rows.append(("JSON output", str(json_output)))
        if not rows:
            return None
        table = Table(expand=True, box=None, pad_edge=False)
        table.add_column("Output", style="bold cyan", no_wrap=True)
        table.add_column("Details", overflow="fold")
        for label, value in rows:
            table.add_row(label, value)
        return Panel(table, title="Outputs", border_style="cyan")

    def _header_panel(self) -> Panel:
        return Panel(
            self._subtitle(),
            title="SCORE-Bench",
            border_style="blue",
        )

    def _progress_panel(self) -> Panel:
        return Panel(self.progress, title="Progress", border_style="cyan")

    def _summary_panel(self) -> Panel:
        return Panel(self._summary_table(), title="Summary", border_style="green")

    def _recent_results_panel(self) -> Panel:
        return Panel(
            self._recent_results_table(),
            title="Recent Results",
            border_style="white",
        )

    def _worst_cases_panel(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_row(
            self._compact_worst_cases_table(
                ranked_results=self._sorted_results_by_precision()[:5],
                title="By Precision",
                metric_label="Precision",
                metric_value=lambda score: format_ratio(score.precision),
            ),
            self._compact_worst_cases_table(
                ranked_results=self._sorted_results_by_runtime()[:5],
                title="By Runtime",
                metric_label="Time",
                metric_value=lambda score: format_seconds(score.elapsed_seconds),
            ),
        )
        return Panel(
            table,
            title="Worst Cases",
            border_style="yellow",
        )

    def _score_distribution_panel(self) -> Panel:
        return Panel(
            self._score_distribution_table(),
            title="Score Bands",
            border_style="cyan",
        )

    def _active_cases_panel(self) -> Panel:
        return Panel(
            self._active_cases_table(),
            title="Active Cases",
            border_style="blue",
        )

    def _focus_case_panel(self) -> Panel:
        return Panel(
            self._focus_case_table(),
            title="Focus Case",
            border_style="magenta",
        )

    def _summary_table(self) -> Table:
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_row(
            self._summary_metric(
                "Completed",
                f"{self.completed_cases}/{self.total_cases}",
                "bold white",
            ),
            self._summary_metric(
                "Healthy",
                str(self.status_counts.get("ok", 0)),
                "bold green",
            ),
            self._summary_metric(
                "Errors",
                str(self.status_counts.get("error", 0)),
                "bold red",
            ),
            self._summary_metric(
                "Missing PDFs",
                str(self.status_counts.get("missing_pdf", 0)),
                "bold yellow",
            ),
        )
        return table

    def _summary_metric(self, label: str, value: str, style: str) -> Text:
        text = Text()
        text.append(f"{label}\n", style="dim")
        text.append(value, style=style)
        return text

    def _status_table(self) -> Table:
        table = Table.grid(expand=True)
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        table.add_row("Jobs", str(self.jobs))
        table.add_row("Elapsed", format_seconds(perf_counter() - self.started_at))
        table.add_row(
            "Filters",
            ",".join(self.filters) if self.filters else "all cases",
        )
        table.add_row("Limit", str(self.limit) if self.limit is not None else "none")
        return table

    def _active_cases_table(self) -> Table:
        table = Table(expand=True, box=None, pad_edge=False)
        table.add_column("State", style="bold", no_wrap=True)
        table.add_column("Case", no_wrap=True)
        table.add_column("Age", justify="right", no_wrap=True)
        table.add_column("Stem", overflow="ellipsis")
        running = sorted(self.running_cases.values(), key=lambda item: item.case_number)
        if running:
            for active in running:
                table.add_row(
                    Text("running", style="bold green"),
                    str(active.case_number),
                    format_seconds(perf_counter() - active.started_at),
                    active.stem,
                )
        else:
            table.add_row(
                Text("running", style="dim"),
                "-",
                "-",
                "Waiting for an available case...",
            )
        queued_preview = list(self.queued_cases)[: max(1, 6 - len(running))]
        for case_number, stem in queued_preview:
            table.add_row(Text("queued", style="yellow"), str(case_number), "-", stem)
        queued_remaining = len(self.queued_cases) - len(queued_preview)
        if queued_remaining > 0:
            table.add_row(
                Text("queued", style="yellow"),
                f"+{queued_remaining}",
                "-",
                "more queued cases",
            )
        completed = self.completed_cases
        if completed > 0:
            table.add_row(
                Text("done", style="cyan"),
                str(completed),
                "-",
                "completed so far",
            )
        return table

    def _focus_case_table(self) -> Table:
        table = Table.grid(expand=True)
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        focus = self._focus_case_result()
        if focus is None:
            table.add_row("State", "Waiting for scored cases...")
            return table
        score = focus.score
        table.add_row("Case", f"{focus.case_number}. {score.stem}")
        table.add_row("Status", score.status)
        table.add_row("CCT", format_ratio(score.cct))
        table.add_row("Recall", format_ratio(score.percent_tokens_found))
        table.add_row("Precision", format_ratio(score.precision))
        table.add_row("Missing", format_focus_tokens(score.missing_top))
        table.add_row("Extra", format_focus_tokens(score.extra_top))
        if score.error is not None:
            table.add_row("Error", score.error)
        else:
            table.add_row("Run", format_focus_run_state(self))
        return table

    def _score_distribution_table(self) -> Table:
        table = Table.grid(expand=True)
        table.add_column(style="bold", ratio=1)
        table.add_column(justify="right", no_wrap=True)
        table.add_column(ratio=2)
        counts = score_band_counts(self.completed_results)
        for label, _lower, _upper, style in score_bands():
            count = counts[label]
            table.add_row(
                Text(label, style=style),
                str(count),
                score_band_bar(count, self.total_cases, style),
            )
        return table

    def _recent_results_table(self) -> Table:
        table = Table(expand=True, box=box.SIMPLE_HEAVY, pad_edge=False)
        table.add_column("#", style="bold", justify="right", no_wrap=True)
        table.add_column("CCT", justify="right", no_wrap=True)
        table.add_column("Recall", justify="right", no_wrap=True)
        table.add_column("Precision", justify="right", no_wrap=True)
        table.add_column("Added", justify="right", no_wrap=True)
        table.add_column("Time", justify="right", no_wrap=True)
        table.add_column("Tokens", justify="right", no_wrap=True)
        table.add_column("Stem", overflow="ellipsis")
        if not self.recent_results:
            table.add_row("-", "-", "-", "-", "-", "-", "-", "Waiting for first completed case...")
            return table
        visible_results = list(self.recent_results)[: self._recent_results_visible_limit()]
        for result in visible_results:
            score = result.score
            table.add_row(
                str(result.case_number),
                format_ratio(score.cct),
                format_ratio(score.percent_tokens_found),
                format_ratio(score.precision),
                format_ratio(score.percent_tokens_added),
                format_seconds(score.elapsed_seconds),
                f"{score.matched_tokens}/{score.gt_tokens}",
                score.stem,
                style=row_style(score),
            )
        return table

    def _recent_results_visible_limit(self) -> int:
        terminal_height = CONSOLE.size.height
        reserved_height = 26
        available_body_rows = max(4, terminal_height - reserved_height)
        return max(6, min(len(self.recent_results), available_body_rows))

    def _compact_worst_cases_table(
        self,
        *,
        ranked_results: list[NumberedCaseScore],
        title: str,
        metric_label: str,
        metric_value: Callable[[CaseScore], str],
    ) -> Table:
        table = Table(expand=True)
        table.add_column("Case", style="bold", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column(metric_label, justify="right", no_wrap=True)
        table.add_column("CCT", justify="right", no_wrap=True)
        table.add_column("Stem", overflow="ellipsis")
        if not ranked_results:
            table.add_row("-", "-", "-", "-", "Waiting for scored cases...")
            return table
        for result in ranked_results:
            score = result.score
            table.add_row(
                str(result.case_number),
                Text(score.status, style=status_style(score.status)),
                metric_value(score),
                format_ratio(score.cct),
                score.stem,
                style=row_style(score),
            )
        table.title = title
        return table

    def _focus_case_result(self) -> NumberedCaseScore | None:
        if not self.completed_results:
            return None
        return min(self.completed_results, key=numbered_score_sort_key)

    def _sorted_results_by_cct(self) -> list[NumberedCaseScore]:
        return sorted(self.completed_results, key=numbered_score_sort_key)

    def _sorted_results_by_precision(self) -> list[NumberedCaseScore]:
        return sorted(self.completed_results, key=numbered_score_precision_sort_key)

    def _sorted_results_by_runtime(self) -> list[NumberedCaseScore]:
        return sorted(self.completed_results, key=numbered_score_runtime_sort_key)

    def _ranked_results_table(
        self,
        *,
        ranked_results: list[NumberedCaseScore],
        title: str,
    ) -> Table:
        table = Table(expand=True)
        table.add_column("Rank", justify="right", no_wrap=True)
        table.add_column("Case", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("CCT", justify="right", no_wrap=True)
        table.add_column("Recall", justify="right", no_wrap=True)
        table.add_column("Precision", justify="right", no_wrap=True)
        table.add_column("Added", justify="right", no_wrap=True)
        table.add_column("Time", justify="right", no_wrap=True)
        table.add_column("Tokens", justify="right", no_wrap=True)
        table.add_column("Pred", justify="right", no_wrap=True)
        table.add_column("Stem", overflow="ellipsis")
        if not ranked_results:
            table.add_row("-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-")
            return table
        for rank, result in enumerate(ranked_results, start=1):
            score = result.score
            table.add_row(
                str(rank),
                str(result.case_number),
                Text(score.status, style=status_style(score.status)),
                format_ratio(score.cct),
                format_ratio(score.percent_tokens_found),
                format_ratio(score.precision),
                format_ratio(score.percent_tokens_added),
                format_seconds(score.elapsed_seconds),
                f"{score.matched_tokens}/{score.gt_tokens}",
                str(score.predicted_tokens),
                score.stem,
                style=row_style(score),
            )
        table.title = title
        return table

    def _subtitle(self) -> str:
        parts = [
            f"jobs={self.jobs}",
            f"elapsed={format_seconds(perf_counter() - self.started_at)}",
        ]
        if self.filters:
            parts.append(f"filters={','.join(self.filters)}")
        if self.limit is not None:
            parts.append(f"limit={self.limit}")
        return " | ".join(parts)


def status_style(status: str) -> str:
    return {
        "ok": "green",
        "error": "red",
        "missing_pdf": "yellow",
    }.get(status, "white")


def row_style(score: CaseScore) -> str:
    if score.status != "ok":
        return status_style(score.status)
    if is_full_precision_score(score):
        return "dim"
    if score.cct < 0.85:
        return "bold red"
    if score.cct < 0.95:
        return "yellow"
    return "white"


def score_bands() -> list[tuple[str, float, float | None, str]]:
    return [
        ("<0.70", 0.0, 0.70, "bold red"),
        ("0.70-0.85", 0.70, 0.85, "red"),
        ("0.85-0.95", 0.85, 0.95, "yellow"),
        ("0.95-0.99", 0.95, 0.99, "green"),
        ("0.99+", 0.99, None, "bold green"),
    ]


def score_band_counts(results: list[NumberedCaseScore]) -> dict[str, int]:
    counts = {label: 0 for label, _lower, _upper, _style in score_bands()}
    for result in results:
        for label, lower, upper, _style in score_bands():
            if score_in_band(result.score, lower, upper):
                counts[label] += 1
                break
    return counts


def score_in_band(score: CaseScore, lower: float, upper: float | None) -> bool:
    if score.status != "ok":
        return False
    if score.cct < lower:
        return False
    if upper is None:
        return True
    return score.cct < upper


def score_band_bar(count: int, total: int, style: str) -> Text:
    width = 16
    filled = 0 if total <= 0 else max(0, min(width, round(width * count / total)))
    bar = Text()
    bar.append("█" * filled, style=style)
    bar.append("·" * (width - filled), style="dim")
    return bar


def recent_results_history_limit(total_cases: int) -> int:
    return max(24, min(64, total_cases))


def format_focus_tokens(tokens: list[tuple[str, int]] | None) -> str:
    if not tokens:
        return "-"
    return ", ".join(f"{token}×{count}" for token, count in tokens[:3])


def format_focus_run_state(ui: ScoreBenchUI) -> str:
    return (
        f"{ui.completed_cases}/{ui.total_cases} complete"
        f" | jobs={ui.jobs}"
        f" | elapsed={format_seconds(perf_counter() - ui.started_at)}"
    )


def format_ratio(value: float) -> str:
    return f"{value:.4f}"


def format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def update_live_ui(
    live: Live,
    ui: ScoreBenchUI,
    case_number: int,
    score: CaseScore,
) -> None:
    ui.on_score(case_number, score)
    live.update(ui.render_live(), refresh=True)


def update_plain_ui(
    ui: ScoreBenchUI,
    case_number: int,
    score: CaseScore,
) -> None:
    ui.on_score(case_number, score)


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


def tokenize(text: str) -> list[str]:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score core_pdf text extraction against SCORE-Bench content ground truth."
    )
    parser.add_argument("--root", type=Path, default=SCORE_BENCH_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of SCORE-Bench cases to score concurrently.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Opt into the full-screen live Rich dashboard.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--full-results",
        action="store_true",
        help="Print the full sorted results table in addition to the capped summary tables.",
    )
    parser.add_argument("--fail-on-errors", action="store_true")
    args = parser.parse_args()

    cases = iter_score_bench_cases(args.root)
    if args.case:
        filters = [case_filter.casefold() for case_filter in args.case]
        cases = [case for case in cases if any(f in case.stem.casefold() for f in filters)]
    else:
        filters = []
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        CONSOLE.print(
            Panel(
                f"No SCORE-Bench cases found under {args.root}.",
                title="SCORE-Bench",
                border_style="red",
            )
        )
        return 2

    ui = ScoreBenchUI(
        total_cases=len(cases),
        jobs=args.jobs,
        filters=filters,
        limit=args.limit,
    )
    ui.set_case_queue(cases)

    try:
        if args.live:
            with Live(
                ui.render_live(),
                console=CONSOLE,
                screen=True,
                auto_refresh=False,
                transient=True,
            ) as live:
                scores = score_cases(
                    cases,
                    on_case_started=ui.on_case_started,
                    on_score=lambda case_number, score: update_live_ui(
                        live,
                        ui,
                        case_number,
                        score,
                    ),
                    jobs=args.jobs,
                )
        else:
            scores = score_cases(
                cases,
                on_case_started=ui.on_case_started,
                on_score=lambda case_number, score: update_plain_ui(
                    ui,
                    case_number,
                    score,
                ),
                jobs=args.jobs,
            )
    except ValueError as exc:
        CONSOLE.print(
            Panel(
                str(exc),
                title="Invalid Arguments",
                border_style="red",
            )
        )
        return 2
    except KeyboardInterrupt:
        CONSOLE.print(Panel("Interrupted.", title="SCORE-Bench", border_style="yellow"))
        sys.exit(130)

    ranked_scores = sorted(
        (
            NumberedCaseScore(case_number=case_number, score=score)
            for case_number, score in enumerate(scores, start=1)
        ),
        key=numbered_score_sort_key,
    )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps([asdict(result.score) for result in ranked_scores], indent=2) + "\n",
            encoding="utf-8",
        )
    if args.live:
        CONSOLE.clear()
    for panel in ui.render_final(full_results=args.full_results):
        CONSOLE.print(panel)
    output_panel = ui.render_output_panel(json_output=args.json_output)
    if output_panel is not None:
        CONSOLE.print(output_panel)
    error_panel = ui.render_error_panel()
    if error_panel is not None:
        CONSOLE.print(error_panel)

    if args.fail_on_errors and any(score.status != "ok" for score in scores):
        return 1
    return 0


def numbered_score_sort_key(item: NumberedCaseScore) -> tuple[float, int, str]:
    return (item.score.cct, item.case_number, item.score.stem)


def numbered_score_precision_sort_key(
    item: NumberedCaseScore,
) -> tuple[float, float, int, str]:
    return (
        item.score.precision,
        item.score.cct,
        item.case_number,
        item.score.stem,
    )


def numbered_score_runtime_sort_key(
    item: NumberedCaseScore,
) -> tuple[float, float, int, str]:
    return (
        -item.score.elapsed_seconds,
        item.score.cct,
        item.case_number,
        item.score.stem,
    )


if __name__ == "__main__":
    raise SystemExit(main())
