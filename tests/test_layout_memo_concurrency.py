# SPDX-License-Identifier: AGPL-3.0-only
"""Concurrency guards for the layout/geometry memos stored on shared TextRuns."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from core_pdf import PdfDocument
from core_pdf.impl import document as engine_document
from tests.helpers.paths import score_bench_pdf


def internal_page():
    return PdfDocument.open(score_bench_pdf("Employee_Health_Benefits_Assess-p006.pdf"))


def test_concurrent_geometry_extraction_on_one_page_is_consistent() -> None:
    with internal_page() as document:
        page = document.pages[0]

        with ThreadPoolExecutor(max_workers=8) as pool:
            issue_results = list(pool.map(lambda _: page.extract_geometry_issues(), range(16)))
            summary_results = list(pool.map(lambda _: page.extract_geometry_summary(), range(16)))
            run_results = list(pool.map(lambda _: page.text_diagnostics().runs, range(16)))

    assert all(result == issue_results[0] for result in issue_results)
    assert all(result == summary_results[0] for result in summary_results)
    assert run_results[0]
    assert all(result == run_results[0] for result in run_results)
    assert summary_results[0].text_run_count == len(run_results[0])


def test_get_text_lines_returns_one_shared_list_under_concurrency(monkeypatch) -> None:
    workers = 8

    original_layout_line = engine_document.LayoutLine

    class SlowLayoutLine(original_layout_line):  # type: ignore[valid-type,misc]
        def __init__(self, runs, *args, **kwargs) -> None:
            time.sleep(0.0005)
            original_layout_line.__init__(self, runs, *args, **kwargs)

    monkeypatch.setattr(engine_document, "LayoutLine", SlowLayoutLine)

    with internal_page() as document:
        page = document.pages[0]

        page.get_page_program()
        page.text_lines = None

        start = threading.Barrier(workers)

        def call_after_barrier(_: int) -> list[object]:
            start.wait()
            return page.get_text_lines()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            line_lists = list(pool.map(call_after_barrier, range(workers)))

    first = line_lists[0]
    assert all(result is first for result in line_lists), (
        f"get_text_lines produced {len({id(r) for r in line_lists})} distinct lists"
    )
