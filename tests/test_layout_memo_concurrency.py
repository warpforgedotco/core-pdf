# SPDX-License-Identifier: AGPL-3.0-only
"""Concurrency guards for layout and geometry queries on shared pages."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from core_pdf import PdfDocument
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
