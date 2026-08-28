# SPDX-License-Identifier: AGPL-3.0-only
"""Performance bounds for page content interpretation and glyph capture."""

from __future__ import annotations

from pathlib import Path

import pytest

from core_pdf import PdfDocument

FIXTURES = Path(__file__).parents[6] / "fixtures"
LATIN_PDF = FIXTURES / "pypdf" / "resources" / "crazyones.pdf"
CID_PDF = FIXTURES / "pdfminer.six" / "samples" / "jo.pdf"


def internal_reinterpret(page) -> object:
    page.page_program_cache = None
    return page.get_page_program()


@pytest.mark.benchmark_high_impact
def test_latin_page_capture_benchmark(benchmark) -> None:
    with PdfDocument.open(LATIN_PDF) as document:
        page = document.pages[0]
        page.get_page_program()  # warm stream, font, and decoder caches
        program = benchmark(internal_reinterpret, page)
        assert len(program.products.glyphs) == 730


@pytest.mark.benchmark_high_impact
def test_cid_page_capture_benchmark(benchmark) -> None:
    with PdfDocument.open(CID_PDF) as document:
        page = document.pages[0]
        page.get_page_program()  # warm stream, font, and decoder caches
        program = benchmark(internal_reinterpret, page)
        assert len(program.products.glyphs) == 898
