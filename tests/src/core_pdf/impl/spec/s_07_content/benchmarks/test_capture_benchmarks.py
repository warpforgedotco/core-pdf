# SPDX-License-Identifier: AGPL-3.0-only
"""Performance bounds for page content interpretation and glyph capture."""

from __future__ import annotations

import pytest

from core_pdf import PdfDocument
from tests.helpers.paths import FIXTURES

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
        assert len(program.glyphs) == 730


@pytest.mark.benchmark_high_impact
def test_cid_page_capture_benchmark(benchmark) -> None:
    with PdfDocument.open(CID_PDF) as document:
        page = document.pages[0]
        page.get_page_program()  # warm stream, font, and decoder caches
        program = benchmark(internal_reinterpret, page)
        assert len(program.glyphs) == 898
