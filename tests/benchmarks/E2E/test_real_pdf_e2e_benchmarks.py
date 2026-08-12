# SPDX-License-Identifier: AGPL-3.0-only
"""End-to-end benchmarks over real-world PDFs from the SCORE-Bench corpus.

Complements ``test_synthetic_document_benchmarks.py``: these fixtures exercise
the routing, OCR, and font-substitution paths that only show up in real
scanned/authored documents, at the cost of needing the SCORE-Bench LFS assets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.engine.rendering import RenderOptions

FIXTURES = Path(__file__).parents[2] / "fixtures" / "SCORE-Bench" / "src"
PAGE_CASES = (
    pytest.param(
        "global-AIDS-strategy-p74-75-p001.pdf",
        id="hybrid",
    ),
    pytest.param(
        "esp32_s3_circuit_schematic.pdf",
        id="ocr",
    ),
    pytest.param(
        "Employee_Health_Benefits_Assess-p006.pdf",
        id="native",
    ),
    pytest.param(
        "NASA-SNA-8-D-027III-Rev2-CsmLmSpacecraftOperationalDataBook-Volume3-MassProperties-Pg54.pdf",
        id="heavy-corrupt-native-ocr",
    ),
    pytest.param(
        "153rd-Omaha-Pow-Wow-p001.pdf",
        id="heavy-image-only-ocr",
    ),
    pytest.param(
        "NASA-SNA-8-D-027III-Rev2-CsmLmSpacecraftOperationalDataBook-Volume3-MassProperties-pg856.pdf",
        id="heavy-native-object-streams",
    ),
    pytest.param(
        "Zhand-Ilavsky-p012.pdf",
        id="heavy-hybrid-image-supplement",
    ),
    pytest.param(
        "Index_to_Positions_table_vertical_text-p001.pdf",
        id="heavy-hidden-vertical-text",
    ),
    pytest.param(
        "Mission-costs_p27-35-p008.pdf",
        id="heavy-wide-image-supplement",
    ),
)


def internal_fixture(name: str) -> Path:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip()
    return path


def internal_open_extract_render(path: Path) -> dict[str, Any]:
    with PdfDocument.open(path) as document:
        extracted = document.extract()
        raster_pixels = 0
        for page in document.pages:
            rendered = page.render(RenderOptions(include_text=False))
            raster = rendered.rasterize(scale=2.0, cache=False)
            raster_pixels += raster.width * raster.height
        return {
            "pages": len(extracted.pages),
            "text_chars": len(extracted.text),
            "raster_pixels": raster_pixels,
        }


@pytest.mark.parametrize("fixture_name", PAGE_CASES)
def test_real_pdf_open_extract_render_benchmark(benchmark, fixture_name: str) -> None:
    result = benchmark.pedantic(
        internal_open_extract_render,
        args=(internal_fixture(fixture_name),),
        iterations=1,
        rounds=1,
    )
    assert result["pages"] > 0
    assert result["raster_pixels"] > 0
