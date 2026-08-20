# SPDX-License-Identifier: AGPL-3.0-only
"""End-to-end benchmarks over real-world PDFs from the SCORE-Bench corpus.

Complements ``test_synthetic_document_benchmarks.py``: these fixtures exercise
the routing, OCR, and font-substitution paths that only show up in real
scanned/authored documents, at the cost of needing the SCORE-Bench LFS assets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.engine.rendering import RenderOptions

FIXTURES = Path(__file__).parents[2] / "fixtures" / "SCORE-Bench" / "src"
internal_HIGH_IMPACT_FIXTURES = frozenset(
    {
        "Employee_Health_Benefits_Assess-p006.pdf",
        "esp32_s3_circuit_schematic.pdf",
        "global-AIDS-strategy-p74-75-p001.pdf",
    }
)


def internal_shard_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Select one deterministic corpus shard, defaulting to the complete corpus."""
    try:
        shard_count = int(os.environ.get("CORE_PDF_BENCHMARK_SHARD_COUNT", "1"))
        shard_index = int(os.environ.get("CORE_PDF_BENCHMARK_SHARD_INDEX", "0"))
    except ValueError as exc:
        raise pytest.UsageError("benchmark shard index and count must be integers") from exc
    if shard_count < 1:
        raise pytest.UsageError("benchmark shard count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise pytest.UsageError("benchmark shard index must be within the configured shard count")
    return paths[shard_index::shard_count]


def internal_page_case(path: Path) -> Any:
    marks = pytest.mark.benchmark_high_impact if path.name in internal_HIGH_IMPACT_FIXTURES else ()
    return pytest.param(path.name, id=path.stem, marks=marks)


internal_PDF_PATHS = tuple(
    path
    for path in sorted(FIXTURES.glob("*"))
    if path.is_file() and path.suffix.casefold() == ".pdf"
)
PAGE_CASES = tuple(internal_page_case(path) for path in internal_shard_paths(internal_PDF_PATHS))


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
