"""Benchmarks for rendering a page: composing a display list, and rasterizing it.

Not collected by a bare `pytest` run; see tests/benchmarks/README.md.

These exist because rendering had no benchmark. Extraction and rendering are
siblings over the same captured page program, so a change can move one without
the other -- the glyph flyweights moved extraction and not rendering, the
rasterizer kernels moved rendering and not extraction. Measuring only one of
them left the other to wall-clock timings on a developer machine, which is the
instrument this suite exists to replace.

Compose and rasterize are separate benchmarks rather than one, because they
fail differently: compose walks the program and builds display items, while
rasterize is scanline fills and compositing. A regression in one would be
invisible inside the sum.
"""

import os
from functools import cache

import pytest
from pytest_codspeed import BenchmarkFixture

from core_pdf import PdfDocument
from core_pdf.impl.render.model import RenderOptions
from core_pdf.impl.render.page import compose_page
from tests.benchmarks.corpus import RENDER_SAMPLES, Sample

REQUIRE_FIXTURES = os.environ.get("CORE_PDF_BENCHMARK_REQUIRE_FIXTURES") == "1"


@cache
def sample_bytes(sample: Sample) -> bytes:
    if not sample.file.exists():
        message = f"fixture not present, needs a submodule checkout: {sample.path}"
        if REQUIRE_FIXTURES:
            raise FileNotFoundError(message)
        pytest.skip(message)
    data = sample.file.read_bytes()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"not a PDF, probably an unfetched LFS pointer: {sample.path}")
    return data


@pytest.mark.parametrize("sample", RENDER_SAMPLES, ids=lambda sample: sample.id)
def test_compose_page(benchmark: BenchmarkFixture, sample: Sample) -> None:
    """Capture is measured by the extract benchmarks and stays outside this one."""
    with PdfDocument(sample_bytes(sample)) as document:
        page = document.pages[0]
        program = page.get_page_program()
        options = RenderOptions()
        rendered = benchmark(lambda: compose_page(page, options, page_program=program))
        assert rendered is not None


@pytest.mark.parametrize("sample", RENDER_SAMPLES, ids=lambda sample: sample.id)
def test_rasterize_page(benchmark: BenchmarkFixture, sample: Sample) -> None:
    """Composing is measured above and is left outside the measured region."""
    with PdfDocument(sample_bytes(sample)) as document:
        rendered = document.pages[0].render()
        raster = benchmark(lambda: rendered.rasterize(scale=1.0))
        assert raster.width > 0
