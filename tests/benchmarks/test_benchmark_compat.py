import os
from functools import cache

import pytest
from pytest_codspeed import BenchmarkFixture

import core_pdf_compat.pdfplumber
from core_pdf.impl.fonts_cmap import MAX_CMAP_EXPANSION, ToUnicodeCMap
from tests.benchmarks.corpus import COMPAT_SAMPLES, Sample

REQUIRE_FIXTURES = os.environ.get("CORE_PDF_BENCHMARK_REQUIRE_FIXTURES") == "1"

# Two full two-byte ranges: the most a ToUnicode CMap may expand from ranges.
LARGEST_ACCEPTED_CMAP = (
    b"1 begincodespacerange <0000> <ffff> endcodespacerange\n"
    b"2 beginbfrange <0000> <ffff> <0041> <0000> <ffff> <0042> endbfrange"
)


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


def chars_then_layout(page: core_pdf_compat.pdfplumber.Page) -> int:
    page.flush_cache()
    return len(page.chars) + len(list(page.layout))


@pytest.mark.parametrize("sample", COMPAT_SAMPLES, ids=lambda sample: sample.id)
def test_pdfplumber_chars_then_layout(benchmark: BenchmarkFixture, sample: Sample) -> None:
    with core_pdf_compat.pdfplumber.open(sample_bytes(sample)) as pdf:
        page = pdf.pages[0]
        assert benchmark(chars_then_layout, page) > 0


def test_parse_largest_accepted_tounicode_cmap(benchmark: BenchmarkFixture) -> None:
    cmap = benchmark(ToUnicodeCMap, LARGEST_ACCEPTED_CMAP)
    assert len(cmap.mappings) == MAX_CMAP_EXPANSION // 2
