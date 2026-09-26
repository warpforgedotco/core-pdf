import os
from functools import cache

import pytest
from pytest_codspeed import BenchmarkFixture

from core_pdf import PdfDocument
from tests.benchmarks.corpus import (
    EXTRACT_SAMPLES,
    OPEN_SAMPLES,
    SLICE_PAGES,
    SLICE_SAMPLES,
    Sample,
)

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


def open_document(data: bytes) -> int:
    with PdfDocument(data) as document:
        return len(document.pages)


def extract_pages(document: PdfDocument, limit: int) -> int:
    return sum(len(page.extract().blocks) for page in document.pages[:limit])


@pytest.mark.parametrize("sample", OPEN_SAMPLES, ids=lambda sample: sample.id)
def test_open_document(benchmark: BenchmarkFixture, sample: Sample) -> None:
    data = sample_bytes(sample)
    assert benchmark(open_document, data) >= 1


@pytest.mark.parametrize("sample", EXTRACT_SAMPLES, ids=lambda sample: sample.id)
def test_extract_first_page(benchmark: BenchmarkFixture, sample: Sample) -> None:
    with PdfDocument(sample_bytes(sample)) as document:
        assert benchmark(extract_pages, document, 1) >= 0


@pytest.mark.parametrize("sample", SLICE_SAMPLES, ids=lambda sample: sample.id)
def test_extract_page_slice(benchmark: BenchmarkFixture, sample: Sample) -> None:
    with PdfDocument(sample_bytes(sample)) as document:
        assert benchmark(extract_pages, document, SLICE_PAGES) >= 0
