"""Benchmarks for opening documents and extracting pages.

These are not collected by a bare `pytest` run: `testpaths` in `pyproject.toml`
does not include this directory, so they only run when named explicitly. See
`tests/benchmarks/README.md` for how to run them under CodSpeed.
"""

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

# CI sets this so a submodule that failed to check out fails the job instead
# of silently skipping every benchmark and reporting success having measured
# nothing. Locally the skip is the friendlier behaviour.
REQUIRE_FIXTURES = os.environ.get("CORE_PDF_BENCHMARK_REQUIRE_FIXTURES") == "1"


@cache
def sample_bytes(sample: Sample) -> bytes:
    """Read a sample once, outside the measured region.

    Reference corpora are git submodules, so a file can legitimately be absent
    in a working tree that has not run `git submodule update --init`.
    """
    if not sample.file.exists():
        message = f"fixture not present, needs a submodule checkout: {sample.path}"
        if REQUIRE_FIXTURES:
            raise FileNotFoundError(message)
        pytest.skip(message)
    data = sample.file.read_bytes()
    if not data.startswith(b"%PDF"):
        # Parts of some reference corpora are kept in git LFS. Without a pull
        # those paths exist as small pointer files, which would otherwise be
        # benchmarked as if they were documents.
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
    # Opening is measured by test_open_document and is left outside the
    # measured region here. Keeping it inside made this benchmark mostly a
    # document-open measurement for samples with an expensive xref: the
    # billionaires_page sample spent 8.3 s of its 10.6 s opening the file.
    with PdfDocument(sample_bytes(sample)) as document:
        assert benchmark(extract_pages, document, 1) >= 0


@pytest.mark.parametrize("sample", SLICE_SAMPLES, ids=lambda sample: sample.id)
def test_extract_page_slice(benchmark: BenchmarkFixture, sample: Sample) -> None:
    with PdfDocument(sample_bytes(sample)) as document:
        assert benchmark(extract_pages, document, SLICE_PAGES) >= 0
