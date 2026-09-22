"""Benchmarks for opening documents and extracting pages.

These are not collected by a bare `pytest` run: `testpaths` in `pyproject.toml`
does not include this directory, so they only run when named explicitly. See
`tests/benchmarks/README.md` for how to run them under CodSpeed.
"""

from functools import cache

import pytest

from core_pdf import PdfDocument
from tests.benchmarks.corpus import (
    EXTRACT_SAMPLES,
    OPEN_SAMPLES,
    SLICE_PAGES,
    SLICE_SAMPLES,
    Sample,
)


@cache
def sample_bytes(sample: Sample) -> bytes:
    """Read a sample once, outside the measured region.

    Reference corpora are git submodules, so a file can legitimately be absent
    in a working tree that has not run `git submodule update --init`.
    """
    if not sample.file.exists():
        pytest.skip(f"fixture not present, needs a submodule checkout: {sample.path}")
    return sample.file.read_bytes()


def open_document(data: bytes) -> int:
    with PdfDocument(data) as document:
        return len(document.pages)


def extract_pages(data: bytes, limit: int) -> int:
    with PdfDocument(data) as document:
        return sum(len(page.extract().blocks) for page in document.pages[:limit])


@pytest.mark.parametrize("sample", OPEN_SAMPLES, ids=lambda sample: sample.id)
def test_open_document(benchmark, sample):
    data = sample_bytes(sample)
    assert benchmark(open_document, data) >= 1


@pytest.mark.parametrize("sample", EXTRACT_SAMPLES, ids=lambda sample: sample.id)
def test_extract_first_page(benchmark, sample):
    data = sample_bytes(sample)
    benchmark(extract_pages, data, 1)


@pytest.mark.parametrize("sample", SLICE_SAMPLES, ids=lambda sample: sample.id)
def test_extract_page_slice(benchmark, sample):
    data = sample_bytes(sample)
    benchmark(extract_pages, data, SLICE_PAGES)
