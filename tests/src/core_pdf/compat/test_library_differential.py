from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

FIXTURES_ROOT = Path("tests/fixtures")
ALL_PDFS = tuple(sorted(FIXTURES_ROOT.rglob("*.pdf")))
XRAY_ROOT = Path("tests/fixtures/x-ray").resolve()


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_pdfplumber_matches_real_library_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_pdfplumber = pytest.importorskip("pdfplumber")
    from core_pdf.api.compat import pdfplumber as compat_pdfplumber

    with real_pdfplumber.open(pdf_path) as expected, compat_pdfplumber.open(pdf_path) as actual:
        assert len(actual.pages) == len(expected.pages)
        for actual_page, expected_page in zip(actual.pages, expected.pages, strict=True):
            assert actual_page.extract_text() == expected_page.extract_text()
            assert _words(actual_page.extract_words()) == _words(expected_page.extract_words())
            assert (actual_page.width, actual_page.height) == (
                expected_page.width,
                expected_page.height,
            )


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_pypdf_matches_real_library_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_pypdf = pytest.importorskip("pypdf")
    from core_pdf.api.compat import pypdf as compat_pypdf

    expected = real_pypdf.PdfReader(pdf_path, strict=False)
    actual = compat_pypdf.PdfReader(pdf_path, strict=False)
    assert len(actual.pages) == len(expected.pages)
    for actual_page, expected_page in zip(actual.pages, expected.pages, strict=True):
        assert actual_page.extract_text() == expected_page.extract_text()
        assert tuple(actual_page.mediabox) == tuple(expected_page.mediabox)
        assert actual_page.rotation == expected_page.rotation


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_pymupdf_matches_real_library_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_pymupdf = pytest.importorskip("pymupdf")
    from core_pdf.api.compat import pymupdf as compat_pymupdf

    with real_pymupdf.open(pdf_path) as expected, compat_pymupdf.open(pdf_path) as actual:
        assert len(actual) == len(expected)
        for actual_page, expected_page in zip(actual, expected, strict=True):
            assert actual_page.get_text() == expected_page.get_text()
            assert actual_page.get_text("words") == expected_page.get_text("words")
            assert tuple(actual_page.rect) == tuple(expected_page.rect)


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_pikepdf_matches_real_library_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_pikepdf = pytest.importorskip("pikepdf")
    from core_pdf.api.compat import pikepdf as compat_pikepdf

    with real_pikepdf.Pdf.open(pdf_path) as expected, compat_pikepdf.Pdf.open(pdf_path) as actual:
        assert len(actual.pages) == len(expected.pages)
        assert [tuple(page.mediabox) for page in actual.pages] == [
            tuple(page.mediabox) for page in expected.pages
        ]
        assert _metadata(actual.docinfo) == _metadata(expected.docinfo)


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_unstructured_matches_real_library_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_partition = pytest.importorskip("unstructured.partition.pdf").partition_pdf
    from core_pdf.api.compat.unstructured import partition_pdf as compat_partition

    expected = real_partition(filename=str(pdf_path), strategy="fast")
    actual = compat_partition(pdf_path)
    assert [(item.category, item.text) for item in actual] == [
        (item.category, item.text) for item in expected
    ]


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_llamaindex_matches_real_reader_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_reader = pytest.importorskip("llama_index.readers.file").PDFReader
    from core_pdf.api.compat.llamaindex import load_data

    expected = real_reader().load_data(file=pdf_path)
    actual = load_data(pdf_path)
    assert [document.text for document in actual] == [document.text for document in expected]


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_xray_matches_real_library_on_all_fixture_pdfs(
    monkeypatch: pytest.MonkeyPatch,
    pdf_path: Path,
) -> None:
    monkeypatch.syspath_prepend(str(XRAY_ROOT))
    sys.modules.pop("xray", None)
    real_xray = pytest.importorskip("xray")
    from core_pdf.api.compat.xray import inspect

    assert inspect(pdf_path) == real_xray.inspect(pdf_path)


def _metadata(value: Any) -> dict[str, str]:
    return {
        str(key).lstrip("/"): str(item)
        for key, item in value.items()
        if str(key) not in {"info", "xmp"}
    }


def _words(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: round(value, 5) if isinstance(value, float) else value for key, value in word.items()}
        for word in values
    ]
