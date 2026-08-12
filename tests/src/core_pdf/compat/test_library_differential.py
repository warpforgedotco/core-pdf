from __future__ import annotations

import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import pytest

FIXTURES_ROOT = Path("tests/fixtures")
ALL_PDFS = tuple(sorted(FIXTURES_ROOT.rglob("*.pdf")))
XRAY_ROOT = Path("tests/fixtures/x-ray").resolve()


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_pdfplumber_matches_real_library_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_pdfplumber = pytest.importorskip("pdfplumber")
    from core_pdf.api.compat import pdfplumber as compat_pdfplumber

    with ExitStack() as stack:
        pair = _open_pair(
            stack,
            lambda: real_pdfplumber.open(pdf_path),
            lambda: compat_pdfplumber.open(pdf_path),
        )
        if pair is None:
            return
        expected, actual = pair
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

    def snapshot(reader_type: Any) -> tuple[tuple[str, tuple[float, ...], int], ...]:
        with reader_type(pdf_path, strict=False) as reader:
            return tuple(
                (page.extract_text(), tuple(page.mediabox), page.rotation) for page in reader.pages
            )

    pair = _call_pair(
        lambda: snapshot(real_pypdf.PdfReader), lambda: snapshot(compat_pypdf.PdfReader)
    )
    if pair is None:
        return
    expected, actual = pair
    assert actual == expected


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_pymupdf_matches_real_library_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_pymupdf = pytest.importorskip("pymupdf")
    from core_pdf.api.compat import pymupdf as compat_pymupdf

    with ExitStack() as stack:
        pair = _open_pair(
            stack,
            lambda: real_pymupdf.open(pdf_path),
            lambda: compat_pymupdf.open(pdf_path),
        )
        if pair is None:
            return
        expected, actual = pair
        assert len(actual) == len(expected)
        for actual_page, expected_page in zip(actual, expected, strict=True):
            assert actual_page.get_text() == expected_page.get_text()
            assert actual_page.get_text("words") == expected_page.get_text("words")
            assert tuple(actual_page.rect) == tuple(expected_page.rect)


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_pikepdf_matches_real_library_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_pikepdf = pytest.importorskip("pikepdf")
    from core_pdf.api.compat import pikepdf as compat_pikepdf

    with ExitStack() as stack:
        pair = _open_pair(
            stack,
            lambda: real_pikepdf.Pdf.open(pdf_path),
            lambda: compat_pikepdf.Pdf.open(pdf_path),
        )
        if pair is None:
            return
        expected, actual = pair
        assert len(actual.pages) == len(expected.pages)
        assert [tuple(page.mediabox) for page in actual.pages] == [
            tuple(page.mediabox) for page in expected.pages
        ]
        assert _metadata(actual.docinfo) == _metadata(expected.docinfo)


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_unstructured_matches_real_library_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_partition = pytest.importorskip("unstructured.partition.pdf").partition_pdf
    from core_pdf.api.compat.unstructured import partition_pdf as compat_partition

    pair = _call_pair(
        lambda: real_partition(filename=str(pdf_path), strategy="fast"),
        lambda: compat_partition(pdf_path),
    )
    if pair is None:
        return
    expected, actual = pair
    assert [(item.category, item.text) for item in actual] == [
        (item.category, item.text) for item in expected
    ]


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_llamaindex_matches_real_reader_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_reader = pytest.importorskip("llama_index.readers.file").PDFReader
    from core_pdf.api.compat.llamaindex import load_data

    pair = _call_pair(
        lambda: real_reader().load_data(file=pdf_path),
        lambda: load_data(pdf_path),
    )
    if pair is None:
        return
    expected, actual = pair
    assert [document.text for document in actual] == [document.text for document in expected]


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_xray_matches_real_library_on_all_fixture_pdfs(
    monkeypatch: pytest.MonkeyPatch,
    pdf_path: Path,
) -> None:
    monkeypatch.syspath_prepend(str(XRAY_ROOT))
    real_pymupdf = pytest.importorskip("pymupdf")
    monkeypatch.setitem(sys.modules, "fitz", real_pymupdf)
    for module_name in tuple(sys.modules):
        if module_name == "xray" or module_name.startswith("xray."):
            sys.modules.pop(module_name)
    real_xray = pytest.importorskip("xray")
    from core_pdf.api.compat.xray import inspect

    pair = _call_pair(lambda: real_xray.inspect(pdf_path), lambda: inspect(pdf_path))
    if pair is None:
        return
    expected, actual = pair
    assert actual == expected


def _open_pair(
    stack: ExitStack,
    expected_factory: Callable[[], Any],
    actual_factory: Callable[[], Any],
) -> tuple[Any, Any] | None:
    try:
        expected = stack.enter_context(expected_factory())
    except Exception:
        try:
            stack.enter_context(actual_factory())
        except Exception:
            return None
        pytest.fail("reference rejected the PDF but compat accepted it")
    try:
        actual = stack.enter_context(actual_factory())
    except Exception as error:
        pytest.fail(f"reference accepted the PDF but compat rejected it: {error!r}")
    return expected, actual


def _call_pair(
    expected_factory: Callable[[], Any],
    actual_factory: Callable[[], Any],
) -> tuple[Any, Any] | None:
    try:
        expected = expected_factory()
    except Exception:
        try:
            actual_factory()
        except Exception:
            return None
        pytest.fail("reference rejected the PDF but compat accepted it")
    try:
        actual = actual_factory()
    except Exception as error:
        pytest.fail(f"reference accepted the PDF but compat rejected it: {error!r}")
    return expected, actual


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
