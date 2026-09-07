from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .support import FIXTURES_ROOT

real_pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.compat_differential
internal_PDFS = tuple(
    FIXTURES_ROOT / "PyMuPDF/tests/resources" / name
    for name in ("small-table.pdf", "test_2957_1.pdf", "test_4043.pdf")
)


@pytest.mark.parametrize("pdf_path", internal_PDFS, ids=lambda path: path.name)
@pytest.mark.parametrize("source", ["path", "filename", "bytes", "bytearray", "buffer"])
def test_open_and_navigate(pdf_path: Path, source: str) -> None:
    def snapshot(module: Any) -> dict[str, Any]:
        data = pdf_path.read_bytes()
        if source == "path":
            document = module.open(pdf_path)
        elif source == "filename":
            document = module.Document(filename=str(pdf_path))
        else:
            stream = {"bytes": data, "bytearray": bytearray(data), "buffer": BytesIO(data)}[source]
            document = module.open(stream=stream, filetype="pdf")
        with document:
            count = len(document)
            pages = [
                (page.number, tuple(page.rect), page.rotation, page.parent is document)
                for page in document
            ]
            return {
                "name": document.name,
                "metadata": document.metadata,
                "count": (count, document.page_count, document.chapter_page_count(0)),
                "state": (document.is_pdf, document.is_closed, document.is_encrypted),
                "chapters": document.chapter_count,
                "pages": pages,
                "negative": document[-count - 1].number,
                "load_negative": document.load_page(-count - 1).number,
                "chapter_address": document[(0, count - 1)].number,
                "load_chapter_address": document.load_page((0, count - 1)).number,
                "membership": [
                    index in document
                    for index in (-count - 1, 0, count, (0, 0), (1, 0), (0, -1), True, "0")
                ],
                "slice": [page.number for page in document[::-1]],
                "range": [page.number for page in document.pages(0, count + 10)],
                "reverse": [page.number for page in document.pages(count - 1, -1, -1)],
                "wrapping_start": [page.number for page in document.pages(-count - 1)],
            }

    assert snapshot(compat_pymupdf) == snapshot(real_pymupdf)


@pytest.mark.parametrize("pdf_path", internal_PDFS, ids=lambda path: path.name)
def test_empty_creation_page_insertion_and_close(pdf_path: Path) -> None:
    def snapshot(module: Any) -> tuple[Any, ...]:
        with module.open(pdf_path) as source:
            width, height = tuple(source[0].rect)[2:]
        document = module.open()
        empty = (
            document.name,
            document.metadata.copy(),
            len(document),
            list(document.pages()),
            document.get_toc(),
            document.chapter_page_count(0),
        )
        original = document.new_page(width=width, height=height)
        first = (original.number, tuple(original.rect), original.get_text())
        document.new_page(pno=0, width=height, height=width)
        orphan = (original.parent, original.number)
        with pytest.raises(AssertionError):
            _ = original.rect
        inserted = [(page.number, tuple(page.rect), page.get_text()) for page in document.pages()]
        page = document.load_page(1)
        document.close()
        closed = (document.is_closed, page.parent, page.number)
        errors = []
        for operation in (
            lambda: len(document),
            lambda: document.page_count,
            lambda: document.load_page(0),
            lambda: list(document.pages()),
            lambda: document.new_page(),
            lambda: document.close(),
        ):
            with pytest.raises(ValueError) as error:
                operation()
            errors.append(str(error.value))
        return empty, first, orphan, inserted, closed, errors

    assert snapshot(compat_pymupdf) == snapshot(real_pymupdf)


@pytest.mark.parametrize("pdf_path", internal_PDFS, ids=lambda path: path.name)
def test_document_access_errors(pdf_path: Path) -> None:
    def snapshot(module: Any) -> list[tuple[str, str]]:
        with module.open(pdf_path) as document:
            outcomes = []
            for operation in (
                lambda: document[len(document)],
                lambda: document.load_page(len(document)),
                lambda: document.load_page("bad"),
                lambda: document[True],
                lambda: document[(1, 0)],
                lambda: document.load_page((0, -1)),
                lambda: list(document.pages(0, len(document), 0)),
                lambda: list(document.pages(len(document) + 1)),
                lambda: document.new_page(pno=-2),
                lambda: document.new_page(pno=len(document) + 1),
                lambda: module.open(stream=pdf_path.read_bytes()[:0]),
            ):
                try:
                    operation()
                except Exception as error:
                    outcomes.append((type(error).__name__, str(error)))
                else:
                    pytest.fail("expected the operation to reject its arguments")
            return outcomes

    assert snapshot(compat_pymupdf) == snapshot(real_pymupdf)
