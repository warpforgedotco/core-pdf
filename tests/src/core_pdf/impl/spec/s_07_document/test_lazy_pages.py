from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from core_pdf.impl.runtime.cache import ExtractionCache
from core_pdf.impl.spec.s_07_document.document_pages import LazyPageList
from core_pdf.impl.spec.s_07_syntax.types import PdfDict


class FakePage:
    created = 0

    def __init__(self, document: object, page_dict: PdfDict, page_number: int) -> None:
        type(self).created += 1
        self.document = document
        self.page_dict = page_dict
        self.page_number = page_number
        self.extraction_cache: ExtractionCache | None = None


class FakeDocument:
    def __init__(self, count: int = 3) -> None:
        self.internal_cache_lock = threading.RLock()
        self.page_dicts_cache: list[PdfDict] | None = None
        self.page_class: type | None = FakePage
        self.internal_dicts: list[PdfDict] = [{"page": index + 1} for index in range(count)]

    def iter_page_dicts_stream(self) -> Iterator[PdfDict]:
        yield from self.internal_dicts

    def page_count(self) -> int:
        return len(self.internal_dicts)


def test_lazy_pages_are_read_only_and_realize_only_requested_items() -> None:
    FakePage.created = 0
    pages = LazyPageList[FakePage](FakeDocument())

    assert len(pages) == 3
    assert not hasattr(pages, "append")
    assert FakePage.created == 0

    first = pages[0]
    assert pages[0] is first
    assert first.page_number == 1
    assert FakePage.created == 1

    sliced = pages[:2]
    assert isinstance(sliced, tuple)
    assert sliced[0] is first
    assert tuple(page.page_number for page in sliced) == (1, 2)
    assert pages[-1].page_number == 3


def test_lazy_pages_realize_each_page_once_under_concurrent_indexing() -> None:
    FakePage.created = 0
    pages = LazyPageList[FakePage](FakeDocument())

    with ThreadPoolExecutor(max_workers=8) as executor:
        resolved = tuple(executor.map(lambda _: pages[1], range(32)))

    assert all(page is resolved[0] for page in resolved)
    assert FakePage.created == 2
