# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol, SupportsIndex, cast, overload

from core_pdf.impl.engine.spec.s_07_document.page import PdfPage
from core_pdf.impl.types import PdfDict

PageFactory = Callable[[object, PdfDict, int], PdfPage]


class PageListDocument(Protocol):
    page_dicts_cache: list[PdfDict] | None

    def iter_page_dicts_stream(self) -> Iterator[PdfDict]: ...

    def page_count(self) -> int: ...


class LazyPageList(list[PdfPage]):
    __slots__ = ("document", "page_dict_iter", "complete")

    document: PageListDocument
    page_dict_iter: Iterator[PdfDict] | None
    complete: bool

    def __init__(self, document: PageListDocument) -> None:
        super().__init__()
        self.document = document
        self.page_dict_iter: Iterator[PdfDict] | None = None
        self.complete = False

    def next_page_dict(self) -> PdfDict:
        document = self.document
        cached_dicts = document.page_dicts_cache
        current_len = list.__len__(self)
        if cached_dicts is not None:
            if current_len >= len(cached_dicts):
                self.complete = True
                raise IndexError("page index out of range")
            return cached_dicts[current_len]

        if self.page_dict_iter is None:
            self.page_dict_iter = document.iter_page_dicts_stream()
        try:
            return next(self.page_dict_iter)
        except StopIteration:
            self.complete = True
            document.page_dicts_cache = [page.page_dict for page in list.__iter__(self)]
            invalidate = getattr(document, "invalidate_document_extraction_cache", None)
            if callable(invalidate):
                invalidate()
            raise IndexError("page index out of range") from None

    def ensure(self, index: int) -> None:
        while list.__len__(self) <= index:
            page_dict = self.next_page_dict()
            page_class = cast(PageFactory, getattr(self.document, "page_class", PdfPage))
            list.append(self, page_class(self.document, page_dict, list.__len__(self) + 1))

    def __len__(self) -> int:
        return self.document.page_count()

    def __iter__(self) -> Iterator[PdfPage]:
        index = 0
        while True:
            try:
                yield self[index]
            except IndexError:
                return
            index += 1

    @overload
    def __getitem__(self, item: SupportsIndex) -> PdfPage: ...

    @overload
    def __getitem__(self, item: slice[SupportsIndex | None]) -> list[PdfPage]: ...

    def __getitem__(
        self, item: SupportsIndex | slice[SupportsIndex | None]
    ) -> PdfPage | list[PdfPage]:
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            return [self[page_index] for page_index in range(start, stop, step)]
        index = item.__index__()
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError("page index out of range")
        self.ensure(index)
        return list.__getitem__(self, index)


__all__ = ("LazyPageList",)
