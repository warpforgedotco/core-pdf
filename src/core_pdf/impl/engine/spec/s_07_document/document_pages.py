# SPDX-License-Identifier: AGPL-3.0-only
"""Shared page-tree constants and the lazily populated page list."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import (
    Generic,
    Protocol,
    SupportsIndex,
    TypeVar,
    cast,
    overload,
)

from core_pdf.impl.runtime.cache import ExtractionCache
from core_pdf.impl.runtime.cache_lock import document_cache_lock
from core_pdf.impl.types import PdfDict


class PageListItem(Protocol):
    page_dict: PdfDict
    extraction_cache: ExtractionCache | None


internal_PageT = TypeVar("internal_PageT", bound=PageListItem)
PageFactory = Callable[[object, PdfDict, int], internal_PageT]


class PageListDocument(Protocol):
    page_dicts_cache: list[PdfDict] | None
    page_class: type | None

    def iter_page_dicts_stream(self) -> Iterator[PdfDict]: ...

    def page_count(self) -> int: ...


class LazyPageList(Sequence[internal_PageT], Generic[internal_PageT]):
    """A read-only sequence that resolves page dictionaries only when requested."""

    __slots__ = ("document", "page_dict_iter", "complete", "internal_items")

    document: PageListDocument
    page_dict_iter: Iterator[PdfDict] | None
    complete: bool

    def __init__(self, document: PageListDocument) -> None:
        self.document = document
        self.page_dict_iter = None
        self.complete = False
        self.internal_items: list[internal_PageT] = []

    def next_page_dict(self) -> PdfDict:
        document = self.document
        cached_dicts = document.page_dicts_cache
        current_len = len(self.internal_items)
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
            document.page_dicts_cache = [page.page_dict for page in self.internal_items]
            raise IndexError("page index out of range") from None

    def ensure(self, index: int) -> None:
        with document_cache_lock(self.document):
            while len(self.internal_items) <= index:
                page_dict = self.next_page_dict()
                page_class = self.document.page_class
                if page_class is None:
                    from core_pdf.impl.engine.spec.s_07_document.page import PdfPage

                    page_class = PdfPage
                factory = cast(PageFactory[internal_PageT], page_class)
                self.internal_items.append(
                    factory(self.document, page_dict, len(self.internal_items) + 1)
                )

    def __len__(self) -> int:
        return self.document.page_count()

    def __iter__(self) -> Iterator[internal_PageT]:
        index = 0
        count = len(self)
        while index < count:
            try:
                yield self[index]
            except IndexError:
                return
            index += 1

    @overload
    def __getitem__(self, item: SupportsIndex) -> internal_PageT: ...

    @overload
    def __getitem__(self, item: slice[SupportsIndex | None]) -> tuple[internal_PageT, ...]: ...

    def __getitem__(
        self, item: SupportsIndex | slice[SupportsIndex | None]
    ) -> internal_PageT | tuple[internal_PageT, ...]:
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            return tuple(self[page_index] for page_index in range(start, stop, step))
        index = item.__index__()
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError("page index out of range")
        self.ensure(index)
        return self.internal_items[index]


__all__ = (
    "LazyPageList",
    "PageListItem",
)
