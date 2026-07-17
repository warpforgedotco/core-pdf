# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

from core_pdf.impl.types import PageSelection

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_document.page import PdfPage


class DocumentSelectionHost(Protocol):
    @property
    def pages(self) -> Sequence[PdfPage]: ...

    def selected_page_indexes(self, pages: PageSelection | None = None) -> list[int]: ...


class DocumentSelectionMixin:
    def selected_page_indexes(
        self: DocumentSelectionHost, pages: PageSelection | None = None
    ) -> list[int]:
        page_count = len(self.pages)
        if pages is None:
            return list(range(page_count))
        if type(pages) is int:
            selected = [pages - 1]
        elif isinstance(pages, range):
            selected = [page_number - 1 for page_number in pages]
        elif isinstance(pages, str):
            selected = []
            for part in pages.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    parts = part.split("-", 1)
                    if not parts[0].strip() or not parts[1].strip():
                        raise ValueError(f"invalid page selection: {pages!r}")
                    try:
                        start = int(parts[0])
                        end = int(parts[1])
                    except ValueError as exc:
                        raise ValueError(f"invalid page selection: {pages!r}") from exc
                    step = 1 if end >= start else -1
                    selected.extend(range(start - 1, end - 1 + step, step))
                else:
                    try:
                        selected.append(int(part) - 1)
                    except ValueError as exc:
                        raise ValueError(f"invalid page selection: {pages!r}") from exc
        elif isinstance(pages, Sequence):
            try:
                selected = [int(cast(Any, page_number)) - 1 for page_number in pages]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid page selection: {pages!r}") from exc
        else:
            raise TypeError(f"invalid page selection: {pages!r}")

        if not selected:
            raise ValueError(f"invalid page selection: {pages!r}")

        normalized: list[int] = []
        seen: set[int] = set()
        for page_index in selected:
            if page_index < 0 or page_index >= page_count:
                raise IndexError(f"page selection out of range: {page_index + 1}")
            if page_index not in seen:
                normalized.append(page_index)
                seen.add(page_index)
        return normalized

    def iter_selected_pages(
        self: DocumentSelectionHost, pages: PageSelection | None = None
    ) -> Iterator[tuple[int, PdfPage]]:
        for page_index in self.selected_page_indexes(pages):
            yield page_index, self.pages[page_index]


__all__ = ("DocumentSelectionMixin",)
