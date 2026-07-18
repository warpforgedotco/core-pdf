# SPDX-License-Identifier: AGPL-3.0-only
"""Transactional editing of the immutable document IR."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from core_document.model import Document, Page


class DocumentEditor:
    """Build a new document from an existing document without mutating it."""

    def __init__(self, document: Document) -> None:
        self._original = document
        self._pages = list(document.pages)
        self._metadata = dict(document.metadata)
        self._committed = False
        self._rolled_back = False

    @property
    def original(self) -> Document:
        return self._original

    def set_metadata(self, key: str, value: Any) -> DocumentEditor:
        self._ensure_active()
        self._metadata[key] = value
        return self

    def update_metadata(self, values: Mapping[str, Any]) -> DocumentEditor:
        self._ensure_active()
        self._metadata.update(values)
        return self

    def replace_page(self, page_number: int, page: Page) -> DocumentEditor:
        self._ensure_active()
        index = self._page_index(page_number)
        self._pages[index] = page
        return self

    def insert_page(self, position: int, page: Page) -> DocumentEditor:
        self._ensure_active()
        if position < 1 or position > len(self._pages) + 1:
            raise IndexError(f"page insertion position out of range: {position}")
        self._pages.insert(position - 1, page)
        return self

    def delete_page(self, page_number: int) -> DocumentEditor:
        self._ensure_active()
        del self._pages[self._page_index(page_number)]
        return self

    def commit(self) -> Document:
        self._ensure_active()
        self._committed = True
        return Document(
            pages=tuple(
                replace(page, page_number=index + 1) for index, page in enumerate(self._pages)
            ),
            metadata=self._metadata,
            diagnostics=self._original.diagnostics,
            schema_version=self._original.schema_version,
        )

    def rollback(self) -> None:
        self._ensure_active()
        self._rolled_back = True
        self._pages.clear()
        self._metadata.clear()

    def _page_index(self, page_number: int) -> int:
        if page_number < 1 or page_number > len(self._pages):
            raise IndexError(f"page number out of range: {page_number}")
        return page_number - 1

    def _ensure_active(self) -> None:
        if self._committed:
            raise RuntimeError("document editor has already been committed")
        if self._rolled_back:
            raise RuntimeError("document editor has already been rolled back")


__all__ = ("DocumentEditor",)
