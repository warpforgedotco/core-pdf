# SPDX-License-Identifier: AGPL-3.0-only
"""Transactional editing of the immutable document IR."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from core_pdf.impl.structured.model import Document, Page


class DocumentEditor:
    """Build a new document from an existing document without mutating it."""

    def __init__(self, document: Document) -> None:
        self.internal_original = document
        self.internal_pages = list(document.pages)
        self.internal_metadata = dict(document.metadata)
        self.internal_committed = False
        self.internal_rolled_back = False

    @property
    def original(self) -> Document:
        return self.internal_original

    def set_metadata(self, key: str, value: Any) -> DocumentEditor:
        self.internal_ensure_active()
        self.internal_metadata[key] = value
        return self

    def update_metadata(self, values: Mapping[str, Any]) -> DocumentEditor:
        self.internal_ensure_active()
        self.internal_metadata.update(values)
        return self

    def replace_page(self, page_number: int, page: Page) -> DocumentEditor:
        self.internal_ensure_active()
        index = self.internal_page_index(page_number)
        self.internal_pages[index] = page
        return self

    def insert_page(self, position: int, page: Page) -> DocumentEditor:
        self.internal_ensure_active()
        if position < 1 or position > len(self.internal_pages) + 1:
            raise IndexError(f"page insertion position out of range: {position}")
        self.internal_pages.insert(position - 1, page)
        return self

    def delete_page(self, page_number: int) -> DocumentEditor:
        self.internal_ensure_active()
        del self.internal_pages[self.internal_page_index(page_number)]
        return self

    def commit(self) -> Document:
        self.internal_ensure_active()
        self.internal_committed = True
        return Document(
            pages=tuple(
                replace(page, page_number=index + 1)
                for index, page in enumerate(self.internal_pages)
            ),
            metadata=self.internal_metadata,
            diagnostics=self.internal_original.diagnostics,
            schema_version=self.internal_original.schema_version,
        )

    def rollback(self) -> None:
        self.internal_ensure_active()
        self.internal_rolled_back = True
        self.internal_pages.clear()
        self.internal_metadata.clear()

    def internal_page_index(self, page_number: int) -> int:
        if page_number < 1 or page_number > len(self.internal_pages):
            raise IndexError(f"page number out of range: {page_number}")
        return page_number - 1

    def internal_ensure_active(self) -> None:
        if self.internal_committed:
            raise RuntimeError("document editor has already been committed")
        if self.internal_rolled_back:
            raise RuntimeError("document editor has already been rolled back")


__all__ = ("DocumentEditor",)
