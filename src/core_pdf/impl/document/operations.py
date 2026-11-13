# SPDX-License-Identifier: AGPL-3.0-only

"""The guard every document operation runs under, so close waits for it."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core_pdf.impl.document.pdf_document import PdfDocument


class DocumentOperation(AbstractContextManager["DocumentOperation"]):
    __slots__ = ("document", "released")

    def __init__(self, document: PdfDocument[Any]) -> None:
        self.document = document
        self.released = False

    @property
    def cancelled(self) -> bool:
        return self.document.operation_cancelled.is_set()

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.document.release_operation()

    def __exit__(self, *args: object) -> None:
        self.release()
