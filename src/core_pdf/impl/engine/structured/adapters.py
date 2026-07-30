# SPDX-License-Identifier: AGPL-3.0-only
"""Extension points for document-level enrichment and transformation."""

from __future__ import annotations

from typing import Protocol

from core_pdf.impl.engine.structured.model import Document


class DocumentAdapter(Protocol):
    """An optional, immutable transformation applied after extraction."""

    def apply(self, document: Document) -> Document:
        """Return an enriched or transformed document."""


__all__ = ("DocumentAdapter",)
