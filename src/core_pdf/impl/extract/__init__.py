# SPDX-License-Identifier: AGPL-3.0-only
"""Canonical page and document extraction entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core_pdf.impl.extract.pipeline import (
    extract_document,
    extract_page,
)

if TYPE_CHECKING:
    from core_pdf.impl.extract.ocr.tesseract import prewarm_runtime


def __getattr__(name: str) -> Any:
    """Resolve ``prewarm_runtime`` lazily.

    Importing ``extract.ocr.tesseract`` binds OCR support modules, which a
    native-text document never touches. Keeping the re-export lazy leaves that
    cost on the OCR path where it belongs.
    """
    if name == "prewarm_runtime":
        from core_pdf.impl.extract.ocr.tesseract import prewarm_runtime

        return prewarm_runtime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "extract_document",
    "extract_page",
    "prewarm_runtime",
)
