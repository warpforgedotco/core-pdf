# SPDX-License-Identifier: AGPL-3.0-only
"""Capture, recognize, and emit PDF content.

The package exports the pipeline entry points and the shared stage models;
stage internals live in (and are imported from) the owning submodule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core_pdf.impl.engine.parse.model import (
    CapturedPage,
    FusionPolicy,
    GlyphEvidence,
    ObservationBatch,
    ObservationSource,
    OcrPass,
    OcrPassScope,
    PageEvidence,
    PagePlanReason,
    PageRoute,
    ParsedBlock,
    ParsedLine,
    ParsedPage,
    ParseReport,
    ReadingOrderEvidence,
    RecognitionReport,
    RecognitionResult,
    TextQualityStats,
    WorkPlan,
)
from core_pdf.impl.engine.parse.ocr_bootstrap import internal_prepare_ocr_signals
from core_pdf.impl.engine.parse.pipeline import (
    extract_page,
    page_extraction,
    parse_document,
    parse_page,
)

# Claim the main thread's signal handlers now, while we are certainly on it.
# parse.ocr_tesseract itself stays unimported until a page actually needs recognition.
internal_prepare_ocr_signals()

if TYPE_CHECKING:
    from core_pdf.impl.engine.parse.ocr_tesseract import prewarm_runtime


def __getattr__(name: str) -> Any:
    """Resolve ``prewarm_runtime`` lazily.

    Importing ``parse.ocr_tesseract`` binds tesserocr and PIL, which a
    native-text document never touches. Keeping the re-export lazy leaves that
    cost on the OCR path where it belongs.
    """
    if name == "prewarm_runtime":
        from core_pdf.impl.engine.parse.ocr_tesseract import prewarm_runtime

        return prewarm_runtime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "CapturedPage",
    "FusionPolicy",
    "GlyphEvidence",
    "ObservationBatch",
    "ObservationSource",
    "OcrPass",
    "OcrPassScope",
    "PageEvidence",
    "PagePlanReason",
    "PageRoute",
    "ParseReport",
    "ParsedBlock",
    "ParsedLine",
    "ParsedPage",
    "ReadingOrderEvidence",
    "RecognitionReport",
    "RecognitionResult",
    "TextQualityStats",
    "WorkPlan",
    "extract_page",
    "page_extraction",
    "parse_document",
    "parse_page",
    "prewarm_runtime",
)
