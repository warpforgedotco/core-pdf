# SPDX-License-Identifier: AGPL-3.0-only
"""PDF writing primitives."""

from core_pdf.impl.engine.writing.api import PdfDocumentWritingMixin
from core_pdf.impl.engine.writing.document import serialize_pdf_file
from core_pdf.impl.engine.writing.incremental import append_incremental_update
from core_pdf.impl.engine.writing.objects import serialize_pdf_object

__all__ = (
    "PdfDocumentWritingMixin",
    "append_incremental_update",
    "serialize_pdf_file",
    "serialize_pdf_object",
)
