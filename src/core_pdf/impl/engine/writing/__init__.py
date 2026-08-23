# SPDX-License-Identifier: AGPL-3.0-only
"""PDF writing primitives."""

from core_pdf.impl.engine.writing.document import (
    serialize_encrypted_pdf_file,
    serialize_pdf_file,
)
from core_pdf.impl.engine.writing.encryption import (
    StandardPdfEncryption,
    StandardPdfEncryptionContext,
)
from core_pdf.impl.engine.writing.fonts import (
    PdfFontProvider,
    PdfFontResource,
    StandardType1FontProvider,
    TrueTypeFontProvider,
)
from core_pdf.impl.engine.writing.incremental import append_incremental_update
from core_pdf.impl.engine.writing.object_graph import PdfObjectGraph
from core_pdf.impl.engine.writing.objects import serialize_pdf_object
from core_pdf.impl.engine.writing.semantic import serialize_document_to_pdf
from core_pdf.impl.engine.writing.signatures import (
    PdfSignaturePlan,
    PdfSignatureProvider,
    apply_signature_plan,
)

__all__ = (
    "PdfObjectGraph",
    "PdfFontProvider",
    "PdfFontResource",
    "StandardType1FontProvider",
    "TrueTypeFontProvider",
    "StandardPdfEncryption",
    "StandardPdfEncryptionContext",
    "PdfSignaturePlan",
    "PdfSignatureProvider",
    "apply_signature_plan",
    "append_incremental_update",
    "serialize_pdf_file",
    "serialize_encrypted_pdf_file",
    "serialize_pdf_object",
    "serialize_document_to_pdf",
)
