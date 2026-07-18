# SPDX-License-Identifier: AGPL-3.0-only
"""PDF writing primitives."""

from core_pdf.impl.engine.writing.document import serialize_pdf_file
from core_pdf.impl.engine.writing.objects import serialize_pdf_object

__all__ = ("serialize_pdf_file", "serialize_pdf_object")
