"""Shared strict-reader validation for repaired page-tree roots."""

from __future__ import annotations

from core_pdf import PdfDocument
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.types import PdfReference


def internal_has_malformed_shadowed_definition(
    pdf: PdfDocument,
    reference: PdfReference,
) -> bool:
    """Return whether recovery retained a stale object over a malformed revision.

    The native engine intentionally salvages the last parseable definition when a
    damaged xref cannot be followed.  Strict third-party readers instead reject a
    file when a later definition of a structural object is malformed.  Merely
    seeing the object header again is insufficient: object-number text can occur
    inside streams, and valid incremental revisions can redefine objects.
    """
    if not pdf.xref_was_recovered or pdf.strict_xref_validation_error() is None:
        return False
    entry = pdf.xref.get((reference.object_number << 16) | reference.generation_number)
    if entry is None or entry.object_stream is not None:
        return False
    data = bytes(pdf.raw_data)
    header = f"{reference.object_number} {reference.generation_number} obj".encode()
    position = data.find(header, entry.offset + len(header))
    lexer = PdfLexer(
        pdf.raw_data,
        reference_resolver=pdf.resolver.resolve,
        decipher=pdf.decipher,
    )
    try:
        while position >= 0:
            lexer.rewind(position)
            try:
                lexer.parse_indirect_object()
            except (PdfParseError, TypeError, ValueError):
                return True
            position = data.find(header, position + len(header))
    finally:
        lexer.close()
    return False


__all__ = ("internal_has_malformed_shadowed_definition",)
