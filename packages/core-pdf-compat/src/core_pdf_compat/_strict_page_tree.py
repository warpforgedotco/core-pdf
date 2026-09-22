from __future__ import annotations

from core_pdf import PdfDocument
from core_pdf.impl.document.recovery.lexer import PdfLexer
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.types import PdfReference


def has_malformed_shadowed_definition(
    pdf: PdfDocument,
    reference: PdfReference,
) -> bool:
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
            except PdfParseError, TypeError, ValueError:
                return True
            position = data.find(header, position + len(header))
    finally:
        lexer.close()
    return False


__all__ = ("has_malformed_shadowed_definition",)
