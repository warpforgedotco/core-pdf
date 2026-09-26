from __future__ import annotations

from core_pdf import PdfDocument
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.types import PdfReference
from core_pdf_spec.s_07_syntax.xref import key_for

from ._shared import parse_indirect_object_at


def has_malformed_shadowed_definition(
    pdf: PdfDocument,
    reference: PdfReference,
) -> bool:
    if not pdf.xref_was_recovered or pdf.strict_xref_validation_error() is None:
        return False
    entry = pdf.xref.get(key_for(reference.object_number, reference.generation_number))
    if entry is None or entry.object_stream is not None:
        return False
    data = pdf.raw_data
    header = f"{reference.object_number} {reference.generation_number} obj".encode()
    position = data.find(header, entry.offset + len(header))
    while position >= 0:
        try:
            parse_indirect_object_at(
                pdf.raw_data,
                position,
                reference_resolver=pdf.resolver.resolve,
                decipher=pdf.decipher,
            )
        except PdfParseError, TypeError, ValueError:
            return True
        position = data.find(header, position + len(header))
    return False


__all__ = ("has_malformed_shadowed_definition",)
