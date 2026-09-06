"""Resolve the PDF-defined document information dictionary and metadata stream."""

from __future__ import annotations

from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfValueResolver


def info_dictionary(resolver: PdfValueResolver, trailer: PdfDict) -> PdfDict | None:
    reference = trailer.get("Info")
    if reference is None:
        return None
    info = resolver.resolve_dict(reference)
    if not isinstance(info, dict):
        raise ValueError("invalid trailer Info dictionary")
    return info


def metadata_stream(resolver: PdfValueResolver, trailer: PdfDict) -> PdfStream | None:
    root = resolver.resolve_dict(trailer.get("Root"))
    if root is None:
        return None
    if not isinstance(root, dict):
        raise ValueError("invalid trailer Root dictionary")
    metadata = resolver.resolve(root.get("Metadata"))
    if metadata is not None and not isinstance(metadata, PdfStream):
        raise ValueError("invalid Metadata stream")
    return metadata
