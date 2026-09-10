"""Resolve the PDF-defined document information dictionary and metadata stream."""

from __future__ import annotations

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfValueResolver


def info_dictionary(resolver: PdfValueResolver, trailer: PdfDict) -> PdfDict | None:
    reference = trailer.get("Info")
    if reference is None:
        return None
    info = resolver.resolve_dict(reference)
    if not isinstance(info, dict):
        raise ValueError("invalid trailer Info dictionary")
    return info


def metadata_stream(resolver: PdfValueResolver, trailer: PdfDict) -> PdfStream | None:
    root = resolver.deep_resolve(trailer.get("Root"))
    if root is None:
        return None
    return catalog_metadata_stream(resolver, root)


def catalog_metadata_stream(resolver: PdfValueResolver, catalog: object) -> PdfStream | None:
    """Resolve optional Metadata from an already resolved catalog dictionary."""
    if not isinstance(catalog, dict):
        raise ValueError("invalid trailer Root dictionary")
    metadata = resolver.resolve(catalog.get("Metadata"))
    if metadata is not None and not isinstance(metadata, PdfStream):
        raise ValueError("invalid Metadata stream")
    return metadata


__all__ = (
    "catalog_metadata_stream",
    "info_dictionary",
    "metadata_stream",
)
