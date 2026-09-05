# SPDX-License-Identifier: AGPL-3.0-only
"""Shared selected-page metadata and native document extraction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Protocol, TypeVar

from core_pdf.impl._impl.extract.pipeline import internal_PageExtraction
from core_pdf.impl._impl.output.model import SCHEMA_VERSION, Document, Page
from core_pdf.impl._impl.runtime.execution import ExtractionScope

if TYPE_CHECKING:
    from core_pdf.impl.spec.s_07_document.document import PdfDocument
    from core_pdf.impl.spec.s_07_document.page import PdfPage
    from core_pdf.impl.spec.s_07_document.records import RawFormField
    from core_pdf.impl.spec.s_14_structure.tree import PageStructure

internal_Extraction = TypeVar("internal_Extraction", bound=internal_PageExtraction, covariant=True)


class internal_ExtractionBuilder(Protocol[internal_Extraction]):
    def __call__(
        self,
        page: PdfPage,
        *,
        fields: Iterable[RawFormField],
        structure: PageStructure | None,
        hidden_layers: frozenset[str],
    ) -> internal_Extraction: ...


def internal_prepare_document_pages(
    document: PdfDocument,
    pages: Sequence[PdfPage],
    build: internal_ExtractionBuilder[internal_Extraction],
) -> tuple[internal_Extraction, ...]:
    """Collect one selection's metadata and construct its independent page pipelines."""
    hidden_layers = document.oc_hidden_layers() if pages else frozenset()
    structure_tree = None
    with suppress(IndexError, TypeError, ValueError):
        structure_tree = document.structure

    def page_structure(page: PdfPage) -> PageStructure | None:
        if structure_tree is None:
            return None
        try:
            return structure_tree.page_structure(page)
        except (IndexError, TypeError, ValueError):
            return None

    fields_by_page: dict[int, list[RawFormField]] = {}
    with suppress(TypeError, ValueError):
        fields_by_page = document.fields_by_page(pages)
    return tuple(
        build(
            page,
            fields=fields_by_page.get(int(page.page_number) - 1, ()),
            structure=page_structure(page),
            hidden_layers=hidden_layers,
        )
        for page in pages
    )


def internal_assemble_document_pages(
    extractions: tuple[internal_PageExtraction, ...],
    context: ExtractionScope,
) -> tuple[Page, ...]:
    pages: list[Page] = []
    for extraction in extractions:
        context.raise_if_cancelled()
        pages.append(extraction.assembled_page(context))
    return tuple(pages)


def internal_assemble_document(
    document: PdfDocument,
    extractions: tuple[internal_PageExtraction, ...],
    context: ExtractionScope,
) -> Document:
    assembled_pages = internal_assemble_document_pages(extractions, context)
    diagnostics = tuple(diagnostic for page in assembled_pages for diagnostic in page.diagnostics)
    metadata = document.get_metadata()
    return Document(
        pages=assembled_pages,
        metadata=metadata,
        diagnostics=diagnostics,
        schema_version=SCHEMA_VERSION,
    )


def extract_document(
    document: PdfDocument, context: ExtractionScope, pages: Sequence[PdfPage]
) -> Document:
    """Extract exactly the requested pages from native PDF content."""
    extractions = internal_prepare_document_pages(document, tuple(pages), internal_PageExtraction)
    return internal_assemble_document(document, extractions, context)
