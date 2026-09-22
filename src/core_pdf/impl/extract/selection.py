# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from core_pdf.impl.extract.pipeline import PageExtraction
from core_pdf.impl.output.model import SCHEMA_VERSION, Document, Page
from core_pdf.impl.runtime.execution import ExtractionScope

if TYPE_CHECKING:
    from core_pdf.impl.document.document import PdfDocument
    from core_pdf.impl.document.page import PdfPage
    from core_pdf.impl.document.records import RawFormField
    from core_pdf.impl.document.structure import PageStructure

Extraction = TypeVar("Extraction", bound=PageExtraction, covariant=True)


class ExtractionBuilder(Protocol[Extraction]):
    def __call__(
        self,
        page: PdfPage,
        *,
        fields: Iterable[RawFormField],
        structure: PageStructure | None,
        hidden_layers: frozenset[str],
    ) -> Extraction: ...


def prepare_document_pages[Extraction: PageExtraction](
    document: PdfDocument[Any],
    pages: Sequence[PdfPage],
    build: ExtractionBuilder[Extraction],
) -> tuple[Extraction, ...]:
    hidden_layers = document.oc_hidden_layers() if pages else frozenset()
    structure_tree = None
    with suppress(IndexError, TypeError, ValueError):
        structure_tree = document.structure

    def page_structure(page: PdfPage) -> PageStructure | None:
        if structure_tree is None:
            return None
        try:
            return structure_tree.page_structure(page)
        except IndexError, TypeError, ValueError:
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


def assemble_document(
    document: PdfDocument[Any],
    extractions: tuple[PageExtraction, ...],
    context: ExtractionScope,
) -> Document:
    pages: list[Page] = []
    for extraction in extractions:
        context.raise_if_cancelled()
        pages.append(extraction.assembled_page(context))
    assembled_pages = tuple(pages)
    diagnostics = tuple(diagnostic for page in assembled_pages for diagnostic in page.diagnostics)
    metadata = document.get_metadata()
    return Document(
        pages=assembled_pages,
        metadata=metadata,
        diagnostics=diagnostics,
        schema_version=SCHEMA_VERSION,
    )


def extract_document(
    document: PdfDocument[Any], context: ExtractionScope, pages: Sequence[PdfPage]
) -> Document:
    extractions = prepare_document_pages(document, tuple(pages), PageExtraction)
    return assemble_document(document, extractions, context)
