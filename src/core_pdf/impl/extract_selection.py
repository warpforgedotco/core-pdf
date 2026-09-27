# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Protocol, TypeVar

from core_pdf.impl.document_metadata import plain_pdf_value
from core_pdf.impl.execution import ExtractionScope
from core_pdf.impl.extract_pipeline import PageExtraction
from core_pdf.impl.output_model import SCHEMA_VERSION, Document, Page

if TYPE_CHECKING:
    from core_pdf.impl.document_contracts import ExtractionDocument
    from core_pdf.impl.document_page import PdfPage
    from core_pdf.impl.document_records import RawFormField
    from core_pdf.impl.document_structure import PageStructure

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
    document: ExtractionDocument,
    pages: Sequence[PdfPage],
    build: ExtractionBuilder[Extraction],
) -> tuple[Extraction, ...]:
    return tuple(iter_document_pages(document, pages, build))


def iter_document_pages[Extraction: PageExtraction](
    document: ExtractionDocument,
    pages: Sequence[PdfPage],
    build: ExtractionBuilder[Extraction],
) -> Iterator[Extraction]:
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
    for page in pages:
        yield build(
            page,
            fields=fields_by_page.get(int(page.page_number) - 1, ()),
            structure=page_structure(page),
            hidden_layers=hidden_layers,
        )


def assemble_document(
    document: ExtractionDocument,
    extractions: Iterable[PageExtraction],
    context: ExtractionScope,
) -> Document:
    pages: list[Page] = []
    for extraction in extractions:
        context.raise_if_cancelled()
        pages.append(extraction.assembled_page(context))
    assembled_pages = tuple(pages)
    diagnostics = tuple(diagnostic for page in assembled_pages for diagnostic in page.diagnostics)
    metadata = {key: plain_pdf_value(value) for key, value in document.get_metadata().items()}
    return Document(
        pages=assembled_pages,
        metadata=metadata,
        diagnostics=diagnostics,
        schema_version=SCHEMA_VERSION,
    )


def extract_document(
    document: ExtractionDocument, context: ExtractionScope, pages: Sequence[PdfPage]
) -> Document:
    extractions = iter_document_pages(document, tuple(pages), PageExtraction)
    return assemble_document(document, extractions, context)
