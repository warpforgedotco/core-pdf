# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from core_pdf.impl.pdf_values import ReferenceResolver

if TYPE_CHECKING:
    from core_pdf.impl.capture_recording import TextState
    from core_pdf.impl.document_document import DocumentOperation
    from core_pdf.impl.document_metadata import MetadataRecord
    from core_pdf.impl.document_page import PdfPage
    from core_pdf.impl.document_records import RawFormField
    from core_pdf.impl.document_structure import StructureTree
    from core_pdf.impl.recovery_policy import Recovery
    from core_pdf.impl.recovery_resolver import ObjectResolver
    from core_pdf_spec.s_07_syntax.types import PdfDict


class ResolverHost(Protocol):
    @property
    def resolver(self) -> ObjectResolver: ...


class CaptureDocument(ResolverHost, ReferenceResolver, Protocol):
    pass


class LayeredDocument(CaptureDocument, Protocol):
    def oc_hidden_layers(self) -> frozenset[str]: ...


class CapturePage(Protocol):
    @property
    def document(self) -> LayeredDocument: ...

    @property
    def resources(self) -> PdfDict: ...

    def annotation_dicts(self) -> list[PdfDict]: ...

    def get_fields(self) -> list[RawFormField]: ...

    def effective_page_clip(self) -> tuple[float, float, float, float] | None: ...

    def consume_contents(self, state: TextState) -> None: ...


class StructureHost(ResolverHost, Protocol):
    @property
    def recovery(self) -> Recovery: ...

    @property
    def structure(self) -> StructureTree | None: ...

    @property
    def pages(self) -> Sequence[PdfPage]: ...

    def page_index_for(self, page_obj: object) -> int | None: ...


class PageHost(LayeredDocument, StructureHost, Protocol):
    @property
    def recovery_enabled(self) -> bool: ...

    def cached_fields_by_page(self) -> dict[int, list[RawFormField]]: ...

    def page_label(self, page_index: int) -> str | None: ...

    def acquire_operation(self) -> DocumentOperation: ...


class ExtractionDocument(LayeredDocument, StructureHost, Protocol):
    def fields_by_page(
        self, pages: Sequence[Any] | None = None
    ) -> dict[int, list[RawFormField]]: ...

    def get_metadata(self) -> MetadataRecord: ...


def resolve_optional_dict(
    resolver: ReferenceResolver, value: object, message: str
) -> PdfDict | None:
    resolved = resolver.resolve(value)
    if resolved is None or isinstance(resolved, dict):
        return resolved
    raise ValueError(message)


__all__ = (
    "CaptureDocument",
    "CapturePage",
    "ExtractionDocument",
    "LayeredDocument",
    "PageHost",
    "ResolverHost",
    "StructureHost",
    "resolve_optional_dict",
)
