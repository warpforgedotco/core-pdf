from __future__ import annotations

from typing import Protocol

from core_pdf.impl.engine.spec.s_07_document.models import (
    FieldRecord,
    NamedDestination,
    OutlineItem,
)
from core_pdf.impl.engine.spec.s_07_objects.resolver import ObjectResolver
from core_pdf.impl.engine.spec.s_07_syntax.primitives import PdfDictLike, PdfObject


class DocumentMixinProtocol(Protocol):
    resolver: ObjectResolver
    acroform_cache: PdfDictLike | None
    fields_cache: list[FieldRecord] | None
    oc_layers: dict[str, bool] | None
    named_destinations_cache: dict[str, NamedDestination] | None

    @property
    def acroform(self) -> PdfDictLike | None: ...

    def catalog(self) -> PdfDictLike: ...

    def collect_field_records(
        self,
        node: PdfObject,
        inherited_name: str = "",
        inherited_type: str = "",
        inherited_value: PdfObject = None,
        _depth: int = 0,
    ) -> list[FieldRecord]: ...

    def destination_from_list(self, resolved_list: list[PdfObject]) -> NamedDestination: ...

    def load_oc_layers(self) -> None: ...

    def normalize_destination_value(
        self,
        val: PdfObject,
        seen: set[str] | None = None,
        targets: dict[str, PdfObject] | None = None,
        normalized: dict[str, NamedDestination] | None = None,
        resolving: set[str] | None = None,
    ) -> NamedDestination: ...

    def page_index_for(self, page_obj: PdfObject) -> int | None: ...

    def populate_named_destinations(self) -> None: ...

    def resolve_destination(self, dest: PdfObject, seen: set[str] | None = None) -> int | None: ...

    def resolve_named_destination(
        self, name: str, seen: set[str] | None = None
    ) -> NamedDestination | None: ...

    def walk_name_tree(
        self, node: PdfObject, results: dict[str, PdfObject], _depth: int = 0
    ) -> None: ...

    def walk_outlines(self, item: PdfDictLike, level: int) -> list[OutlineItem]: ...
