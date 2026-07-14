# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Iterator, Protocol

from core_pdf.impl.models import FieldRecord, NamedDestination, OutlineItem
from core_pdf.impl.types import PdfArray, PdfDict


class FormsResolver(Protocol):
    def resolve(self, ref: object) -> object: ...

    def resolve_box(self, value: object) -> tuple[float, float, float, float] | None: ...

    def resolve_name(self, value: object) -> str | None: ...

    def resolve_name_like_value(self, resolved: object) -> str | None: ...

    def resolve_str(self, value: object) -> str | None: ...


class FormsDocumentProtocol(Protocol):
    resolver: FormsResolver
    acroform_cache: PdfDict | None
    fields_cache: list[FieldRecord] | None
    xref_was_recovered: bool
    page_tree_was_recovered: bool

    def catalog(self) -> PdfDict: ...

    def iter_page_dicts(self) -> Iterator[PdfDict]: ...

    @property
    def acroform(self) -> PdfDict | None: ...

    def collect_field_records(
        self,
        node: object,
        inherited_name: str = "",
        inherited_type: str = "",
        inherited_value: object = None,
        depth: int = 0,
        seen: set[int] | None = None,
    ) -> list[FieldRecord]: ...

    def discover_widget_field_records(self, existing: list[FieldRecord]) -> list[FieldRecord]: ...

    def fields(self) -> list[FieldRecord]: ...


class NavigationResolver(Protocol):
    def resolve(self, ref: object) -> object: ...

    def resolve_int(self, value: object, default: int | None = None) -> int | None: ...

    def resolve_name(self, value: object) -> str | None: ...

    def resolve_name_like_value(self, resolved: object) -> str | None: ...

    def resolve_str(self, value: object) -> str | None: ...


class NavigationDocumentProtocol(Protocol):
    resolver: NavigationResolver
    named_destinations_cache: dict[str, NamedDestination] | None
    xref_was_recovered: bool
    page_tree_was_recovered: bool

    def catalog(self) -> PdfDict: ...

    def page_index_for(self, page_obj: object) -> int | None: ...

    def extract_outline_count(self, current: PdfDict) -> int: ...

    def validate_outline_count(self, value: object) -> int: ...

    def walk_outlines(self, item: object, level: int) -> list[OutlineItem]: ...

    def resolve_destination(self, dest: object, seen: set[str] | None = None) -> int | None: ...

    def populate_named_destinations(self) -> None: ...

    def resolve_named_destination(
        self, name: str, seen: set[str] | None = None
    ) -> NamedDestination | None: ...

    def normalize_destination_value(
        self,
        val: object,
        seen: set[str] | None = None,
        targets: dict[str, object] | None = None,
        normalized: dict[str, NamedDestination] | None = None,
        resolving: set[str] | None = None,
    ) -> NamedDestination: ...

    def destination_from_list(self, resolved_list: PdfArray) -> NamedDestination: ...


class LayersResolver(Protocol):
    def resolve(self, ref: object) -> object: ...

    def resolve_name(self, value: object) -> str | None: ...

    def resolve_str(self, value: object) -> str | None: ...


class LayersDocumentProtocol(Protocol):
    resolver: LayersResolver
    oc_layers: dict[str, bool] | None
    xref_was_recovered: bool
    page_tree_was_recovered: bool

    def catalog(self) -> PdfDict: ...

    def ocg_key(self, ref: object, resolved: object) -> tuple[int, int] | int | None: ...

    def load_oc_layers(self) -> None: ...

    def oc_hidden_layers(self) -> frozenset[str]: ...
