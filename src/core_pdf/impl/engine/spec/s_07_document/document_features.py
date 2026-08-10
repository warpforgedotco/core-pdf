# SPDX-License-Identifier: AGPL-3.0-only
"""Attachments, forms, navigation, and optional-content layers."""

from __future__ import annotations

from typing import Any, cast

from core_pdf.impl.engine.spec.s_07_document.document_lock import (
    document_cache_lock,
    document_recovery_enabled,
)
from core_pdf.impl.engine.spec.s_07_document.forms import FormsMixin
from core_pdf.impl.engine.spec.s_07_document.name_trees import iter_name_tree_items
from core_pdf.impl.engine.spec.s_07_document.navigation import NavigationMixin
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.models import RawEmbeddedFile
from core_pdf.impl.objects import PdfReference, PdfStream
from core_pdf.impl.types import PdfDict


class DocumentFeaturesMixin(NavigationMixin, FormsMixin):
    """Document features that are independent of source and page-tree loading."""

    __slots__ = ()

    embedded_files_cache: list[RawEmbeddedFile] | None
    oc_layers: dict[str, bool] | None

    def embedded_files(self: Any) -> list[RawEmbeddedFile]:
        with document_cache_lock(self):
            records = self.embedded_files_cache
            if records is None:
                records = self.build_embedded_files()
                self.embedded_files_cache = records
            return list(records)

    def build_embedded_files(self: Any) -> list[RawEmbeddedFile]:
        names = self.resolver.resolve(lookup_dict_key(self.catalog(), "Names"))
        if not isinstance(names, dict):
            return []
        embedded_tree = self.resolver.resolve(lookup_dict_key(names, "EmbeddedFiles"))
        if embedded_tree is None:
            return []
        if not isinstance(embedded_tree, dict):
            raise ValueError("invalid EmbeddedFiles name tree")

        recover = document_recovery_enabled(self)
        records: list[RawEmbeddedFile] = []
        for name, value in iter_name_tree_items(
            embedded_tree,
            self.resolver.resolve,
            self.resolver.resolve_str,
            recover=recover,
        ):
            try:
                record = self.embedded_file_record(name, value)
            except ValueError:
                if recover:
                    continue
                raise
            if record is not None:
                records.append(record)
        return records

    def embedded_file_record(self: Any, name: str, value: object) -> RawEmbeddedFile | None:
        filespec = self.resolver.resolve(value)
        if not isinstance(filespec, dict):
            raise ValueError("invalid embedded file spec")
        filespec = cast(PdfDict, filespec)
        ef = self.resolver.resolve(lookup_dict_key(filespec, "EF"))
        if not isinstance(ef, dict):
            raise ValueError("invalid embedded file stream")
        ef = cast(PdfDict, ef)
        stream = self.resolver.resolve(lookup_dict_key(ef, "UF") or lookup_dict_key(ef, "F"))
        if not isinstance(stream, PdfStream):
            raise ValueError("invalid embedded file stream")
        filename = (
            self.resolver.resolve_str(lookup_dict_key(filespec, "UF"))
            or self.resolver.resolve_str(lookup_dict_key(filespec, "F"))
            or name
        )
        return RawEmbeddedFile(name, filename, filespec, stream, stream.data)

    @staticmethod
    def ocg_key(ref: object, resolved: object) -> tuple[int, int] | int | None:
        if isinstance(ref, PdfReference):
            return (ref.object_number, ref.generation_number)
        if isinstance(resolved, dict):
            return id(resolved)
        return None

    def load_oc_layers(self: Any) -> None:
        with document_cache_lock(self):
            if self.oc_layers is None:
                self.internal_load_oc_layers()

    def internal_load_oc_layers(self: Any) -> None:
        self.oc_layers = {}
        recover = document_recovery_enabled(self)
        try:
            catalog = self.catalog()
        except ValueError:
            return
        oc = self.resolver.resolve(lookup_dict_key(catalog, "OCProperties"))
        if oc is None:
            return
        if not isinstance(oc, dict):
            if recover:
                return
            raise ValueError("invalid OCProperties dictionary")
        ocgs = self.resolver.resolve(lookup_dict_key(oc, "OCGs"))
        if ocgs is None:
            return
        if not isinstance(ocgs, list):
            if recover:
                return
            raise ValueError("invalid OCProperties OCGs array")

        on_layers: set[tuple[int, int] | int] = set()
        default_config = self.resolver.resolve(lookup_dict_key(oc, "D"))
        if default_config is not None and not isinstance(default_config, dict):
            if recover:
                default_config = None
            else:
                raise ValueError("invalid OCProperties D dictionary")
        if default_config is not None:
            base_state_value = lookup_dict_key(default_config, "BaseState")
            base_state = (
                self.resolver.resolve_name(base_state_value)
                if base_state_value is not None
                else None
            )
            if base_state_value is not None and base_state is None:
                if not recover:
                    raise ValueError("invalid OCProperties BaseState value")
            elif base_state not in (None, "ON", "OFF", "Unchanged"):
                if recover:
                    base_state = None
                else:
                    raise ValueError("invalid OCProperties BaseState value")
            if base_state != "OFF":
                for ocg in ocgs:
                    key = self.ocg_key(ocg, self.resolver.resolve(ocg))
                    if key is not None:
                        on_layers.add(key)

            on_refs = lookup_dict_key(default_config, "ON")
            if not isinstance(on_refs, list):
                on_refs = []
            for on_ref in on_refs:
                ocg_resolved = self.resolver.resolve(on_ref)
                if not isinstance(ocg_resolved, dict):
                    if recover:
                        continue
                    raise ValueError("invalid OCProperties ON entry")
                key = self.ocg_key(on_ref, ocg_resolved)
                if key is not None:
                    on_layers.add(key)

            off_refs = lookup_dict_key(default_config, "OFF")
            if not isinstance(off_refs, list):
                off_refs = []
            for off_ref in off_refs:
                ocg_resolved = self.resolver.resolve(off_ref)
                if not isinstance(ocg_resolved, dict):
                    if recover:
                        continue
                    raise ValueError("invalid OCProperties OFF entry")
                key = self.ocg_key(off_ref, ocg_resolved)
                if key is not None:
                    on_layers.discard(key)

        for ocg_ref in ocgs:
            ocg_resolved = self.resolver.resolve(ocg_ref)
            if not isinstance(ocg_resolved, dict):
                if recover:
                    continue
                raise ValueError("invalid OCProperties OCG entry")
            name = self.resolver.resolve_str(lookup_dict_key(ocg_resolved, "Name"))
            if not name:
                if recover:
                    continue
                raise ValueError("invalid OCProperties OCG name")
            key = self.ocg_key(ocg_ref, ocg_resolved)
            self.oc_layers[name] = key in on_layers if key is not None else False

    def oc_hidden_layers(self: Any) -> frozenset[str]:
        if self.oc_layers is None:
            self.load_oc_layers()
        return frozenset(name for name, on in (self.oc_layers or {}).items() if not on)


__all__ = ("DocumentFeaturesMixin",)
