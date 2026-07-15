# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Protocol, cast

from core_pdf.impl.engine.spec.s_07_document.name_trees import iter_name_tree_items
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.models import EmbeddedFileRecord
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.types import PdfDict


class EmbeddedResolver(Protocol):
    def resolve(self, ref: object) -> object: ...

    def resolve_str(self, value: object) -> str | None: ...


class DocumentEmbeddedHost(Protocol):
    embedded_files_cache: list[EmbeddedFileRecord] | None
    resolver: EmbeddedResolver
    xref_was_recovered: bool
    page_tree_was_recovered: bool

    def catalog(self) -> PdfDict: ...

    def build_embedded_files(self) -> list[EmbeddedFileRecord]: ...

    def embedded_file_record(self, name: str, value: object) -> EmbeddedFileRecord | None: ...


class DocumentEmbeddedMixin:
    embedded_files_cache: list[EmbeddedFileRecord] | None

    def embedded_files(self: DocumentEmbeddedHost) -> list[EmbeddedFileRecord]:
        records = self.embedded_files_cache
        if records is None:
            records = self.build_embedded_files()
            self.embedded_files_cache = records
        return list(records)

    def build_embedded_files(self: DocumentEmbeddedHost) -> list[EmbeddedFileRecord]:
        names = self.resolver.resolve(lookup_dict_key(self.catalog(), "Names"))
        if not isinstance(names, dict):
            return []
        embedded_tree = self.resolver.resolve(lookup_dict_key(names, "EmbeddedFiles"))
        if embedded_tree is None:
            return []
        if not isinstance(embedded_tree, dict):
            raise ValueError("invalid EmbeddedFiles name tree")

        recover = self.xref_was_recovered or self.page_tree_was_recovered
        records: list[EmbeddedFileRecord] = []
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

    def embedded_file_record(
        self: DocumentEmbeddedHost, name: str, value: object
    ) -> EmbeddedFileRecord | None:
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
        return EmbeddedFileRecord(name, filename, filespec, stream, stream.data)


__all__ = ("DocumentEmbeddedMixin",)
