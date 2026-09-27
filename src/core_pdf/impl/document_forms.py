# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence

from core_pdf.impl.document_contracts import DocumentState, resolve_optional_dict
from core_pdf.impl.document_fields import collect_field_records
from core_pdf.impl.document_page import PdfPage
from core_pdf.impl.document_records import RawEmbeddedFile, RawFormField
from core_pdf.impl.recovery_trees import iter_name_tree_items
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict


class DocumentForms[PageT: PdfPage](DocumentState[PageT]):
    __slots__ = ()

    @property
    def acroform(self) -> PdfDict | None:
        return self.catalog_dict("AcroForm", recoverable=True)

    def fields(self) -> list[RawFormField]:
        af = self.acroform
        records: list[RawFormField] = []
        if af is not None:
            field_list = af.get("Fields")
            if field_list is None:
                field_list = []
            elif not isinstance(field_list, list):
                self.malformed("invalid AcroForm Fields array")
                field_list = []
            for field in field_list:
                field_obj = self.resolver.resolve(field)
                records.extend(
                    collect_field_records(self.resolver, field_obj, self.recovery.malformed)
                )
        if not records or self.recovery_enabled:
            records.extend(self.discover_widget_field_records(records))
        return records

    def fields_by_page(
        self,
        pages: Sequence[PageT] | None = None,
    ) -> dict[int, list[RawFormField]]:
        if pages is not None:
            return self.group_fields_by_page(tuple(pages))
        return {
            page_index: list(fields) for page_index, fields in self.cached_fields_by_page().items()
        }

    def cached_fields_by_page(self) -> dict[int, list[RawFormField]]:
        return self.caches.get("fields_by_page", self.group_all_fields_by_page)

    def group_all_fields_by_page(self) -> dict[int, list[RawFormField]]:
        return self.group_fields_by_page(self.pages)

    def group_fields_by_page(
        self, page_sequence: tuple[PageT, ...]
    ) -> dict[int, list[RawFormField]]:
        page_indexes_by_dict = {
            id(page.page_dict): page.page_number - 1
            for page in page_sequence
            if isinstance(page, PdfPage)
        }
        grouped: dict[int, list[RawFormField]] = {}
        annot_page_index: dict[int, int] | None = None

        def widget_page_index(widget: object) -> int | None:
            nonlocal annot_page_index
            pg_ref = widget.get("P") if isinstance(widget, dict) else None
            if pg_ref is not None:
                pg_obj = self.resolver.resolve(pg_ref)
                return page_indexes_by_dict.get(id(pg_obj)) if isinstance(pg_obj, dict) else None
            if annot_page_index is None:
                annot_page_index = {
                    id(annot): page.page_number - 1
                    for page in page_sequence
                    if isinstance(page, PdfPage)
                    for annot in page.annotation_dicts()
                }
            return annot_page_index.get(id(widget))

        for field in self.fields():
            page_indexes: set[int] = set()
            if field.widget:
                if not isinstance(field.widget, dict):
                    raise ValueError("invalid field widget entry")
                page_index = widget_page_index(field.widget)
                if page_index is not None:
                    page_indexes.add(page_index)
            elif field.kids:
                if not isinstance(field.kids, list):
                    raise ValueError("invalid field kids array")
                for kid_ref in field.kids:
                    kid = self.resolver.as_dict(kid_ref)
                    if kid is not None and self.resolver.name_at(kid, "Subtype") == "Widget":
                        page_index = widget_page_index(kid)
                        if page_index is not None:
                            page_indexes.add(page_index)
            for page_index in page_indexes:
                grouped.setdefault(page_index, []).append(field)
        return grouped

    def discover_widget_field_records(self, existing: list[RawFormField]) -> list[RawFormField]:
        seen_widgets = {id(record.widget) for record in existing if isinstance(record.widget, dict)}
        records: list[RawFormField] = []
        for page in self.pages:
            for annot in page.annotation_dicts():
                if id(annot) in seen_widgets:
                    continue
                subtype = self.resolver.resolve_name_or_text(annot.get("Subtype")) or ""
                if subtype != "Widget":
                    continue
                root = self.widget_field_root(annot)
                if id(root) in seen_widgets:
                    continue
                seen_widgets.add(id(root))
                seen_widgets.add(id(annot))
                records.extend(collect_field_records(self.resolver, root, self.recovery.malformed))
        return records

    def widget_field_root(self, annot: PdfDict) -> PdfDict:
        node = annot
        seen = {id(node)}
        for _ in range(50):
            parent = self.resolver.dict_at(node, "Parent")
            if parent is None or id(parent) in seen:
                break
            seen.add(id(parent))
            node = parent
        return node

    def embedded_files(self) -> list[RawEmbeddedFile]:
        names = self.resolver.dict_at(self.catalog(), "Names")
        if names is None:
            return []
        embedded_tree = resolve_optional_dict(
            self.resolver, names.get("EmbeddedFiles"), "invalid EmbeddedFiles name tree"
        )
        if embedded_tree is None:
            return []

        recovery = self.recovery
        records: list[RawEmbeddedFile] = []
        for name, value in iter_name_tree_items(
            embedded_tree,
            self.resolver.resolve,
            self.resolver.resolve_str,
            on_malformed=recovery.malformed,
        ):
            try:
                record = self.embedded_file_record(name, value)
            except ValueError:
                if recovery.enabled:
                    continue
                raise
            records.append(record)
        return records

    def embedded_file_record(self, name: str, value: object) -> RawEmbeddedFile:
        filespec = self.resolver.as_dict(value)
        if filespec is None:
            raise ValueError("invalid embedded file spec")
        ef = self.resolver.dict_at(filespec, "EF")
        if ef is None:
            raise ValueError("invalid embedded file stream")
        stream = self.resolver.resolve(ef.get("UF") or ef.get("F"))
        if not isinstance(stream, PdfStream):
            raise ValueError("invalid embedded file stream")
        filename = (
            self.resolver.resolve_str(filespec.get("UF"))
            or self.resolver.resolve_str(filespec.get("F"))
            or name
        )
        return RawEmbeddedFile(name, filename, filespec, stream, stream.data)
