# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from core_pdf.impl.engine.spec.s_07_document.page_links import (
    PDFKEY_A,
    PDFKEY_ANNOTS,
    PDFKEY_RECT,
    PDFKEY_S,
    PDFKEY_SUBTYPE,
    link_target_direct,
    link_target_resolved,
    lookup_pdf_key,
    pdf_box_direct,
    pdf_name_direct,
)
from core_pdf.impl.engine.spec.s_07_document.protocols import FormsDocumentProtocol
from core_pdf.impl.engine.spec.s_07_objects.object_cache import InheritedValueMap
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.models import AnnotationRecord, LinkRecord
from core_pdf.impl.objects import MISSING, MissingObject, PdfReference
from core_pdf.impl.types import PdfDict, PdfObject

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument
    from core_pdf.impl.models import FieldRecord


class PageInteractionsHost(Protocol):
    document: PdfDocument
    page_number: int
    links: list[LinkRecord] | MissingObject

    @property
    def inherited_values(self) -> InheritedValueMap: ...

    def has_annotation_subtype(self, subtype_name: str) -> bool: ...

    def annotation_dicts(self) -> list[PdfDict]: ...

    def has_destination_annotation(self) -> bool: ...


class PageInteractionsMixin:
    links: list[LinkRecord] | MissingObject

    def has_annotation_subtype(self: PageInteractionsHost, subtype_name: str) -> bool:
        for annot in self.annotation_dicts():
            subtype = self.document.resolver.resolve_name(lookup_dict_key(annot, "Subtype"))
            if subtype == subtype_name:
                return True
        return False

    def annotation_dicts(self: PageInteractionsHost) -> list[PdfDict]:
        raw_annots = self.document.resolver.resolve(
            lookup_dict_key(self.inherited_values, "Annots")
        )
        if raw_annots is None:
            return []
        annots = raw_annots if isinstance(raw_annots, list) else [raw_annots]
        resolved_annots: list[PdfDict] = []
        for annot_ref in annots:
            annot = self.document.resolver.resolve(annot_ref)
            if isinstance(annot, dict):
                resolved_annots.append(cast(PdfDict, annot))
        return resolved_annots

    def has_destination_annotation(self: PageInteractionsHost) -> bool:
        for annot in self.annotation_dicts():
            if lookup_dict_key(annot, "Dest") is not None:
                return True
            action = lookup_dict_key(annot, "A")
            if isinstance(action, PdfReference):
                action = self.document.resolver.resolve(action)
            if not isinstance(action, dict):
                continue
            if self.document.resolver.resolve_name(lookup_dict_key(action, "S")) != "GoTo":
                continue
            if lookup_dict_key(action, "D") is not None:
                return True
        return False

    def get_annotations(self: PageInteractionsHost) -> list[AnnotationRecord]:
        recover_annotations = (
            self.document.xref_was_recovered or self.document.page_tree_was_recovered
        )
        annots_raw = self.document.resolver.resolve(
            lookup_dict_key(self.inherited_values, "Annots")
        )
        if annots_raw is None:
            return []
        if not isinstance(annots_raw, list):
            if recover_annotations:
                return []
            raise ValueError("invalid page Annots array")

        results = []
        for annot_ref in annots_raw:
            annot = self.document.resolver.resolve(annot_ref)
            if not isinstance(annot, dict):
                if recover_annotations:
                    continue
                raise ValueError("invalid page annotation entry")

            subtype = self.document.resolver.resolve_name(lookup_dict_key(annot, "Subtype"))
            rect = self.document.resolver.resolve_box(lookup_dict_key(annot, "Rect"))
            if rect is None:
                if recover_annotations:
                    continue
                raise ValueError("invalid page annotation rectangle")
            contents = self.document.resolver.resolve_str(lookup_dict_key(annot, "Contents")) or ""
            dest = lookup_dict_key(annot, "Dest")
            action = lookup_dict_key(annot, "A")
            if isinstance(action, PdfReference):
                action = self.document.resolver.resolve(action)
            if (
                dest is None
                and isinstance(action, dict)
                and self.document.resolver.resolve_name(lookup_dict_key(action, "S")) == "GoTo"
            ):
                dest = lookup_dict_key(action, "D")

            results.append(
                AnnotationRecord(
                    subtype=subtype,
                    rect=rect,
                    contents=contents,
                    dict_=cast(PdfDict, annot),
                    dest=cast(PdfObject | None, dest),
                    action=cast(PdfDict, action) if isinstance(action, dict) else None,
                )
            )
        return results

    def get_links(self: PageInteractionsHost) -> list[LinkRecord]:
        if self.links is not MISSING:
            return cast(list[LinkRecord], self.links)

        raw_annots = self.document.resolver.resolve(
            lookup_pdf_key(self.inherited_values, "Annots", PDFKEY_ANNOTS)
        )
        if raw_annots is None:
            self.links = []
            return []
        annots = raw_annots if isinstance(raw_annots, list) else [raw_annots]

        resolver = self.document.resolver
        resolve = self.document.resolve
        page_number = self.page_number
        records: list[LinkRecord] = []
        append_record = records.append

        for annot in annots:
            if isinstance(annot, PdfReference):
                annot = resolve(annot)
            if not isinstance(annot, dict):
                continue

            subtype = pdf_name_direct(lookup_pdf_key(annot, "Subtype", PDFKEY_SUBTYPE))
            if subtype is None:
                subtype = resolver.resolve_name(lookup_pdf_key(annot, "Subtype", PDFKEY_SUBTYPE))
            if subtype != "Link":
                continue

            rect = pdf_box_direct(lookup_pdf_key(annot, "Rect", PDFKEY_RECT))
            if rect is None:
                rect = resolver.resolve_box(lookup_pdf_key(annot, "Rect", PDFKEY_RECT))
            if rect is None:
                continue

            action = lookup_pdf_key(annot, "A", PDFKEY_A)
            if isinstance(action, PdfReference):
                action = resolve(action)
            link_type = None
            url = None
            if isinstance(action, dict):
                action = cast(PdfDict, action)
                raw_type = lookup_pdf_key(action, "S", PDFKEY_S)
                link_type = pdf_name_direct(raw_type) or resolver.resolve_name(raw_type)
                url = link_target_direct(action, link_type)
                if url is None:
                    url = link_target_resolved(resolver, action, link_type)

            append_record(
                LinkRecord(
                    bbox=rect,
                    url=url,
                    link_type=link_type,
                    page_number=page_number,
                    dict_=cast(PdfDict, annot),
                )
            )

        self.links = records
        return records

    def get_fields(self: PageInteractionsHost) -> list[FieldRecord]:
        all_fields = cast(FormsDocumentProtocol, self.document).fields()
        page_fields = []
        raw_annots = self.document.resolve(lookup_dict_key(self.inherited_values, "Annots"))
        page_annots = raw_annots if isinstance(raw_annots, list) else [raw_annots]
        page_annot_ids = {
            id(annot)
            for annot in (
                self.document.resolve(annot) for annot in page_annots if annot is not None
            )
            if isinstance(annot, dict)
        }

        for field in all_fields:
            if field.widget:
                if not isinstance(field.widget, dict):
                    raise ValueError("invalid field widget entry")
                pg_ref = lookup_dict_key(field.widget, "P")
                if pg_ref is not None:
                    pg_obj = self.document.resolver.resolve(pg_ref)
                    if (
                        isinstance(pg_obj, dict)
                        and self.document.page_index_for(pg_obj) == self.page_number - 1
                    ):
                        page_fields.append(field)
                elif id(field.widget) in page_annot_ids:
                    page_fields.append(field)
            elif field.kids:
                if not isinstance(field.kids, list):
                    raise ValueError("invalid field kids array")
                for kid_ref in field.kids:
                    kid = self.document.resolver.resolve(kid_ref)
                    if (
                        isinstance(kid, dict)
                        and self.document.resolver.resolve_name(lookup_dict_key(kid, "Subtype"))
                        == "Widget"
                    ):
                        pg_ref = lookup_dict_key(kid, "P")
                        if pg_ref is not None:
                            pg_obj = self.document.resolver.resolve(pg_ref)
                            if (
                                isinstance(pg_obj, dict)
                                and self.document.page_index_for(pg_obj) == self.page_number - 1
                            ):
                                page_fields.append(field)
                                break
                        elif id(kid) in page_annot_ids:
                            page_fields.append(field)
                            break
        return page_fields


__all__ = ("PageInteractionsMixin",)
