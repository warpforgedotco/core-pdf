# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, cast

from core_pdf.impl.engine.spec.s_07_document.document_labels import (
    infer_page_tree_node_type,
)
from core_pdf.impl.engine.spec.s_07_objects.object_cache import InheritedValueMap
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import (
    collect_inherited_values,
    lookup_dict_key,
)
from core_pdf.impl.objects import PdfReference, PdfStream
from core_pdf.impl.types import PdfDict, PdfObject

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_syntax.xref import PdfXRefEntry


RECOVERABLE_PAGE_INHERITED_KEYS = (
    "MediaBox",
    "CropBox",
    "BleedBox",
    "TrimBox",
    "ArtBox",
    "Rotate",
    "Resources",
    "Annots",
)


class DocumentPageRecoveryResolver(Protocol):
    def resolve(self, ref: object) -> object: ...

    def resolve_name(self, value: object) -> str | None: ...


class DocumentPageRecoveryMixin:
    xref: dict[int, "PdfXRefEntry"]
    resolver: DocumentPageRecoveryResolver

    def discover_page_dicts(self) -> Iterator[PdfDict]:
        candidates: list[tuple[int, int, int, PdfDict]] = []
        pages_nodes: list[tuple[int, int, int, PdfDict]] = []
        seen_objects: set[int] = set()
        for key, entry in sorted(
            self.xref.items(),
            key=lambda item: (
                item[1].offset if item[1].object_stream is None else 0,
                item[0] >> 16,
            ),
        ):
            if not entry.in_use:
                continue
            try:
                obj = self.resolver.resolve(PdfReference(key >> 16, key & 0xFFFF))
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            marker = id(obj)
            if marker in seen_objects:
                continue
            seen_objects.add(marker)
            pages_score = self.pages_candidate_score(obj)
            if pages_score > 0:
                pages_nodes.append((pages_score, entry.offset, key >> 16, obj))
            score = self.page_candidate_score(obj)
            if score <= 0:
                continue
            candidates.append((score, entry.offset, key >> 16, obj))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        pages_nodes.sort(key=lambda item: (-item[0], item[1], item[2]))
        seen_signatures: set[tuple[object, ...]] = set()
        for ignored, ignored, ignored, page_dict in candidates:
            repaired_page = self.repair_recovered_page_inherited_values(
                page_dict, [node for ignored, ignored, ignored, node in pages_nodes]
            )
            signature = self.recovered_page_signature(repaired_page)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            yield repaired_page

    def page_candidate_score(self, obj: PdfDict) -> int:
        node_type = self.resolver.resolve_name(lookup_dict_key(obj, "Type"))
        if node_type is None:
            node_type = infer_page_tree_node_type(obj)
        if node_type == "Pages":
            return -100
        if node_type not in (None, "Page"):
            return -100

        score = 0
        if node_type == "Page":
            score += 20
        if lookup_dict_key(obj, "Kids") is not None:
            score -= 30
        if lookup_dict_key(obj, "Contents") is not None:
            score += 12
        if lookup_dict_key(obj, "MediaBox") is not None:
            score += 8
        if lookup_dict_key(obj, "Resources") is not None:
            score += 4
        if lookup_dict_key(obj, "Parent") is not None:
            score += 2
        if lookup_dict_key(obj, "Annots") is not None:
            score += 1
        if score < 16:
            return -100
        return score

    def pages_candidate_score(self, obj: PdfDict) -> int:
        node_type = self.resolver.resolve_name(lookup_dict_key(obj, "Type"))
        if node_type is None:
            node_type = infer_page_tree_node_type(obj)
        if node_type != "Pages":
            return -100
        score = 20
        try:
            kids = self.resolver.resolve(lookup_dict_key(obj, "Kids"))
        except Exception:
            kids = None
        if isinstance(kids, list):
            score += min(len(kids), 20)
        try:
            count = self.resolver.resolve(lookup_dict_key(obj, "Count"))
        except Exception:
            count = None
        if type(count) is int and count >= 0:
            score += min(count, 20)
        if lookup_dict_key(obj, "Resources") is not None:
            score += 5
        if lookup_dict_key(obj, "MediaBox") is not None:
            score += 5
        return score

    def repair_recovered_page_inherited_values(
        self, page_dict: PdfDict, pages_nodes: list[PdfDict]
    ) -> PdfDict:
        missing = [
            key
            for key in RECOVERABLE_PAGE_INHERITED_KEYS
            if lookup_dict_key(page_dict, key) is None
        ]
        if not missing:
            return page_dict

        sources: list[PdfDict] = []
        parent = lookup_dict_key(page_dict, "Parent")
        if parent is not None:
            try:
                parent_obj = self.resolver.resolve(parent)
            except Exception:
                parent_obj = None
            if isinstance(parent_obj, dict):
                sources.append(parent_obj)
        sources.extend(pages_nodes)
        if not sources:
            return page_dict

        repaired: PdfDict | None = None
        for source in sources:
            source_values = self.collect_inherited_values_from_node(source, missing)
            if not source_values:
                continue
            if repaired is None:
                repaired = dict(page_dict)
            for key, value in source_values.items():
                if lookup_dict_key(repaired, key) is None:
                    repaired[key] = cast(PdfObject, value)
            missing = [key for key in missing if lookup_dict_key(repaired, key) is None]
            if not missing:
                break
        return repaired if repaired is not None else page_dict

    def collect_inherited_values_from_node(
        self, node: PdfDict, keys: list[str]
    ) -> InheritedValueMap:
        def resolve_ref(value: object) -> object:
            try:
                return self.resolver.resolve(value)
            except Exception:
                return None

        return collect_inherited_values(node, tuple(keys), resolve_ref)

    def recovered_page_signature(self, page_dict: PdfDict) -> tuple[object, ...]:
        contents = lookup_dict_key(page_dict, "Contents")
        normalized_contents = self.normalized_reference_signature(contents)
        if normalized_contents is not None:
            return ("Contents", normalized_contents)
        media_box = lookup_dict_key(page_dict, "MediaBox")
        resources = lookup_dict_key(page_dict, "Resources")
        return (
            "Shape",
            self.normalized_reference_signature(media_box),
            self.normalized_reference_signature(resources),
            id(page_dict),
        )

    def normalized_reference_signature(self, value: object) -> object:
        if isinstance(value, PdfReference):
            return ("R", value.object_number, value.generation_number)
        if isinstance(value, list):
            return tuple(self.normalized_reference_signature(item) for item in value)
        if isinstance(value, tuple):
            return tuple(self.normalized_reference_signature(item) for item in value)
        if isinstance(value, dict):
            return ("D", id(value))
        if isinstance(value, PdfStream):
            return ("S", id(value))
        return value


__all__ = ("DocumentPageRecoveryMixin", "RECOVERABLE_PAGE_INHERITED_KEYS")
