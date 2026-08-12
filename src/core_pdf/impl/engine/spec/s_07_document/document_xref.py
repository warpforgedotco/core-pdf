# SPDX-License-Identifier: AGPL-3.0-only
"""Document-level cross-reference loading and repair."""

from __future__ import annotations

import struct
from collections.abc import Iterator
from typing import cast

from core_pdf.impl.engine.spec.s_07_document.document_labels import (
    MAX_PAGE_TREE_DEPTH,
    infer_page_tree_node_type,
    resolve_page_tree_node_type,
)
from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_objects.object_cache import (
    CachedPdfObject,
    ResolvedObjectCache,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_objects.resolver import ObjectResolver
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_07_syntax.xref import (
    PdfXRefEntry,
    XRefScanner,
    parse_object_marker_prefix,
)
from core_pdf.impl.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf.impl.objects import PdfReference, PdfStream
from core_pdf.impl.types import PdfByteBuffer, PdfDict, PdfObject

TRAILER_METADATA_KEYS = ("Info", "ID", "Encrypt")


class DocumentXRefMixin:
    raw_data: PdfByteBuffer
    xref: dict[int, PdfXRefEntry]
    trailer_dict: PdfDict
    xref_was_recovered: bool

    def scan_xref(self) -> None:
        data = self.raw_data
        try:
            start = XRefScanner.find_startxref(data)
        except ValueError as exc:
            raise PdfParseError("invalid xref section") from exc
        if start is None:
            if b"startxref" in data:
                raise PdfParseError("missing startxref")
            self.xref = XRefScanner.brute_force_scan(data)
            self.xref_was_recovered = True
            if not self.xref:
                self.trailer_dict = {}
                return
            catalog_ref = self.infer_catalog_root()
            self.trailer_dict = {"Root": catalog_ref} if catalog_ref is not None else {}
            self.trailer_dict = self.merge_recovered_trailer_metadata(self.trailer_dict)
            return
        if start < 0:
            raise PdfParseError("invalid xref section")

        try:
            self.xref, self.trailer_dict = XRefScanner.load_section_chain(data, start, set())
            self.repair_stale_xref_offsets()
            self.trailer_dict = self.merge_recovered_trailer_metadata(self.trailer_dict)
            root_ref = lookup_dict_key(self.trailer_dict, "Root")
            if root_ref is None or not self.is_valid_catalog_root(root_ref):
                self.xref.update(XRefScanner.brute_force_scan(data))
                self.xref_was_recovered = True
                catalog_ref = self.infer_catalog_root()
                if catalog_ref is not None:
                    self.trailer_dict = dict(self.trailer_dict)
                    self.trailer_dict["Root"] = catalog_ref
                self.trailer_dict = self.merge_recovered_trailer_metadata(self.trailer_dict)
        except (PdfParseError, PdfUnsupportedError, ValueError, struct.error, OSError):
            self.xref = XRefScanner.brute_force_scan(data)
            self.xref_was_recovered = True
            if not self.xref:
                self.trailer_dict = {}
                return
            catalog_ref = self.infer_catalog_root()
            self.trailer_dict = {"Root": catalog_ref} if catalog_ref is not None else {}
            self.trailer_dict = self.merge_recovered_trailer_metadata(self.trailer_dict)

    def repair_stale_xref_offsets(self) -> None:
        recovered_xref: dict[int, PdfXRefEntry] | None = None
        repaired = False
        for key, entry in list(self.xref.items()):
            if not entry.in_use or entry.object_stream is not None or entry.offset < 0:
                continue
            if self.xref_entry_matches_header(key, entry):
                continue
            if recovered_xref is None:
                recovered_xref = XRefScanner.brute_force_scan(self.raw_data)
            replacement = recovered_xref.get(key)
            if (
                replacement is None
                or not replacement.in_use
                or replacement.object_stream is not None
            ):
                continue
            if replacement.offset != entry.offset:
                self.xref[key] = replacement
                repaired = True

        if repaired:
            self.xref_was_recovered = True

    def xref_entry_matches_header(self, key: int, entry: PdfXRefEntry) -> bool:
        data = self.raw_data
        offset = entry.offset
        data_len = len(data)
        if offset < 0 or offset >= data_len:
            return False

        expected_object_number = key >> 16
        expected_generation_number = key & 0xFFFF

        pos = offset
        if pos < data_len and 48 <= data[pos] <= 57:
            obj_num = 0
            while pos < data_len and 48 <= data[pos] <= 57:
                obj_num = obj_num * 10 + (data[pos] - 48)
                pos += 1
            if pos < data_len and data[pos] in (0, 9, 10, 12, 13, 32):
                while pos < data_len and data[pos] in (0, 9, 10, 12, 13, 32):
                    pos += 1
                if pos < data_len and 48 <= data[pos] <= 57:
                    gen_num = 0
                    while pos < data_len and 48 <= data[pos] <= 57:
                        gen_num = gen_num * 10 + (data[pos] - 48)
                        pos += 1
                    if (
                        pos < data_len
                        and data[pos] in (0, 9, 10, 12, 13, 32)
                        and obj_num == expected_object_number
                        and gen_num == expected_generation_number
                    ):
                        while pos < data_len and data[pos] in (0, 9, 10, 12, 13, 32):
                            pos += 1
                        if (
                            pos + 3 <= data_len
                            and data[pos : pos + 3] == b"obj"
                            and (pos + 3 == data_len or data[pos + 3] in (0, 9, 10, 12, 13, 32))
                        ):
                            return True

        search_end = min(data_len, offset + 64)
        marker = data.find(b"obj", offset, search_end)
        while marker >= 0:
            parsed = parse_object_marker_prefix(data, marker)
            if parsed is not None:
                parsed_offset, object_number, generation_number = parsed
                return (
                    parsed_offset == offset
                    and object_number == expected_object_number
                    and generation_number == expected_generation_number
                )
            marker = data.find(b"obj", marker + 3, search_end)
        return False

    def is_valid_catalog_root(self, root_ref: object) -> bool:
        resolver = ObjectResolver(self.raw_data, self.xref, self.trailer_dict)
        try:
            root = resolver.resolve(root_ref)
            if not isinstance(root, dict):
                return False
            if normalize_pdf_name(lookup_dict_key(root, "Type")) != "Catalog":
                return False
            pages = resolver.resolve(lookup_dict_key(root, "Pages"))
            if not isinstance(pages, dict):
                return False
            pages = cast(PdfDict, pages)
            node_type = resolve_page_tree_node_type(resolver, pages)
            if node_type != "Pages":
                return False
            kids = resolver.resolve(lookup_dict_key(pages, "Kids"))
            count = resolver.resolve(lookup_dict_key(pages, "Count"))
            return isinstance(kids, list) or (type(count) is int and count >= 0)
        except Exception:
            return False
        finally:
            resolver.close()

    def infer_catalog_root(self) -> PdfReference | None:
        data = self.raw_data
        object_cache: ResolvedObjectCache = {}
        resolver = ObjectResolver(self.raw_data, self.xref, self.trailer_dict)
        lexer = PdfLexer(data)
        entries_by_ref = {
            (k >> 16, k & 0xFFFF): entry for k, entry in self.xref.items() if entry.in_use
        }

        def resolve_for_inference(value: object, depth: int = 0) -> object:
            if depth > 12:
                return None
            if not isinstance(value, PdfReference):
                return value
            key = (value.object_number, value.generation_number)
            if key in object_cache:
                return object_cache[key]
            entry = entries_by_ref.get(key)
            if entry is None and value.generation_number != 0:
                key = (value.object_number, 0)
                entry = entries_by_ref.get(key)
            if entry is None:
                return None
            if entry.object_stream is not None:
                try:
                    resolved = resolver.resolve(value)
                except Exception:
                    return None
                object_cache[key] = cast(CachedPdfObject, resolved)
                return resolved
            lexer.rewind(entry.offset)
            try:
                resolved = lexer.parse_indirect_object()
            except Exception:
                return None
            object_cache[key] = cast(CachedPdfObject, resolved)
            return resolved

        def page_tree_score(node: object, depth: int = 0, seen: set[int] | None = None) -> int:
            if depth > MAX_PAGE_TREE_DEPTH:
                return -1000
            if seen is None:
                seen = set()
            node = resolve_for_inference(node, depth)
            if not isinstance(node, dict):
                return -100
            marker = id(node)
            if marker in seen:
                return -100
            seen.add(marker)
            node_type = normalize_pdf_name(
                resolve_for_inference(lookup_dict_key(node, "Type"), depth + 1)
            )
            if node_type is None:
                node_type = infer_page_tree_node_type(cast(PdfDict, node))
            if node_type == "Page":
                score = 10
                if lookup_dict_key(node, "Contents") is not None:
                    score += 3
                if lookup_dict_key(node, "MediaBox") is not None:
                    score += 2
                return score
            if node_type != "Pages":
                return -50
            kids = resolve_for_inference(lookup_dict_key(node, "Kids"), depth + 1)
            count = resolve_for_inference(lookup_dict_key(node, "Count"), depth + 1)
            score = 15
            if type(count) is int and count >= 0:
                score += min(count, 20)
            if not isinstance(kids, list) or not kids:
                return score - 20
            child_scores = [page_tree_score(kid, depth + 1, seen.copy()) for kid in kids[:32]]
            valid_children = [child_score for child_score in child_scores if child_score > 0]
            if not valid_children:
                return score - 30
            return score + sum(valid_children)

        def catalog_score(obj: object) -> int:
            if not isinstance(obj, dict):
                return -1000
            type_name = normalize_pdf_name(lookup_dict_key(obj, "Type"))
            pages = lookup_dict_key(obj, "Pages")
            score = 0
            if type_name == "Catalog":
                score += 100
            elif pages is not None:
                score += 25
            else:
                return -100
            if pages is not None:
                pages_score = page_tree_score(pages)
                if pages_score <= 0:
                    score -= 150
                else:
                    score += pages_score
            for key in ("Outlines", "Names", "Dests", "AcroForm", "PageLabels"):
                if lookup_dict_key(obj, key) is not None:
                    score += 2
            return score

        def select_catalog_root() -> PdfReference | None:
            candidates = sorted(
                {
                    (
                        k >> 16,
                        k & 0xFFFF,
                        entry.offset if entry.object_stream is None else 0,
                        entry.object_stream is not None,
                    )
                    for k, entry in self.xref.items()
                    if entry.in_use
                    and (
                        (entry.object_stream is None and entry.offset >= 0)
                        or entry.object_stream is not None
                    )
                },
                key=lambda item: (item[3], item[2], item[0]),
            )
            scored: list[tuple[int, int, int, int]] = []
            for obj_num, gen_num, offset, compressed in candidates:
                if compressed:
                    try:
                        obj = resolver.resolve(PdfReference(obj_num, gen_num))
                    except Exception:
                        continue
                else:
                    lexer.rewind(offset)
                    try:
                        obj = lexer.parse_indirect_object()
                    except Exception:
                        continue
                object_cache[(obj_num, gen_num)] = cast(CachedPdfObject, obj)
                score = catalog_score(obj)
                if score > -100:
                    scored.append((score, -offset, obj_num, gen_num))
            if not scored:
                return None
            scored.sort(reverse=True)
            ignored, ignored, obj_num, gen_num = scored[0]
            return PdfReference(obj_num, gen_num)

        try:
            return select_catalog_root()
        finally:
            object_cache.clear()
            lexer.close()
            resolver.close()

    def merge_recovered_trailer_metadata(self, trailer: PdfDict) -> PdfDict:
        missing_keys = [
            key for key in TRAILER_METADATA_KEYS if lookup_dict_key(trailer, key) is None
        ]
        if not missing_keys:
            return trailer
        if not getattr(self, "xref_was_recovered", False) and not any(
            self.raw_data.find(b"/" + key.encode("ascii")) >= 0 for key in missing_keys
        ):
            return trailer
        if missing_keys == ["Encrypt"] and not getattr(self, "xref_was_recovered", False):
            return trailer
        if missing_keys == ["Encrypt"] and self.raw_data.find(b"Encrypt") < 0:
            return trailer
        recovered = self.infer_trailer_metadata()
        if not recovered:
            return trailer
        merged = dict(trailer)
        for key, value in recovered.items():
            if lookup_dict_key(merged, key) is None:
                merged[key] = cast(PdfObject, value)
        return merged

    def infer_trailer_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {}

        for candidate in self.iter_literal_trailer_dictionaries():
            for key in TRAILER_METADATA_KEYS:
                value = lookup_dict_key(candidate, key)
                if self.is_valid_trailer_metadata_value(key, value):
                    metadata[key] = value

        missing_keys = [key for key in TRAILER_METADATA_KEYS if key not in metadata]
        if not missing_keys:
            return metadata
        if missing_keys == ["Encrypt"] and metadata and self.raw_data.find(b"Encrypt") < 0:
            return metadata

        for candidate in self.iter_recoverable_xref_stream_dictionaries():
            for key in missing_keys:
                value = lookup_dict_key(candidate, key)
                if self.is_valid_trailer_metadata_value(key, value):
                    metadata[key] = value
        return metadata

    def iter_literal_trailer_dictionaries(self) -> Iterator[PdfDict]:
        data = self.raw_data
        lexer = PdfLexer(data)
        try:
            search_from = 0
            while True:
                marker = data.find(b"trailer", search_from)
                if marker < 0:
                    break
                search_from = marker + len(b"trailer")
                dict_start = data.find(b"<<", search_from, search_from + 4096)
                if dict_start < 0:
                    continue
                lexer.rewind(dict_start)
                try:
                    candidate = lexer.parse_dictionary()
                except Exception:
                    continue
                yield candidate
        finally:
            lexer.close()

    def iter_recoverable_xref_stream_dictionaries(self) -> Iterator[PdfDict]:
        lexer = PdfLexer(self.raw_data)
        try:
            for key, entry in sorted(self.xref.items()):
                if not entry.in_use or entry.object_stream is not None or entry.offset < 0:
                    continue
                lexer.rewind(entry.offset)
                try:
                    obj = lexer.parse_indirect_object()
                except Exception:
                    continue
                if not isinstance(obj, PdfStream):
                    continue
                dictionary = obj.dictionary
                if normalize_pdf_name(lookup_dict_key(dictionary, "Type")) == "XRef" or (
                    lookup_dict_key(dictionary, "W") is not None
                    and lookup_dict_key(dictionary, "Size") is not None
                ):
                    yield cast(PdfDict, dictionary)
        finally:
            lexer.close()

    def is_valid_trailer_metadata_value(self, key: str, value: object) -> bool:
        if key == "Info":
            return isinstance(value, (PdfReference, dict))
        if key == "ID":
            return isinstance(value, (list, tuple)) and len(value) > 0
        if key == "Encrypt":
            return isinstance(value, (PdfReference, dict))
        return False


__all__ = ("DocumentXRefMixin", "MAX_PAGE_TREE_DEPTH", "TRAILER_METADATA_KEYS")
