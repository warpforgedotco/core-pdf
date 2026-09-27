# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import mmap
import struct
from array import array
from collections.abc import Callable, Iterator
from functools import partial
from itertools import compress, repeat
from operator import and_, is_, itemgetter, not_, truth
from typing import TYPE_CHECKING, Any

import numpy

from core_pdf.impl.document_page_tree import (
    MAX_PAGE_TREE_DEPTH,
    infer_page_tree_node_type,
    resolve_page_tree_node_type,
)
from core_pdf.impl.document_standards import find_pdf_header
from core_pdf.impl.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf.impl.memo import DocumentCaches
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.recovery_lexer import PdfLexer, reader_rules_for
from core_pdf.impl.recovery_resolver import ObjectResolver
from core_pdf.impl.recovery_xref import StrictXRefScanner, XRefScanner, iter_indirect_object_headers
from core_pdf.impl.types import MISSING, PdfReference
from core_pdf_cythonized import object_headers_match
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict, ResolvedObjectCache
from core_pdf_spec.s_07_syntax.xref import PdfXRefEntry, iter_xref_revisions, merge_xref_sections
from core_pdf_spec.standards import SemanticContext


class XRefRecovery:
    if not TYPE_CHECKING:
        __slots__ = ()

    raw_data: bytes | mmap.mmap
    xref: dict[int, PdfXRefEntry]
    trailer_dict: PdfDict
    resolver: ObjectResolver
    xref_was_recovered: bool
    xref_recovery_reason: str | None
    recovery_scan_all_revisions: bool
    pending_parsed_objects: tuple[SemanticContext | None, dict[int, object]] | None
    caches: DocumentCaches

    @property
    def xref_context(self) -> SemanticContext | None:
        return self.resolver.semantic_context

    def strict_xref_validation_error(self) -> str | None:
        return self.caches.get_keyed(
            "strict_xref_error", self.xref_context, self.read_strict_xref_error
        )

    def read_strict_xref_error(self) -> str | None:
        start = XRefScanner.find_startxref(self.raw_data, semantic_context=self.xref_context)
        if start is None:
            return None
        try:
            read_section = partial(
                StrictXRefScanner.recover_section_at,
                self.raw_data,
                semantic_context=self.xref_context,
            )
            for _revision in iter_xref_revisions(start, read_section):
                pass
        except (PdfParseError, PdfUnsupportedError, ValueError, struct.error, OSError) as error:
            return str(error)
        return None

    def brute_force_xref(self) -> dict[int, PdfXRefEntry]:
        parsed: dict[int, object] = {}
        context = self.xref_context
        xref = XRefScanner.brute_force_scan(
            self.raw_data,
            stop_at_first_trailer=not self.recovery_scan_all_revisions,
            semantic_context=context,
            parsed_objects=parsed,
        )
        self.pending_parsed_objects = (context, parsed)
        return xref

    def brute_forced_objects(self) -> dict[int, object]:
        scanned = self.pending_parsed_objects
        if scanned is None or scanned[0] != self.xref_context:
            return {}
        return scanned[1]

    def scan_xref(self) -> None:
        self.pending_parsed_objects = None
        try:
            self.scan_xref_sections()
        finally:
            parsed = self.brute_forced_objects()
            self.pending_parsed_objects = None
            self.resolver.adopt_parsed_objects(parsed, reader_rules_for(self.xref_context))

    def scan_xref_sections(self) -> None:
        data = self.raw_data
        try:
            start = XRefScanner.find_startxref(data, semantic_context=self.xref_context)
        except ValueError as exc:
            raise PdfParseError("invalid xref section") from exc
        if start is None and b"startxref" in data:
            raise PdfParseError("missing startxref")
        if start is not None and start < 0:
            raise PdfParseError("invalid xref section")

        brute_forced: dict[int, PdfXRefEntry] | None = None

        def brute_force() -> dict[int, PdfXRefEntry]:
            nonlocal brute_forced
            if brute_forced is None:
                brute_forced = self.brute_force_xref()
            return brute_forced

        recovery_reason = None
        if start is not None:
            try:
                read_section = partial(
                    XRefScanner.recover_section_at,
                    data,
                    semantic_context=self.xref_context,
                )
                revisions = list(iter_xref_revisions(start, read_section))
                self.xref = merge_xref_sections(revision.entries for revision in revisions)
                self.trailer_dict = revisions[0].trailer
                self.repair_stale_xref_offsets(brute_force)
                self.trailer_dict = self.merge_recovered_trailer_metadata(self.trailer_dict)
                root_ref = self.trailer_dict.get("Root")
                if root_ref is None or not self.is_valid_catalog_root(root_ref):
                    self.xref.update(brute_force())
                    self.xref_was_recovered = True
                    catalog_ref = self.infer_catalog_root()
                    if catalog_ref is not None:
                        self.trailer_dict = dict(self.trailer_dict)
                        self.trailer_dict["Root"] = catalog_ref
                    self.trailer_dict = self.merge_recovered_trailer_metadata(self.trailer_dict)
            except (PdfParseError, PdfUnsupportedError, ValueError, struct.error, OSError) as error:
                recovery_reason = str(error)
            else:
                return

        self.xref = brute_force()
        self.xref_was_recovered = True
        if recovery_reason is not None:
            self.xref_recovery_reason = recovery_reason
        if not self.xref:
            self.trailer_dict = {}
            return
        catalog_ref = self.infer_catalog_root()
        self.trailer_dict = {"Root": catalog_ref} if catalog_ref is not None else {}
        self.trailer_dict = self.merge_recovered_trailer_metadata(self.trailer_dict)

    def repair_stale_xref_offsets(
        self, brute_force: Callable[[], dict[int, PdfXRefEntry]] | None = None
    ) -> None:
        header_offset = self.pdf_header_offset()
        recovered_xref: dict[int, PdfXRefEntry] | None = None
        repaired = False
        keys = list(self.xref)
        entries = list(self.xref.values())
        offsets = list(map(ENTRY_OFFSET, entries))
        selected = list(
            map(
                and_,
                map(
                    and_,
                    map(truth, map(ENTRY_IN_USE, entries)),
                    map(is_, map(ENTRY_OBJECT_STREAM, entries), repeat(None), strict=False),
                    strict=True,
                ),
                map((0).__le__, offsets),
                strict=True,
            )
        )
        keys = list(compress(keys, selected))
        entries = list(compress(entries, selected))
        matched = object_headers_present(self.raw_data, keys, list(compress(offsets, selected)))
        for index in compress(range(len(keys)), map(not_, matched)):
            key = keys[index]
            entry = entries[index]
            if self.xref_entry_header_nearby(key, entry):
                continue
            if header_offset:
                shifted = entry._replace(offset=entry.offset + header_offset)
                if self.xref_entry_matches_header(key, shifted):
                    self.xref[key] = shifted
                    repaired = True
                    continue
                shifted_offset = self.find_xref_entry_header(
                    key,
                    entry.offset + header_offset,
                )
                if shifted_offset is not None:
                    self.xref[key] = entry._replace(offset=shifted_offset)
                    repaired = True
                    continue
            if recovered_xref is None:
                recovered_xref = self.brute_force_xref() if brute_force is None else brute_force()
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

    def pdf_header_offset(self) -> int:
        return max(0, find_pdf_header(self.raw_data))

    def find_xref_entry_header(self, key: int, offset: int) -> int | None:
        data = self.raw_data
        expected_object_number = key >> 16
        expected_generation_number = key & 0xFFFF
        search_start = max(0, offset - 1024)
        search_end = min(len(data), offset + 1024)
        for parsed_offset, object_number, generation_number in iter_indirect_object_headers(
            data,
            search_start,
            search_end,
            allow_prefix_before_start=True,
        ):
            if (
                object_number == expected_object_number
                and generation_number == expected_generation_number
            ):
                return parsed_offset
        return None

    def xref_entry_matches_header(self, key: int, entry: PdfXRefEntry) -> bool:
        return object_headers_present(self.raw_data, [key], [entry.offset])[
            0
        ] or self.xref_entry_header_nearby(key, entry)

    def xref_entry_header_nearby(self, key: int, entry: PdfXRefEntry) -> bool:
        data = self.raw_data
        offset = entry.offset
        data_len = len(data)
        if offset < 0 or offset >= data_len:
            return False
        search_end = min(data_len, offset + 64)
        for parsed_offset, object_number, generation_number in iter_indirect_object_headers(
            data,
            offset,
            search_end,
            allow_prefix_before_start=True,
        ):
            return (
                parsed_offset == offset
                and object_number == key >> 16
                and generation_number == key & 0xFFFF
            )
        return False

    def is_valid_catalog_root(self, root_ref: object) -> bool:
        resolver = ObjectResolver(
            self.raw_data,
            self.xref,
            semantic_context=self.xref_context,
        )
        try:
            root = resolver.as_dict(root_ref)
            if root is None:
                return False
            if recover_pdf_name(root.get("Type")) != "Catalog":
                return False
            pages = resolver.dict_at(root, "Pages")
            if pages is None:
                return False
            node_type = resolve_page_tree_node_type(resolver, pages)
            if node_type != "Pages":
                return False
            kids = resolver.resolve(pages.get("Kids"))
            count = resolver.resolve(pages.get("Count"))
            return isinstance(kids, list) or (type(count) is int and count >= 0)
        except Exception:
            return False
        finally:
            resolver.close()

    def infer_catalog_root(self) -> PdfReference | None:
        data = self.raw_data
        object_cache: ResolvedObjectCache = {}
        resolver = ObjectResolver(
            self.raw_data,
            self.xref,
            semantic_context=self.xref_context,
        )
        lexer = PdfLexer(data, semantic_context=self.xref_context)
        entries_by_ref = {
            (k >> 16, k & 0xFFFF): entry for k, entry in self.xref.items() if entry.in_use
        }
        unclaimed = dict(self.brute_forced_objects())

        def parse_at(offset: int) -> Any:
            parsed = unclaimed.pop(offset, MISSING)
            if parsed is not MISSING:
                return parsed
            lexer.rewind(offset)
            return lexer.parse_indirect_object()

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
                resolved = resolver.resolve_or_none(value)
                object_cache[key] = resolved
                return resolved
            try:
                resolved = parse_at(entry.offset)
            except Exception:
                return None
            object_cache[key] = resolved
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
            node_type = recover_pdf_name(resolve_for_inference(node.get("Type"), depth + 1))
            if node_type is None:
                node_type = infer_page_tree_node_type(node)
            if node_type == "Page":
                score = 10
                if node.get("Contents") is not None:
                    score += 3
                if node.get("MediaBox") is not None:
                    score += 2
                return score
            if node_type != "Pages":
                return -50
            kids = resolve_for_inference(node.get("Kids"), depth + 1)
            count = resolve_for_inference(node.get("Count"), depth + 1)
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
            type_name = recover_pdf_name(obj.get("Type"))
            pages = obj.get("Pages")
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
                if obj.get(key) is not None:
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
                    if entry.in_use and (entry.object_stream is not None or entry.offset >= 0)
                },
                key=lambda item: (item[3], item[2], item[0]),
            )
            scored: list[tuple[int, int, int, int]] = []
            for obj_num, gen_num, offset, compressed in candidates:
                if compressed:
                    obj = resolver.resolve_or_none(PdfReference(obj_num, gen_num))
                else:
                    try:
                        obj = parse_at(offset)
                    except Exception:
                        continue
                object_cache[(obj_num, gen_num)] = obj
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
        missing_keys = [key for key in TRAILER_METADATA_KEYS if trailer.get(key) is None]
        if not missing_keys:
            return trailer
        if not self.xref_was_recovered and not any(
            self.raw_data.find(b"/" + key.encode("ascii")) >= 0 for key in missing_keys
        ):
            return trailer
        if missing_keys == ["Encrypt"] and not self.xref_was_recovered:
            return trailer
        if missing_keys == ["Encrypt"] and self.raw_data.find(b"Encrypt") < 0:
            return trailer
        recovered = self.infer_trailer_metadata()
        if not recovered:
            return trailer
        merged = dict(trailer)
        for key, value in recovered.items():
            if merged.get(key) is None:
                merged[key] = value
        return merged

    def infer_trailer_metadata(self) -> PdfDict:
        metadata: PdfDict = {}

        for candidate in self.literal_trailer_dictionaries():
            for key in TRAILER_METADATA_KEYS:
                if key not in candidate:
                    continue
                value = candidate[key]
                if self.is_valid_trailer_metadata_value(key, value):
                    metadata[key] = value

        missing_keys = [key for key in TRAILER_METADATA_KEYS if key not in metadata]
        if not missing_keys:
            return metadata
        if missing_keys == ["Encrypt"] and metadata and self.raw_data.find(b"Encrypt") < 0:
            return metadata

        for candidate in self.iter_recoverable_xref_stream_dictionaries():
            for key in missing_keys:
                if key not in candidate:
                    continue
                value = candidate[key]
                if self.is_valid_trailer_metadata_value(key, value):
                    metadata[key] = value
        return metadata

    def literal_trailer_dictionaries(self) -> tuple[PdfDict, ...]:
        return self.caches.get_keyed(
            "literal_trailers", self.xref_context, self.collect_literal_trailer_dictionaries
        )

    def collect_literal_trailer_dictionaries(self) -> tuple[PdfDict, ...]:
        return tuple(self.iter_literal_trailer_dictionaries())

    def iter_literal_trailer_dictionaries(self) -> Iterator[PdfDict]:
        data = self.raw_data
        lexer = PdfLexer(data, semantic_context=self.xref_context)
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
        data = self.raw_data
        data_len = len(data)
        starts = sorted(
            {
                entry.offset
                for entry in self.xref.values()
                if entry.in_use and entry.object_stream is None and 0 <= entry.offset < data_len
            }
        )
        last = len(starts) - 1
        lexer = PdfLexer(data, semantic_context=self.xref_context)
        try:
            for index, offset in enumerate(starts):
                end = starts[index + 1] if index < last else data_len
                if not self.may_be_xref_stream(data, offset, end):
                    continue
                lexer.rewind(offset)
                try:
                    obj = lexer.parse_indirect_object()
                except Exception:
                    continue
                if not isinstance(obj, PdfStream):
                    continue
                dictionary = obj.dictionary
                if recover_pdf_name(dictionary.get("Type")) == "XRef" or (
                    dictionary.get("W") is not None and dictionary.get("Size") is not None
                ):
                    yield dictionary
        finally:
            lexer.close()

    def may_be_xref_stream(self, data: bytes | mmap.mmap, offset: int, end: int) -> bool:
        if data.find(b"#", offset, end) >= 0 or data.find(b"XRef", offset, end) >= 0:
            return True
        return data.find(b"/W", offset, end) >= 0 and data.find(b"/Size", offset, end) >= 0

    def is_valid_trailer_metadata_value(self, key: str, value: object) -> bool:
        if key == "Info":
            return isinstance(value, (PdfReference, dict))
        if key == "ID":
            return isinstance(value, (list, tuple)) and len(value) > 0
        if key == "Encrypt":
            return value is not None
        return key == "AuthCode"


HUGE_OBJECT_NUMBER = 1 << 63


ENTRY_OFFSET = itemgetter(0)


ENTRY_IN_USE = itemgetter(2)


ENTRY_OBJECT_STREAM = itemgetter(3)


def object_headers_present(data: Any, keys: list[int], offsets: list[int]) -> list[bool]:
    data_len = len(data)
    try:
        key_column = numpy.array(keys, dtype=numpy.int64)
        offset_column = numpy.array(offsets, dtype=numpy.int64)
    except OverflowError:
        return object_headers_present_by_entry(data, keys, offsets)
    numbers = key_column >> 16
    generations = key_column & 0xFFFF
    offset_column[(offset_column < 0) | (offset_column >= data_len)] = -1
    states = object_headers_match(data, numbers, generations, offset_column)
    return (states == 1).tolist()


def object_headers_present_by_entry(data: Any, keys: list[int], offsets: list[int]) -> list[bool]:
    data_len = len(data)
    numbers = array("q")
    generations = array("q")
    clamped = array("q")
    for key, offset in zip(keys, offsets, strict=True):
        number = key >> 16
        numbers.append(number if number < HUGE_OBJECT_NUMBER else -2)
        generations.append(key & 0xFFFF)
        clamped.append(offset if 0 <= offset < data_len else -1)
    states = object_headers_match(data, numbers, generations, clamped)
    present = [state == 1 for state in states.tolist()]
    for index, state in enumerate(states.tolist()):
        if state == 2:
            offset = clamped[index]
            end = offset
            while end < data_len and 0x30 <= data[end] <= 0x39:
                end += 1
            present[index] = int(bytes(data[offset:end])) == keys[index] >> 16
    return present


TRAILER_METADATA_KEYS = ("Info", "ID", "Encrypt", "AuthCode")


__all__ = (
    "ENTRY_IN_USE",
    "ENTRY_OBJECT_STREAM",
    "ENTRY_OFFSET",
    "HUGE_OBJECT_NUMBER",
    "TRAILER_METADATA_KEYS",
    "XRefRecovery",
    "object_headers_present",
    "object_headers_present_by_entry",
)
