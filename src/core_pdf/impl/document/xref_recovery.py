# SPDX-License-Identifier: AGPL-3.0-only

"""Recovering a damaged cross-reference table: scanning, repairing offsets, and inferring
the catalog and trailer.

The PdfDocument methods here are functions taking the document as self;
PdfDocument lists them as its methods.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

from core_pdf.impl.base.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf.impl.base.types import MISSING, PdfReference
from core_pdf.impl.base.values import recover_pdf_name
from core_pdf.impl.document.page_tree import (
    MAX_PAGE_TREE_DEPTH,
    infer_page_tree_node_type,
    page_tree_extent,
    resolve_page_tree_node_type,
)
from core_pdf.impl.document.security import SECURITY_TRAILER_KEYS, has_security_metadata_marker
from core_pdf.impl.document.source import (
    TRAILER_METADATA_KEYS,
    is_xref_stream_dictionary,
    object_headers_present,
    xref_stream_candidates,
)
from core_pdf.impl.document.standards import find_pdf_header
from core_pdf.impl.recovery.lexer import PdfLexer, reader_rules_for
from core_pdf.impl.recovery.resolver import ObjectResolver
from core_pdf.impl.recovery.xref import StrictXRefScanner, XRefScanner, iter_indirect_object_headers
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, ResolvedObjectCache
from core_pdf.impl.spec.s_07_syntax.xref import (
    PdfXRefEntry,
    XRefRevision,
    XRefSectionReader,
    iter_xref_revisions,
    merge_xref_sections,
)
lazy from core_pdf_cythonized import markers_present

if TYPE_CHECKING:
    from core_pdf.impl.document.pdf_document import PdfDocument


def strict_xref_validation_error(self: PdfDocument[Any]) -> str | None:
    return self.caches.get_keyed(
        "strict_xref_error", self.xref_context, self.read_strict_xref_error
    )


def read_strict_xref_error(self: PdfDocument[Any]) -> str | None:
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


def brute_force_xref(self: PdfDocument[Any]) -> dict[int, PdfXRefEntry]:
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


def brute_forced_objects(self: PdfDocument[Any]) -> dict[int, object]:
    scanned = self.pending_parsed_objects
    if scanned is None or scanned[0] != self.xref_context:
        return {}
    return scanned[1]


def scan_xref(self: PdfDocument[Any]) -> None:
    self.pending_parsed_objects = None
    try:
        self.scan_xref_sections()
    finally:
        parsed = self.brute_forced_objects()
        self.pending_parsed_objects = None
        self.resolver.adopt_parsed_objects(parsed, reader_rules_for(self.xref_context))


def scan_xref_sections(self: PdfDocument[Any]) -> None:
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
            revisions = self.read_xref_revisions(start, read_section, brute_force)
            self.xref = merge_xref_sections(revision.entries for revision in revisions)
            self.trailer_dict = self.merge_revision_trailers(revisions)
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


def read_xref_revisions(
    self: PdfDocument[Any],
    start: int,
    read_section: XRefSectionReader,
    brute_force: Callable[[], dict[int, PdfXRefEntry]],
) -> list[XRefRevision]:
    revisions: list[XRefRevision] = []
    try:
        # extend keeps the revisions read before one that fails.
        revisions.extend(iter_xref_revisions(start, read_section))
    except (PdfParseError, PdfUnsupportedError, ValueError, struct.error, OSError) as error:
        # Recovery that stops at the first trailer keeps the sections read before a
        # broken /Prev, which name the catalog, and finds the other objects by scanning.
        if self.recovery_scan_all_revisions or not any(
            "Root" in revision.trailer for revision in revisions
        ):
            raise
        self.xref_was_recovered = True
        self.xref_recovery_reason = str(error)
        revisions.append(XRefRevision(0, brute_force(), {}))
    return revisions


def repair_stale_xref_offsets(
    self: PdfDocument[Any], brute_force: Callable[[], dict[int, PdfXRefEntry]] | None = None
) -> None:
    header_offset = self.pdf_header_offset()
    recovered_xref: dict[int, PdfXRefEntry] | None = None
    repaired = False
    selected = [
        (key, entry)
        for key, entry in self.xref.items()
        if entry.in_use and entry.object_stream is None and entry.offset >= 0
    ]
    matched = object_headers_present(
        self.raw_data, [key for key, _ in selected], [entry.offset for _, entry in selected]
    )
    for (key, entry), present in zip(selected, matched, strict=True):
        if present:
            continue
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
        if replacement is None or not replacement.in_use or replacement.object_stream is not None:
            continue
        if replacement.offset != entry.offset:
            self.xref[key] = replacement
            repaired = True

    if repaired:
        self.xref_was_recovered = True


def pdf_header_offset(self: PdfDocument[Any]) -> int:
    return max(0, find_pdf_header(self.raw_data))


def find_xref_entry_header(self: PdfDocument[Any], key: int, offset: int) -> int | None:
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


def xref_entry_matches_header(self: PdfDocument[Any], key: int, entry: PdfXRefEntry) -> bool:
    return object_headers_present(self.raw_data, [key], [entry.offset])[
        0
    ] or self.xref_entry_header_nearby(key, entry)


def xref_entry_header_nearby(self: PdfDocument[Any], key: int, entry: PdfXRefEntry) -> bool:
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


def is_valid_catalog_root(self: PdfDocument[Any], root_ref: object) -> bool:
    reuse_resolver = (
        self.trailer_dict.get("Encrypt") is None and self.trailer_dict.get("AuthCode") is None
    )
    if reuse_resolver:
        resolver = self.resolver
        resolver.xref = self.xref
    else:
        resolver = ObjectResolver(
            self.raw_data,
            self.xref,
            semantic_context=self.xref_context,
        )
    valid = False
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
        kids, count = page_tree_extent(resolver.resolve, pages)
        valid = kids is not None or count is not None
        return valid
    except Exception:
        return False
    finally:
        if not reuse_resolver:
            resolver.close()
        elif not valid:
            for stream in resolver.detach_parsed_caches():
                stream.close()


def infer_catalog_root(self: PdfDocument[Any]) -> PdfReference | None:
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


def merge_revision_trailers(self: PdfDocument[Any], revisions: Sequence[XRefRevision]) -> PdfDict:
    trailer = revisions[0].trailer
    if len(revisions) == 1:
        return trailer
    missing_keys = [key for key in TRAILER_METADATA_KEYS if trailer.get(key) is None]
    if not missing_keys:
        return trailer
    merged: PdfDict | None = None
    for revision in revisions[1:]:
        candidate = revision.trailer
        for key in missing_keys:
            if (merged or trailer).get(key) is not None or key not in candidate:
                continue
            value = candidate[key]
            if self.is_valid_trailer_metadata_value(key, value):
                if merged is None:
                    merged = dict(trailer)
                merged[key] = value
    return trailer if merged is None else merged


def merge_recovered_trailer_metadata(self: PdfDocument[Any], trailer: PdfDict) -> PdfDict:
    missing_keys = [key for key in TRAILER_METADATA_KEYS if trailer.get(key) is None]
    if not missing_keys:
        return trailer
    if SECURITY_TRAILER_KEYS.issuperset(missing_keys):
        if not has_security_metadata_marker(self.raw_data):
            return trailer
    elif not self.xref_was_recovered and not (
        any(
            markers_present(
                self.raw_data, tuple(b"/" + key.encode("ascii") for key in missing_keys)
            )
        )
        or (
            not SECURITY_TRAILER_KEYS.isdisjoint(missing_keys)
            and has_security_metadata_marker(self.raw_data)
        )
    ):
        return trailer
    recovered = self.infer_trailer_metadata()
    if not recovered:
        return trailer
    merged = dict(trailer)
    for key, value in recovered.items():
        if merged.get(key) is None:
            merged[key] = value
    return merged


def infer_trailer_metadata(self: PdfDocument[Any]) -> PdfDict:
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
    if (
        SECURITY_TRAILER_KEYS.issuperset(missing_keys)
        and metadata
        and not has_security_metadata_marker(self.raw_data)
    ):
        return metadata

    for candidate in self.iter_recoverable_xref_stream_dictionaries():
        for key in missing_keys:
            if key not in candidate:
                continue
            value = candidate[key]
            if self.is_valid_trailer_metadata_value(key, value):
                metadata[key] = value
    return metadata


def literal_trailer_dictionaries(self: PdfDocument[Any]) -> tuple[PdfDict, ...]:
    return self.caches.get_keyed(
        "literal_trailers", self.xref_context, self.collect_literal_trailer_dictionaries
    )


def collect_literal_trailer_dictionaries(self: PdfDocument[Any]) -> tuple[PdfDict, ...]:
    return tuple(self.iter_literal_trailer_dictionaries())


def iter_literal_trailer_dictionaries(self: PdfDocument[Any]) -> Iterator[PdfDict]:
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


def iter_recoverable_xref_stream_dictionaries(self: PdfDocument[Any]) -> Iterator[PdfDict]:
    data = self.raw_data
    data_len = len(data)
    starts = sorted(
        {
            entry.offset
            for entry in self.xref.values()
            if entry.in_use and entry.object_stream is None and 0 <= entry.offset < data_len
        }
    )
    lexer = PdfLexer(data, semantic_context=self.xref_context)
    try:
        for index in xref_stream_candidates(data, starts, data_len):
            start = starts[index]
            # Only a stream's dictionary decides; a dictionary that parses and
            # does not qualify is passed over without parsing its stream.
            lexer.rewind(start)
            try:
                lexer.read_indirect_header()
                lexer.pos = lexer.skip_ignored_at(lexer.pos)
                if data[lexer.pos : lexer.pos + 2] == b"<<" and not is_xref_stream_dictionary(
                    lexer.parse_dictionary()
                ):
                    continue
            except Exception:
                pass
            lexer.rewind(start)
            try:
                obj = lexer.parse_indirect_object()
            except Exception:
                continue
            if not isinstance(obj, PdfStream):
                continue
            dictionary = obj.dictionary
            if is_xref_stream_dictionary(dictionary):
                yield dictionary
    finally:
        lexer.close()


def is_valid_trailer_metadata_value(self: PdfDocument[Any], key: str, value: object) -> bool:
    if key == "Info":
        return isinstance(value, (PdfReference, dict))
    if key == "ID":
        return isinstance(value, (list, tuple)) and len(value) > 0
    if key == "Encrypt":
        return value is not None
    return key == "AuthCode"
