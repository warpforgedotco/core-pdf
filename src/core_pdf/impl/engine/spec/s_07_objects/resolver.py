# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve indirect references against the cross-reference table."""

from __future__ import annotations

import contextlib
import mmap
import threading
from typing import cast

from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_objects.indirect_headers import (
    find_indirect_object_header,
)
from core_pdf.impl.engine.spec.s_07_objects.object_cache import (
    CachedPdfObject,
    DeepObjectCache,
    GenerationZeroObjectCache,
    ObjectCache,
)
from core_pdf.impl.engine.spec.s_07_objects.resolver_values import ResolverValueMixin
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_07_syntax.objects import PdfObjectStream
from core_pdf.impl.engine.spec.s_07_syntax.xref import (
    PdfXRefEntry,
    key_for,
    parse_object_marker_prefix,
)
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.objects import MISSING, PdfReference, PdfStream
from core_pdf.impl.types import Decipher, PdfDict

COMMON_KEYWORDS: tuple[bytes, ...] = (
    b"BT",
    b"ET",
    b"T*",
    b"Td",
    b"TD",
    b"Tj",
    b"TJ",
    b"Tm",
    b"Tf",
    b"TL",
    b"Tc",
    b"Tw",
    b"Tz",
    b"Tr",
    b"Ts",
    b"'",
    b'"',
    b"Do",
    b"BI",
    b"BDC",
    b"BMC",
    b"EMC",
    b"q",
    b"Q",
    b"cm",
    b"g",
    b"rg",
    b"k",
    b"G",
    b"RG",
    b"K",
    b"CS",
    b"cs",
    b"SC",
    b"SCN",
    b"sc",
    b"scn",
    b"sh",
    b"i",
    b"ri",
    b"MP",
    b"DP",
    b"BX",
    b"EX",
    b"true",
    b"false",
    b"null",
    b"R",
)

STREAM_DECODE_KEYS = frozenset(
    {
        "F",
        "Filter",
        "DecodeParms",
        "DP",
        "FFilter",
        "FDecodeParms",
    }
)


class ObjectResolver(ResolverValueMixin):
    __slots__ = (
        "data",
        "xref",
        "xref_gen0",
        "trailer",
        "decipher",
        "objects",
        "objects_gen0",
        "object_streams",
        "deep_cache",
        "kw_cache",
        "lexer_stack",
        "lock",
        "thread_state",
        "recover_missing",
        "recovery_offsets",
    )

    def __init__(
        self,
        data: bytes | bytearray | memoryview | mmap.mmap,
        xref: dict[int, PdfXRefEntry],
        trailer: PdfDict,
        decipher: Decipher | None = None,
        *,
        recover_missing: bool = False,
    ) -> None:
        # Keep an owned view.  Reusing the caller's memoryview lets a temporary
        # resolver.close() release the document's source buffer underneath
        # concurrent readers.
        self.data = memoryview(data)
        self.xref = xref
        self.trailer = trailer
        self.decipher = decipher
        self.recover_missing = recover_missing
        self.recovery_offsets: dict[int, tuple[int, ...]] | None = None
        self.objects: ObjectCache = {}

        max_obj = 0
        if self.xref:
            for k in self.xref:
                obj_num = k >> 16
                if obj_num > max_obj:
                    max_obj = obj_num

        if max_obj < 1000000:
            self.objects_gen0: GenerationZeroObjectCache | None = [MISSING] * (max_obj + 1)
            self.xref_gen0: list[PdfXRefEntry | None] | None = [None] * (max_obj + 1)
            for k, entry in self.xref.items():
                if (k & 0xFFFF) == 0:
                    self.xref_gen0[k >> 16] = entry
        else:
            self.objects_gen0 = None
            self.xref_gen0 = None

        self.object_streams: dict[int, PdfObjectStream] = {}
        self.deep_cache: DeepObjectCache = {}
        self.kw_cache: dict[bytes, object] = {key: key.decode("latin-1") for key in COMMON_KEYWORDS}
        self.lexer_stack: list[PdfLexer] = []
        self.lock = threading.RLock()
        self.thread_state = threading.local()

    def get_lexer(self) -> PdfLexer:
        with self.lock:
            if self.lexer_stack:
                lexer = self.lexer_stack.pop()
                lexer.decipher = self.decipher
                return lexer
        return PdfLexer(
            self.data,
            reference_resolver=self.resolve,
            decipher=self.decipher,
            kw_cache=dict(self.kw_cache),
        )

    def release_lexer(self, lexer: PdfLexer) -> None:
        with self.lock:
            self.lexer_stack.append(lexer)

    def close(self) -> None:
        for lexer in self.lexer_stack:
            lexer.close()
        self.lexer_stack.clear()
        self.objects.clear()
        if self.objects_gen0 is not None:
            self.objects_gen0.clear()
        self.objects_gen0 = None
        self.xref_gen0 = None
        self.object_streams.clear()
        self.deep_cache.clear()
        self.kw_cache.clear()
        self.decipher = None
        self.recovery_offsets = None
        with contextlib.suppress(ValueError):
            self.data.release()
        self.data = memoryview(b"")

    def resolve(self, ref: object) -> object:
        if type(ref) is not PdfReference:
            return ref

        with self.lock:
            cached = self.internal_cached_object(ref)
            if cached is not MISSING:
                return cached

        resolving = getattr(self.thread_state, "resolving", None)
        if resolving is None:
            resolving = set()
            self.thread_state.resolving = resolving
        cache_key = key_for(ref.object_number, ref.generation_number)
        if cache_key in resolving:
            return ref

        resolving.add(cache_key)
        try:
            resolved = self.internal_resolve_reference(ref)
        finally:
            resolving.remove(cache_key)

        with self.lock:
            cached = self.internal_cached_object(ref)
            if cached is not MISSING:
                return cached
            self.internal_store_object(ref, cast(CachedPdfObject, resolved))
        return resolved

    def internal_cached_object(self, ref: PdfReference) -> object:
        obj_num = ref.object_number
        gen_num = ref.generation_number
        if gen_num == 0 and self.objects_gen0 is not None and obj_num < len(self.objects_gen0):
            cached_gen0 = self.objects_gen0[obj_num]
            if cached_gen0 is not MISSING:
                return cached_gen0
        return self.objects.get(key_for(obj_num, gen_num), MISSING)

    def internal_store_object(self, ref: PdfReference, resolved: CachedPdfObject) -> None:
        obj_num = ref.object_number
        gen_num = ref.generation_number
        if gen_num == 0 and self.objects_gen0 is not None and obj_num < len(self.objects_gen0):
            self.objects_gen0[obj_num] = resolved
        else:
            self.objects[key_for(obj_num, gen_num)] = resolved

    def internal_resolve_reference(self, ref: PdfReference) -> object:
        obj_num = ref.object_number
        gen_num = ref.generation_number
        if obj_num < 0 or gen_num < 0:
            raise ValueError("invalid PDF reference")

        cache_key = key_for(obj_num, gen_num)
        resolved: object
        if gen_num == 0 and self.xref_gen0 is not None:
            if obj_num < len(self.xref_gen0):
                entry = self.xref_gen0[obj_num]
            else:
                entry = None
        else:
            entry = self.xref.get(cache_key)
            if (
                entry is None
                and gen_num != 0
                and self.xref_gen0 is not None
                and obj_num < len(self.xref_gen0)
            ):
                entry = self.xref_gen0[obj_num]

        if (entry is None or not entry.in_use) and self.recover_missing:
            lexer = self.get_lexer()
            try:
                resolved = self.recover_missing_indirect_object(lexer, ref)
            finally:
                self.release_lexer(lexer)
        elif entry is None or not entry.in_use:
            resolved = None
        else:
            if entry.object_stream is not None:
                stream_num = entry.object_stream
                with self.lock:
                    container = self.object_streams.get(stream_num)
                if container is None:
                    stream_obj = self.resolve(PdfReference(stream_num))
                    if type(stream_obj) is PdfStream:
                        candidate = PdfObjectStream(stream_obj, kw_cache=dict(self.kw_cache))
                        with self.lock:
                            container = self.object_streams.setdefault(stream_num, candidate)
                resolved = container.get(obj_num, self.resolve) if container is not None else None
            else:
                lexer = self.get_lexer()
                lexer.rewind(entry.offset)
                try:
                    resolved = lexer.parse_indirect_object()
                except Exception:
                    resolved = self.recover_indirect_object(lexer, entry.offset)
                else:
                    pass
                finally:
                    self.release_lexer(lexer)

        if type(resolved) is PdfStream:
            resolved = self.resolve_stream(resolved)
        return resolved

    def resolve_stream(self, stream: PdfStream) -> PdfStream:
        resolved_dict: dict[object, object] | None = None
        for key, value in stream.dictionary.items():
            if normalize_pdf_name(key) not in STREAM_DECODE_KEYS:
                continue
            resolved_value = self.deep_resolve(value, set())
            if resolved_value is not value:
                if resolved_dict is None:
                    resolved_dict = dict(stream.dictionary)
                resolved_dict[key] = resolved_value
        if resolved_dict is None:
            return stream
        return stream.replace(dictionary=resolved_dict)

    def recover_indirect_object(self, lexer: PdfLexer, offset: int) -> object:
        data = lexer.raw_data
        search_start = max(0, offset - 128)
        search_end = min(len(data), offset + 128)
        marker = find_indirect_object_header(
            data,
            search_start,
            search_end,
            lexer.source_buffer,
        )
        if marker is None:
            raise PdfParseError("expected indirect object header")
        lexer.rewind(marker)
        return lexer.parse_indirect_object()

    def internal_recovery_offsets(self, lexer: PdfLexer) -> dict[int, tuple[int, ...]]:
        """Index indirect-object headers once for damaged-xref recovery."""
        with self.lock:
            if self.recovery_offsets is not None:
                return self.recovery_offsets
        source_buffer = lexer.source_buffer
        data = (
            bytes(lexer.raw_data)
            if source_buffer is None
            else cast(bytes | mmap.mmap, source_buffer)
        )
        offsets: dict[int, list[int]] = {}
        search_pos = 0
        while (marker := data.find(b"obj", search_pos)) >= 0:
            search_pos = marker + 3
            parsed = parse_object_marker_prefix(data, marker)
            if parsed is None:
                continue
            offset, object_number, generation_number = parsed
            key = (object_number << 16) | generation_number
            offsets.setdefault(key, []).append(offset)
        indexed = {key: tuple(values) for key, values in offsets.items()}
        with self.lock:
            if self.recovery_offsets is None:
                self.recovery_offsets = indexed
            return self.recovery_offsets

    def recover_missing_indirect_object(self, lexer: PdfLexer, ref: PdfReference) -> object:
        """Resolve a demanded object omitted by a damaged cross-reference table."""
        key = (ref.object_number << 16) | ref.generation_number
        for offset in reversed(self.internal_recovery_offsets(lexer).get(key, ())):
            lexer.rewind(offset)
            try:
                return lexer.parse_indirect_object()
            except Exception:
                continue
        return None
