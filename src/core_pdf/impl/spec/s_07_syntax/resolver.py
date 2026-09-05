# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve indirect references against the cross-reference table."""

from __future__ import annotations

import contextlib
import mmap
import threading
from typing import cast

from core_pdf.impl.exceptions import PdfDecryptionError, PdfParseError, PdfUnsupportedError
from core_pdf.impl.primitives import MISSING, PdfName, PdfReference, PdfString
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.objects import PdfObjectStream
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import (
    CachedPdfObject,
    Decipher,
    ObjectCache,
    PdfDict,
    PdfObject,
)
from core_pdf.impl.spec.s_07_syntax.xref import (
    PdfXRefEntry,
    key_for,
    parse_object_marker_prefix,
)
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
    parse_box,
    parse_float,
    parse_int,
    parse_text_string,
)
from core_pdf.impl.spec.s_07_syntax_primitives.scanning import (
    FindableSizedBuffer,
    full_source_buffer,
)
from core_pdf.impl.spec.s_07_syntax_primitives.text_string import decode_pdf_text_string

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

TERMINAL_TYPES = {int, float, str, bool, type(None), PdfName, bytes}


def internal_find_indirect_object_header(
    data: memoryview,
    search_start: int,
    search_end: int,
    source_buffer: FindableSizedBuffer | None = None,
) -> int | None:
    """Find a complete ``N G obj`` header within a recovery search window."""
    data_len = len(data)
    search_start = max(0, search_start)
    search_end = min(data_len, search_end)
    source = source_buffer if source_buffer is not None else full_source_buffer(data, data_len)
    copied_region = data[search_start:search_end].tobytes() if source is None else None
    pos = search_start
    while pos < search_end:
        if source is not None:
            marker = source.find(b"obj", pos, search_end)
        else:
            assert copied_region is not None
            marker = copied_region.find(b"obj", pos - search_start)
        if marker < 0:
            return None
        if source is None:
            marker += search_start
        parsed = parse_object_marker_prefix(data, marker)
        if parsed is not None and parsed[0] >= search_start:
            return parsed[0]
        pos = marker + 3
    return None


class ObjectResolver:
    __slots__ = (
        "data",
        "xref",
        "trailer",
        "decipher",
        "objects",
        "object_streams",
        "lock",
        "thread_state",
        "recover_missing",
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
        self.objects: ObjectCache = {}
        self.object_streams: dict[int, PdfObjectStream] = {}
        self.lock = threading.RLock()
        self.thread_state = threading.local()

    def get_lexer(self) -> PdfLexer:
        return PdfLexer(
            self.data,
            reference_resolver=self.resolve,
            decipher=self.decipher,
        )

    def release_lexer(self, lexer: PdfLexer) -> None:
        lexer.close()

    def close(self) -> None:
        self.objects.clear()
        self.object_streams.clear()
        self.decipher = None
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

    def deep_resolve(
        self,
        value: object,
        seen: set[int] | None = None,
        internal_memo: dict[int, tuple[object, object]] | None = None,
    ) -> object:
        """Resolve a graph with cycle detection and operation-local sharing."""
        t = type(value)
        terminal_types = TERMINAL_TYPES
        if t in terminal_types:
            return value

        if t is PdfReference:
            # Walk the chain iteratively with its own seen set, the way
            # resolve_str does. Recursing through deep_resolve instead never
            # recorded the reference keys -- `seen` is only populated for
            # containers below -- so a cyclic chain (1 0 R -> 2 0 R -> 1 0 R)
            # recursed until RecursionError. A cycle yields the reference
            # unresolved, matching what resolve() itself returns for one.
            res: object = value
            chain: set[int] = set()
            while type(res) is PdfReference:
                reference_key = key_for(res.object_number, res.generation_number)
                if reference_key in chain:
                    return res
                chain.add(reference_key)
                res = self.resolve(res)
            if type(res) in (dict, list, PdfStream, tuple):
                return self.deep_resolve(res, seen, internal_memo)
            return res

        if t not in (dict, list, tuple, PdfStream):
            return value

        val_id = id(value)
        if internal_memo is None:
            internal_memo = {}
        cached = internal_memo.get(val_id)
        if cached is not None and cached[0] is value:
            return cached[1]
        if seen is None:
            seen = set()
        if val_id in seen:
            return value
        seen.add(val_id)
        try:
            if t is PdfStream:
                stream = cast(PdfStream, value)
                resolved_dict = self.deep_resolve(stream.dictionary, seen, internal_memo)
                resolved_stream = (
                    stream
                    if resolved_dict is stream.dictionary
                    else stream.replace(dictionary=resolved_dict)
                )
                internal_memo[val_id] = (value, resolved_stream)
                return resolved_stream
            if t is list:
                items = cast(list[object], value)
                resolved = [self.deep_resolve(item, seen, internal_memo) for item in items]
                result: object = (
                    items if all(a is b for a, b in zip(items, resolved, strict=True)) else resolved
                )
                internal_memo[val_id] = (value, result)
                return result
            if t is dict:
                mapping = cast(PdfDict, value)
                resolved_mapping = {
                    key: cast(PdfObject, self.deep_resolve(item, seen, internal_memo))
                    for key, item in mapping.items()
                }
                result = (
                    mapping
                    if all(resolved_mapping[key] is item for key, item in mapping.items())
                    else resolved_mapping
                )
                internal_memo[val_id] = (value, result)
                return result
            result = [
                self.deep_resolve(item, seen, internal_memo)
                for item in cast(tuple[object, ...], value)
            ]
            internal_memo[val_id] = (value, result)
            return result
        finally:
            seen.remove(val_id)

    def resolve_dict(self, value: object) -> PdfDict | None:
        resolved = self.deep_resolve(value)
        return cast(PdfDict, resolved) if isinstance(resolved, dict) else None

    def resolve_box(self, value: object) -> tuple[float, float, float, float] | None:
        resolved = self.deep_resolve(value)
        if resolved is None:
            return None
        box = parse_box(resolved)
        if box is None:
            raise ValueError("invalid box value")
        return box

    def resolve_font_dict(self, font: PdfDict) -> PdfDict:
        resolved_font = self.deep_resolve(font)
        if not isinstance(resolved_font, dict):
            raise ValueError("invalid font dictionary")
        return cast(PdfDict, resolved_font)

    def resolve_float(self, value: object, default: float | None = 0.0) -> float | None:
        if type(value) is int:
            return float(value)
        if type(value) is float:
            return value
        if type(value) is bool:
            return default
        return parse_float(self.resolve(value), default=default)

    def resolve_name(self, value: object) -> str | None:
        return normalize_pdf_name(value) or normalize_pdf_name(self.resolve(value))

    def resolve_name_like_value(self, resolved: object) -> str | None:
        val = self.resolve(resolved)
        name = normalize_pdf_name(val)
        if name is not None:
            return name
        if type(val) is PdfString:
            return decode_pdf_text_string(val.data)
        return None

    def resolve_name_or_text(self, value: object, *, name_like: bool = False) -> str | None:
        """A value as a name, falling back to a text string.

        ``name_like`` also accepts a non-name value whose text is a valid name,
        which lenient readers allow for AcroForm field types.
        """
        text = self.resolve_name(value)
        if text is None and name_like:
            text = self.resolve_name_like_value(value)
        return text or self.resolve_str(value)

    def resolve_int(self, value: object, default: int | None = None) -> int | None:
        if type(value) is int:
            return value
        return parse_int(self.resolve(value), default)

    def resolve_str(self, value: object) -> str | None:
        if type(value) is str:
            return value

        # Resolve only an indirect scalar chain. ISO 32000-1:2008 and
        # ISO 32000-2:2020, 12.3.2.2 allow a GoTo destination to be an array
        # beginning with an indirect page reference. Deep-resolving such an
        # array merely to decide whether it is a string walks the page graph.
        resolved = value
        seen: set[int] = set()
        while type(resolved) is PdfReference:
            reference = resolved
            reference_key = key_for(reference.object_number, reference.generation_number)
            if reference_key in seen:
                return None
            seen.add(reference_key)
            resolved = self.resolve(reference)
        return parse_text_string(resolved)

    def internal_cached_object(self, ref: PdfReference) -> object:
        return self.objects.get(key_for(ref.object_number, ref.generation_number), MISSING)

    def internal_store_object(self, ref: PdfReference, resolved: CachedPdfObject) -> None:
        self.objects[key_for(ref.object_number, ref.generation_number)] = resolved

    def internal_resolve_reference(self, ref: PdfReference) -> object:
        obj_num = ref.object_number
        gen_num = ref.generation_number
        if obj_num < 0 or gen_num < 0:
            raise ValueError("invalid PDF reference")

        cache_key = key_for(obj_num, gen_num)
        resolved: object
        entry = self.xref.get(cache_key)
        if entry is None and gen_num != 0:
            entry = self.xref.get(key_for(obj_num, 0))

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
                        candidate = PdfObjectStream(stream_obj)
                        with self.lock:
                            container = self.object_streams.setdefault(stream_num, candidate)
                resolved = container.get(obj_num) if container is not None else None
            else:
                lexer = self.get_lexer()
                lexer.rewind(entry.offset)
                try:
                    resolved = lexer.parse_indirect_object()
                except (PdfDecryptionError, PdfUnsupportedError):
                    raise
                except Exception:
                    resolved = self.recover_indirect_object(lexer, entry.offset)
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
        marker = internal_find_indirect_object_header(
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
        """Find indirect-object headers for one damaged-xref recovery."""
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
            key = key_for(object_number, generation_number)
            offsets.setdefault(key, []).append(offset)
        return {key: tuple(values) for key, values in offsets.items()}

    def recover_missing_indirect_object(self, lexer: PdfLexer, ref: PdfReference) -> object:
        """Resolve a demanded object omitted by a damaged cross-reference table."""
        key = key_for(ref.object_number, ref.generation_number)
        for offset in reversed(self.internal_recovery_offsets(lexer).get(key, ())):
            lexer.rewind(offset)
            try:
                return lexer.parse_indirect_object()
            except (PdfDecryptionError, PdfUnsupportedError):
                raise
            except Exception:
                continue
        return None
