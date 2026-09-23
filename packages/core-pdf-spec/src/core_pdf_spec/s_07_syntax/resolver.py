# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import contextlib
import mmap
import threading
from typing import cast

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.objects import PdfObjectStream
from core_pdf_spec.s_07_syntax.resolution import (
    resolve_object_graph,
    resolve_reference_chain,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import (
    CachedPdfObject,
    Decipher,
    ObjectCache,
    PdfDict,
    PdfObject,
)
from core_pdf_spec.s_07_syntax.xref import (
    PdfXRefEntry,
    key_for,
)
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    decoded_name,
    parse_box,
    require_pdf_integer,
    require_pdf_number,
)
from core_pdf_spec.s_07_syntax_primitives.text_string import decode_pdf_text_string
from core_pdf_spec.standards import SemanticContext
from core_pdf_spec.types import MISSING, PdfReference, PdfString

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


class ObjectResolver:
    __slots__ = (
        "data",
        "xref",
        "decipher",
        "objects",
        "object_streams",
        "lock",
        "thread_state",
        "_semantic_context",
    )

    def __init__(
        self,
        data: bytes | bytearray | memoryview | mmap.mmap,
        xref: dict[int, PdfXRefEntry],
        *,
        decipher: Decipher | None = None,
        semantic_context: SemanticContext | None = None,
    ) -> None:
        self.data = memoryview(data)
        self.xref = xref
        self.decipher = decipher
        self._semantic_context = semantic_context
        self.objects: ObjectCache = {}
        self.object_streams: dict[int, PdfObjectStream] = {}
        self.lock = threading.RLock()
        self.thread_state = threading.local()

    @property
    def semantic_context(self) -> SemanticContext | None:
        return self._semantic_context

    @semantic_context.setter
    def semantic_context(self, context: SemanticContext | None) -> None:
        with self.lock:
            if context == self._semantic_context:
                return
            streams = self.detach_parsed_caches()
            self._semantic_context = context
        for stream in streams:
            stream.close()

    def get_lexer(self) -> PdfLexer:
        return PdfLexer(
            self.data,
            reference_resolver=self.resolve,
            decipher=self.decipher,
            semantic_context=self.semantic_context,
        )

    def release_lexer(self, lexer: PdfLexer) -> None:
        lexer.close()

    def detach_parsed_caches(self) -> tuple[PdfObjectStream, ...]:
        streams = tuple(self.object_streams.values())
        self.objects.clear()
        self.object_streams.clear()
        return streams

    def close(self) -> None:
        with self.lock:
            streams = self.detach_parsed_caches()
            self.decipher = None
        for stream in streams:
            stream.close()
        with contextlib.suppress(ValueError):
            self.data.release()
        self.data = memoryview(b"")

    def resolve(self, ref: object) -> PdfObject:
        # This is where an unknown becomes a PDF object: the parser hands in
        # whatever the file contained, and everything downstream is entitled
        # to treat the result as an object of the model.
        if type(ref) is not PdfReference:
            return cast(PdfObject, ref)

        cache_key = key_for(ref.object_number, ref.generation_number)
        with self.lock:
            cached = self.objects.get(cache_key, MISSING)
            if cached is not MISSING:
                return cast(PdfObject, cached)

        resolving = getattr(self.thread_state, "resolving", None)
        if resolving is None:
            resolving = set()
            self.thread_state.resolving = resolving
        if cache_key in resolving:
            return ref

        resolving.add(cache_key)
        try:
            resolved = cast(PdfObject, self.resolve_reference(ref))
        finally:
            resolving.remove(cache_key)

        with self.lock:
            cached = self.objects.get(cache_key, MISSING)
            if cached is not MISSING:
                return cast(PdfObject, cached)
            self.objects[cache_key] = cast(CachedPdfObject, resolved)
        return resolved

    def deep_resolve(self, value: object) -> PdfObject:
        return cast(PdfObject, resolve_object_graph(value, self.resolve))

    def resolve_dict(self, value: object) -> PdfDict | None:
        resolved = self.deep_resolve(value)
        return resolved if isinstance(resolved, dict) else None

    def resolve_box(self, value: object) -> tuple[float, float, float, float] | None:
        resolved = self.deep_resolve(value)
        if resolved is None:
            return None
        box = parse_box(resolved)
        if box is None:
            raise ValueError("invalid box value")
        return box

    def resolve_font_dict(self, font: PdfDict) -> PdfDict:
        has_resources = "Resources" in font
        font_values = (
            {key: value for key, value in font.items() if key != "Resources"}
            if has_resources
            else font
        )
        resolved_font = self.deep_resolve(font_values)
        if not isinstance(resolved_font, dict):
            raise ValueError("invalid font dictionary")
        result = resolved_font
        if has_resources:
            result["Resources"] = font["Resources"]
        return result

    def resolve_float(self, value: object, default: float | None = 0.0) -> float | None:
        resolved = resolve_reference_chain(value, self.resolve)
        return default if resolved is None else require_pdf_number(resolved)

    def resolve_name(self, value: object) -> str | None:
        return decoded_name(resolve_reference_chain(value, self.resolve))

    def resolve_int(self, value: object, default: int | None = None) -> int | None:
        resolved = resolve_reference_chain(value, self.resolve)
        return default if resolved is None else require_pdf_integer(resolved)

    def resolve_str(self, value: object) -> str | None:
        if type(value) is str:
            return value

        resolved = resolve_reference_chain(value, self.resolve)
        if isinstance(resolved, PdfString):
            return self.decode_text(resolved.data)
        if isinstance(resolved, bytes):
            return self.decode_text(resolved)
        return resolved if isinstance(resolved, str) else None

    def resolve_reference(self, ref: PdfReference) -> object:
        entry = self.xref_entry(ref)
        if entry is None or not entry.in_use:
            resolved = self.missing_object(ref)
        elif entry.object_stream is not None:
            resolved = self.load_compressed_object(ref, entry)
        else:
            lexer = self.get_lexer()
            try:
                resolved = self.load_indirect_object(lexer, entry.offset, expected_reference=ref)
            finally:
                self.release_lexer(lexer)
        return self.resolve_stream(resolved) if type(resolved) is PdfStream else resolved

    def get_object_stream(self, stream_number: int) -> PdfObjectStream | None:
        with self.lock:
            container = self.object_streams.get(stream_number)
        if container is not None:
            return container
        stream = self.resolve(PdfReference(stream_number))
        if type(stream) is not PdfStream:
            return None
        candidate = self.create_object_stream(stream)
        with self.lock:
            container = self.object_streams.setdefault(stream_number, candidate)
        if container is not candidate:
            candidate.close()
        return container

    def load_compressed_object(self, ref: PdfReference, entry: PdfXRefEntry) -> object:
        stream_number = entry.object_stream
        index = entry.index_in_stream
        if (
            ref.generation_number != 0
            or type(stream_number) is not int
            or stream_number <= 0
            or type(index) is not int
            or index < 0
        ):
            raise PdfParseError("invalid compressed object cross-reference entry")
        container = self.get_object_stream(stream_number)
        if container is None:
            raise PdfParseError("compressed object container is not a stream")
        return container.get_at_index(index, expected_reference=ref)

    def resolve_stream(self, stream: PdfStream) -> PdfStream:
        selected = {
            key: value
            for key, value in stream.dictionary.items()
            if decoded_name(key) in STREAM_DECODE_KEYS
        }
        if not selected:
            return stream
        resolved = self.deep_resolve(selected)
        if resolved is selected:
            return stream
        dictionary = dict(stream.dictionary)
        dictionary.update(cast(PdfDict, resolved))
        return stream.replace(dictionary=dictionary)

    def xref_entry(self, ref: PdfReference) -> PdfXRefEntry | None:
        return self.xref.get(key_for(ref.object_number, ref.generation_number))

    def missing_object(self, ref: PdfReference) -> object:
        return None

    def create_object_stream(self, stream: PdfStream) -> PdfObjectStream:
        return PdfObjectStream(stream, semantic_context=self.semantic_context)

    def load_indirect_object(
        self, lexer: PdfLexer, offset: int, *, expected_reference: PdfReference
    ) -> object:
        lexer.rewind(offset)
        return lexer.parse_indirect_object(expected_reference=expected_reference)

    def decode_text(self, data: bytes) -> str:
        return decode_pdf_text_string(data, context=self.semantic_context)


__all__ = ("ObjectResolver",)
