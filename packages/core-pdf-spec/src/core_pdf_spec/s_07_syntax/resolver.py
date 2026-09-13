# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve indirect references against the cross-reference table."""

from __future__ import annotations

import contextlib
import mmap
import threading
from typing import cast

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.objects import PdfObjectStream
from core_pdf_spec.s_07_syntax.resolution import (
    internal_resolve_object_graph,
    resolve_reference_chain,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import (
    CachedPdfObject,
    Decipher,
    ObjectCache,
    PdfDict,
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
        "internal_semantic_context",
    )

    def __init__(
        self,
        data: bytes | bytearray | memoryview | mmap.mmap,
        xref: dict[int, PdfXRefEntry],
        *,
        decipher: Decipher | None = None,
        semantic_context: SemanticContext | None = None,
    ) -> None:
        # Keep an owned view.  Reusing the caller's memoryview lets a temporary
        # resolver.close() release the document's source buffer underneath
        # concurrent readers.
        self.data = memoryview(data)
        self.xref = xref
        self.decipher = decipher
        self.internal_semantic_context = semantic_context
        self.objects: ObjectCache = {}
        self.object_streams: dict[int, PdfObjectStream] = {}
        self.lock = threading.RLock()
        self.thread_state = threading.local()

    @property
    def semantic_context(self) -> SemanticContext | None:
        return self.internal_semantic_context

    @semantic_context.setter
    def semantic_context(self, context: SemanticContext | None) -> None:
        """Select semantics before parsing or after bootstrap, clearing parsed caches.

        Names and dictionary keys depend on the version as well as text strings.
        Previously returned objects retain their values; subsequent resolutions
        use the new context. Callers must finish active parsing before changing it.
        """
        with self.lock:
            if context == self.internal_semantic_context:
                return
            streams = self.internal_detach_parsed_caches()
            self.internal_semantic_context = context
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

    def internal_detach_parsed_caches(self) -> tuple[PdfObjectStream, ...]:
        """Detach owned parsers while the caller holds the resolver lock."""
        streams = tuple(self.object_streams.values())
        self.objects.clear()
        self.object_streams.clear()
        return streams

    def close(self) -> None:
        """Release owned parsing resources after active resolution has finished."""
        with self.lock:
            streams = self.internal_detach_parsed_caches()
            self.decipher = None
        for stream in streams:
            stream.close()
        with contextlib.suppress(ValueError):
            self.data.release()
        self.data = memoryview(b"")

    def resolve(self, ref: object) -> object:
        if type(ref) is not PdfReference:
            return ref

        cache_key = key_for(ref.object_number, ref.generation_number)
        with self.lock:
            cached = self.objects.get(cache_key, MISSING)
            if cached is not MISSING:
                return cached

        resolving = getattr(self.thread_state, "resolving", None)
        if resolving is None:
            resolving = set()
            self.thread_state.resolving = resolving
        if cache_key in resolving:
            return ref

        resolving.add(cache_key)
        try:
            resolved = self.internal_resolve_reference(ref)
        finally:
            resolving.remove(cache_key)

        with self.lock:
            cached = self.objects.get(cache_key, MISSING)
            if cached is not MISSING:
                return cached
            self.objects[cache_key] = cast(CachedPdfObject, resolved)
        return resolved

    def deep_resolve(self, value: object) -> object:
        """Resolve an object graph, preserving sharing, cycles, and unchanged identity."""
        return internal_resolve_object_graph(value, self.resolve)

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
        # Type 3 fonts carry their own resources. Those remain demand-driven,
        # while the decoder needs the font's metrics and character programs.
        has_resources = "Resources" in font
        font_values = (
            {key: value for key, value in font.items() if key != "Resources"}
            if has_resources
            else font
        )
        resolved_font = self.deep_resolve(font_values)
        if not isinstance(resolved_font, dict):
            raise ValueError("invalid font dictionary")
        result = cast(PdfDict, resolved_font)
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

        # Resolve only an indirect scalar chain. ISO 32000-1:2008 and
        # ISO 32000-2:2020, 12.3.2.2 allow a GoTo destination to be an array
        # beginning with an indirect page reference. Deep-resolving such an
        # array merely to decide whether it is a string walks the page graph.
        resolved = resolve_reference_chain(value, self.resolve)
        if isinstance(resolved, PdfString):
            return self.decode_text(resolved.data)
        if isinstance(resolved, bytes):
            return self.decode_text(resolved)
        return resolved if isinstance(resolved, str) else None

    def internal_resolve_reference(self, ref: PdfReference) -> object:
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
        """Get a cached parser, or None when the referenced value is not a stream.

        Parser construction and decoding failures propagate. The caller decides
        whether a missing container is a structural error or recoverable damage.
        """
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
        """Validate an in-use compressed entry against its object-stream header."""
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
        dictionary.update(cast(dict[object, object], resolved))
        return stream.replace(dictionary=dictionary)

    def xref_entry(self, ref: PdfReference) -> PdfXRefEntry | None:
        """Look up the exact object number and generation."""
        return self.xref.get(key_for(ref.object_number, ref.generation_number))

    def missing_object(self, ref: PdfReference) -> object:
        """Return the prescribed null value for an undefined indirect reference."""
        # ISO 32000-2, 7.3.9: undefined indirect references denote null.
        return None

    def create_object_stream(self, stream: PdfStream) -> PdfObjectStream:
        """Create an object-stream parser; malformed stream errors propagate."""
        return PdfObjectStream(stream, semantic_context=self.semantic_context)

    def load_indirect_object(
        self, lexer: PdfLexer, offset: int, *, expected_reference: PdfReference
    ) -> object:
        """Read the demanded object, validating its identity before decoding its body."""
        lexer.rewind(offset)
        return lexer.parse_indirect_object(expected_reference=expected_reference)

    def decode_text(self, data: bytes) -> str:
        """Decode under semantic_context; applications may override reader recovery."""
        return decode_pdf_text_string(data, context=self.semantic_context)


__all__ = ("ObjectResolver",)
