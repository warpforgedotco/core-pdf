from __future__ import annotations

from typing import Any

from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_07_syntax.objects import PdfObjectStream
from core_pdf.impl.engine.spec.s_07_syntax.primitives import (
    MISSING,
    PdfName,
    PdfReference,
    PdfStream,
    PdfString,
    parse_float,
    parse_int,
    parse_name,
)
from core_pdf.impl.engine.spec.s_07_syntax.xref import PdfXRefEntry
from core_pdf.impl.engine.spec.s_09_fonts.encoding import decode_pdf_text_string


class ObjectResolver:
    __slots__ = (
        "data",
        "xref",
        "xref_gen0",
        "trailer",
        "decipher",
        "objects",
        "objects_gen0",
        "object_streams",
        "resolving",
        "deep_cache",
        "kw_cache",
        "lexer_stack",
    )

    def __init__(
        self,
        data: bytes | memoryview,
        xref: dict[int, PdfXRefEntry],
        trailer: dict[str, Any],
        decipher: Any = None,
    ) -> None:
        self.data = memoryview(data) if not isinstance(data, memoryview) else data
        self.xref = xref
        self.trailer = trailer
        self.decipher = decipher
        self.objects: dict[int, Any] = {}

        # Optimized O(1) array-based cache for generation 0 objects and xrefs
        max_obj = 0
        if self.xref:
            # key is (obj_num << 16 | gen_num)
            for k in self.xref:
                obj_num = k >> 16
                if obj_num > max_obj:
                    max_obj = obj_num

        if max_obj < 1000000:
            self.objects_gen0: list[Any] | None = [MISSING] * (max_obj + 1)
            self.xref_gen0: list[PdfXRefEntry | None] | None = [None] * (max_obj + 1)
            for k, entry in self.xref.items():
                if (k & 0xFFFF) == 0:
                    self.xref_gen0[k >> 16] = entry
        else:
            self.objects_gen0 = None
            self.xref_gen0 = None

        self.object_streams: dict[int, PdfObjectStream] = {}
        self.resolving: set[int] = set()
        self.deep_cache: dict[int, Any] = {}
        self.kw_cache: dict[bytes, str] = {}
        self.lexer_stack: list[PdfLexer] = []

    def get_lexer(self) -> PdfLexer:
        if self.lexer_stack:
            return self.lexer_stack.pop()
        return PdfLexer(
            self.data,
            reference_resolver=self.resolve,
            decipher=self.decipher,
            kw_cache=self.kw_cache,
        )

    def release_lexer(self, lexer: PdfLexer) -> None:
        self.lexer_stack.append(lexer)

    def resolve(self, ref: PdfReference | Any) -> Any:
        if type(ref) is not PdfReference:
            return ref

        # Use integer key to avoid tuple allocation
        obj_num = ref.object_number
        gen_num = ref.generation_number
        if obj_num < 0 or gen_num < 0:
            raise ValueError("invalid PDF reference")

        # FAST PATH: O(1) Array Lookup for gen-0
        if gen_num == 0 and self.objects_gen0 is not None and obj_num < len(self.objects_gen0):
            resolved = self.objects_gen0[obj_num]
            if resolved is not MISSING:
                return resolved

        cache_key = (obj_num << 16) | gen_num
        resolved = self.objects.get(cache_key, MISSING)
        if resolved is not MISSING:
            return resolved

        if cache_key in self.resolving:
            return ref  # Break cycle

        self.resolving.add(cache_key)
        try:
            # FAST PATH: O(1) XRef Lookup for gen-0
            if gen_num == 0 and self.xref_gen0 is not None:
                if obj_num < len(self.xref_gen0):
                    entry = self.xref_gen0[obj_num]
                else:
                    entry = None
            else:
                entry = self.xref.get(cache_key)

            if entry is None or not entry.in_use:
                resolved = None
            else:
                if entry.object_stream is not None:
                    # Compressed object (Type 2)
                    stream_num = entry.object_stream
                    container = self.object_streams.get(stream_num)
                    if container is None:
                        stream_obj = self.resolve(PdfReference(stream_num))
                        if type(stream_obj) is PdfStream:
                            container = PdfObjectStream(stream_obj, kw_cache=self.kw_cache)
                            self.object_streams[stream_num] = container
                    resolved = (
                        container.get(obj_num, self.resolve) if container is not None else None
                    )
                else:
                    # Normal object (Type 1)
                    lexer = self.get_lexer()
                    lexer.rewind(entry.offset)
                    try:
                        resolved = lexer.parse_indirect_object()
                    finally:
                        self.release_lexer(lexer)

            if gen_num == 0 and self.objects_gen0 is not None:
                if obj_num < len(self.objects_gen0):
                    self.objects_gen0[obj_num] = resolved
            else:
                self.objects[cache_key] = resolved
            return resolved
        finally:
            self.resolving.remove(cache_key)

    def deep_resolve(self, value: Any, seen: set[int] | None = None) -> Any:
        """
        Recursively resolves all references within an object.
        """
        # FAST PATH: primitives (most common case in traversal)
        t = type(value)
        if t is int or t is float or t is str or t is bool or value is None:
            return value

        if t is PdfReference:
            # Special handling for reference to avoid recursion overhead
            res = self.resolve(value)
            if type(res) in (dict, list, PdfStream, tuple, PdfReference):
                return self.deep_resolve(res, seen)
            return res

        if t not in (dict, list, tuple, PdfStream):
            # Probably PdfName or other primitive wrapper
            if t is PdfName:
                return value
            return value

        val_id = id(value)
        cached = self.deep_cache.get(val_id, MISSING)
        if cached is not MISSING:
            return cached

        if seen is None:
            seen = set()

        if type(value) is PdfStream:
            res = value.replace(dictionary=self.deep_resolve(value.dictionary, seen))
            self.deep_cache[val_id] = res
            return res

        marker = id(value)
        if marker in seen:
            return value
        seen.add(marker)

        if t is list or t is tuple:
            res = [self.deep_resolve(item, seen) for item in value]
            self.deep_cache[val_id] = res
            return res

        if t is dict:
            res = {key: self.deep_resolve(item, seen) for key, item in value.items()}
            self.deep_cache[val_id] = res
            return res

        return value

    def resolve_dict(self, value: Any) -> dict[str, Any] | None:
        resolved = self.deep_resolve(value)
        return resolved if isinstance(resolved, dict) else None

    def resolve_box(self, value: Any) -> tuple[float, float, float, float] | None:
        """Helper to resolve a box reference."""
        resolved = self.deep_resolve(value)
        if resolved is None:
            return None
        if isinstance(resolved, (list, tuple)) and len(resolved) == 4:
            try:
                return (
                    float(resolved[0]),
                    float(resolved[1]),
                    float(resolved[2]),
                    float(resolved[3]),
                )
            except TypeError, ValueError:
                raise ValueError("invalid box value")
        raise ValueError("invalid box value")

    def resolve_font_dict(self, font: dict[str, Any]) -> dict[str, Any]:
        """Helper to resolve font dictionary references."""
        resolved_font = self.deep_resolve(font)
        if not isinstance(resolved_font, dict):
            raise ValueError("invalid font dictionary")
        return resolved_font

    def resolve_list(self, value: Any) -> list[Any] | None:
        resolved = self.deep_resolve(value)
        return resolved if isinstance(resolved, list) else None

    def resolve_float(self, value: Any, default: float = 0.0) -> float:
        """Helper to resolve a float reference."""
        if isinstance(value, (int, float)):
            return float(value)
        return parse_float(self.resolve(value), default=default)

    def resolve_name(self, value: Any) -> str | None:
        if type(value) is PdfName:
            return value.value
        if type(value) is str:
            return value
        return parse_name(self.resolve(value))

    def resolve_name_like_value(self, resolved: Any) -> str | None:
        val = self.resolve(resolved)
        name = parse_name(val)
        if name is not None:
            return name
        if isinstance(val, PdfString):
            return decode_pdf_text_string(val.data)
        return None

    def resolve_int(self, value: Any, default: int | None = None) -> int | None:
        if type(value) is int:
            return value
        return parse_int(self.resolve(value), default)

    def resolve_str(self, value: Any) -> str | None:
        """Helper to resolve a string reference."""
        # Fast path for string
        if type(value) is str:
            return value

        resolved = self.deep_resolve(value)
        if isinstance(resolved, PdfString):
            return decode_pdf_text_string(resolved.data)
        if isinstance(resolved, bytes):
            return decode_pdf_text_string(resolved)
        if type(resolved) is str:
            return resolved
        return None
