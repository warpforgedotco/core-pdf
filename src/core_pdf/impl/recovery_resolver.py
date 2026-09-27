# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, overload

from core_pdf.impl.exceptions import PdfDecryptionError, PdfParseError, PdfUnsupportedError
from core_pdf.impl.graphics_stream_decoding import decode_stream_data
from core_pdf.impl.pdf_values import (
    decode_pdf_text_string,
    lenient_float,
    lenient_int,
    recover_pdf_name,
)
from core_pdf.impl.recovery_lexer import PdfLexer
from core_pdf.impl.recovery_objects import PdfObjectStream
from core_pdf.impl.recovery_xref import iter_indirect_object_headers
from core_pdf.impl.types import MISSING, PdfReference, PdfString
from core_pdf_spec.s_07_filters.pipeline import decode_stream_data as decode_spec_stream_data
from core_pdf_spec.s_07_syntax.lexer import PdfLexer as SyntaxLexer
from core_pdf_spec.s_07_syntax.objects import PdfObjectStream as SyntaxObjectStream
from core_pdf_spec.s_07_syntax.resolution import resolve_reference_chain
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver as SyntaxResolver
from core_pdf_spec.s_07_syntax.resources import resolve_resource_dict as resolve_spec_resources
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfArray, PdfDict, PdfObject, PdfValueResolver
from core_pdf_spec.s_07_syntax.xref import (
    PdfXRefEntry,
    key_for,
)
from core_pdf_spec.s_07_syntax_primitives.tokens import LexicalRules

LEXER_POOL_LIMIT = 8


class ObjectResolver(SyntaxResolver):
    __slots__ = ("lexer_pool", "parsed_objects", "parsed_rules")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.lexer_pool: list[PdfLexer] = []
        self.parsed_objects: dict[int, object] = {}
        self.parsed_rules: LexicalRules | None = None
        super().__init__(*args, **kwargs)

    def adopt_parsed_objects(self, objects: dict[int, object], rules: LexicalRules) -> None:
        with self.lock:
            self.parsed_objects = objects
            self.parsed_rules = rules

    def close(self) -> None:
        with self.lock:
            self.parsed_objects = {}
        super().close()

    def xref_entry(self, ref: PdfReference) -> PdfXRefEntry | None:
        entry = super().xref_entry(ref)
        if entry is None and ref.generation_number != 0:
            entry = self.xref.get(key_for(ref.object_number, 0))
        return entry

    def create_object_stream(self, stream: PdfStream) -> PdfObjectStream:
        return PdfObjectStream(stream, semantic_context=self.semantic_context)

    def load_indirect_object(
        self, lexer: SyntaxLexer, offset: int, *, expected_reference: PdfReference
    ) -> object:
        if (
            self.parsed_objects
            and lexer.decipher is None
            and type(lexer) is PdfLexer
            and lexer.lexical_rules == self.parsed_rules
        ):
            with self.lock:
                parsed = self.parsed_objects.pop(offset, MISSING)
            if parsed is not MISSING:
                return parsed
        try:
            lexer.rewind(offset)
            return lexer.parse_indirect_object()
        except PdfDecryptionError, PdfUnsupportedError:
            raise
        except Exception:
            return self.recover_indirect_object(lexer, offset)

    def load_compressed_object(self, ref: PdfReference, entry: PdfXRefEntry) -> object:
        stream_number = entry.object_stream
        if stream_number is None:
            return None
        container = self.get_object_stream(stream_number)
        return None if container is None else container.get(ref.object_number)

    def resolve_stream(self, stream: PdfStream) -> PdfStream:
        stream = super().resolve_stream(stream)
        return (
            stream.replace(decoder=decode_stream_data)
            if stream.decoder is decode_spec_stream_data
            else stream
        )

    def get_lexer(self) -> PdfLexer:
        pool = self.lexer_pool
        with self.lock:
            lexer = pool.pop() if pool else None
        if lexer is None:
            return PdfLexer(
                self.data,
                reference_resolver=self.resolve,
                decipher=self.decipher,
                semantic_context=self.semantic_context,
            )
        lexer.reference_resolver = self.resolve
        lexer.decipher = self.decipher
        if lexer.semantic_context is not self.semantic_context:
            lexer.semantic_context = self.semantic_context
        return lexer

    def release_lexer(self, lexer: SyntaxLexer) -> None:
        if type(lexer) is PdfLexer and lexer.raw_data.obj is self.data.obj:
            with self.lock:
                if len(self.lexer_pool) < LEXER_POOL_LIMIT:
                    self.lexer_pool.append(lexer)
                    return
        lexer.close()

    def detach_parsed_caches(self) -> tuple[SyntaxObjectStream, ...]:
        with self.lock:
            lexers, self.lexer_pool[:] = list(self.lexer_pool), []
        for lexer in lexers:
            lexer.close()
        return super().detach_parsed_caches()

    def recover_indirect_object(self, lexer: SyntaxLexer, offset: int) -> object:
        data = lexer.raw_data
        search_start = max(0, offset - 128)
        search_end = min(len(data), offset + 128)
        header = next(
            iter_indirect_object_headers(
                data, search_start, search_end, source_buffer=lexer.source_buffer
            ),
            None,
        )
        if header is None:
            raise PdfParseError("expected indirect object header")
        lexer.rewind(header[0])
        return lexer.parse_indirect_object()

    def resolve_or_none(self, value: object) -> PdfObject | None:
        try:
            return self.resolve(value)
        except Exception:
            return None

    def resolve_name_or_text(self, value: object, *, name_like: bool = False) -> str | None:
        text = self.resolve_name(value)
        if text is None and name_like:
            text = self.resolve_name_like_value(value)
        return text or self.resolve_str(value)

    def resolve_name_like_value(self, resolved: object) -> str | None:
        val = self.resolve(resolved)
        name = recover_pdf_name(val)
        if name is not None:
            return name
        if type(val) is PdfString:
            return self.decode_text(val.data)
        return None

    def resolve_float(self, value: object, default: float | None = 0.0) -> float | None:
        return lenient_float(self.resolve(value), default)

    def resolve_name(self, value: object) -> str | None:
        return recover_pdf_name(resolve_reference_chain(value, self.resolve))

    def resolve_int(self, value: object, default: int | None = None) -> int | None:
        return lenient_int(self.resolve(value), default)

    def as_dict(self, value: object) -> PdfDict | None:
        resolved = self.resolve(value)
        return resolved if isinstance(resolved, dict) else None

    def dict_at(self, container: Mapping[Any, object], key: object) -> PdfDict | None:
        resolved = self.resolve(container.get(key))
        return resolved if isinstance(resolved, dict) else None

    def array_at(self, container: Mapping[Any, object], key: object) -> PdfArray | None:
        resolved = self.resolve(container.get(key))
        return resolved if isinstance(resolved, list) else None

    @overload
    def int_at(
        self, container: Mapping[Any, object], key: object, default: None = None
    ) -> int | None: ...

    @overload
    def int_at(self, container: Mapping[Any, object], key: object, default: int) -> int: ...

    def int_at(
        self, container: Mapping[Any, object], key: object, default: int | None = None
    ) -> int | None:
        return lenient_int(self.resolve(container.get(key)), default)

    @overload
    def float_at(
        self, container: Mapping[Any, object], key: object, default: None
    ) -> float | None: ...

    @overload
    def float_at(
        self, container: Mapping[Any, object], key: object, default: float = 0.0
    ) -> float: ...

    def float_at(
        self, container: Mapping[Any, object], key: object, default: float | None = 0.0
    ) -> float | None:
        return lenient_float(self.resolve(container.get(key)), default)

    def name_at(self, container: Mapping[Any, object], key: object) -> str | None:
        return self.resolve_name(container.get(key))

    def resolve_box(
        self, value: object, *, python_syntax: bool = True
    ) -> tuple[float, float, float, float] | None:
        return super().resolve_box(value, python_syntax=python_syntax)

    def decode_text(self, data: bytes) -> str:
        return decode_pdf_text_string(data)


def resolve_resource_dict(value: object, resolver: PdfValueResolver) -> PdfDict | None:
    try:
        return resolve_spec_resources(value, resolver)
    except PdfParseError:
        return None
