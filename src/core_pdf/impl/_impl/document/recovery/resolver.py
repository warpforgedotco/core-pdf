# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve indirect references against the cross-reference table."""

from __future__ import annotations

import mmap

from core_pdf.impl._impl.document.recovery.lexer import PdfLexer
from core_pdf.impl._impl.document.recovery.objects import PdfObjectStream
from core_pdf.impl._impl.document.recovery.text_strings import decode_pdf_text_string
from core_pdf.impl._impl.document.recovery.xref import iter_indirect_object_headers
from core_pdf.impl._impl.graphics.stream_decoding import decode_stream_data
from core_pdf.impl._impl.runtime.scalars import parse_box, parse_float, parse_int
from core_pdf.impl.exceptions import PdfDecryptionError, PdfParseError, PdfUnsupportedError
from core_pdf.impl.spec.s_07_filters.pipeline import decode_stream_data as decode_spec_stream_data
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer as SyntaxLexer
from core_pdf.impl.spec.s_07_syntax.resolver import ObjectResolver as SyntaxResolver
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import (
    Decipher,
    PdfDict,
)
from core_pdf.impl.spec.s_07_syntax.xref import (
    PdfXRefEntry,
    key_for,
)
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf.impl.types import PdfReference, PdfString


class ObjectResolver(SyntaxResolver):
    __slots__ = ("recover_missing",)

    def __init__(
        self,
        data: bytes | bytearray | memoryview | mmap.mmap,
        xref: dict[int, PdfXRefEntry],
        trailer: PdfDict,
        decipher: Decipher | None = None,
        *,
        recover_missing: bool = False,
    ) -> None:
        super().__init__(data, xref, trailer, decipher)
        self.recover_missing = recover_missing

    def internal_xref_entry(self, ref: PdfReference) -> PdfXRefEntry | None:
        entry = super().internal_xref_entry(ref)
        if entry is None and ref.generation_number != 0:
            entry = self.xref.get(key_for(ref.object_number, 0))
        return entry

    def internal_missing_object(self, ref: PdfReference) -> object:
        if not self.recover_missing:
            return None
        lexer = self.get_lexer()
        try:
            return self.recover_missing_indirect_object(lexer, ref)
        finally:
            self.release_lexer(lexer)

    def internal_object_stream(self, stream: PdfStream) -> PdfObjectStream:
        return PdfObjectStream(stream)

    def internal_load_indirect_object(self, lexer: SyntaxLexer, offset: int) -> object:
        try:
            return super().internal_load_indirect_object(lexer, offset)
        except (PdfDecryptionError, PdfUnsupportedError):
            raise
        except Exception:
            return self.recover_indirect_object(lexer, offset)

    def resolve_stream(self, stream: PdfStream) -> PdfStream:
        stream = super().resolve_stream(stream)
        return (
            stream.replace(decoder=decode_stream_data)
            if stream.decoder is decode_spec_stream_data
            else stream
        )

    def get_lexer(self) -> PdfLexer:
        return PdfLexer(
            self.data,
            reference_resolver=self.resolve,
            decipher=self.decipher,
        )

    def internal_recovery_offsets(self, lexer: SyntaxLexer) -> dict[int, tuple[int, ...]]:
        """Find indirect-object headers for one damaged-xref recovery."""
        offsets: dict[int, list[int]] = {}
        for offset, object_number, generation_number in iter_indirect_object_headers(
            lexer.raw_data, 0, len(lexer.raw_data), source_buffer=lexer.source_buffer
        ):
            key = key_for(object_number, generation_number)
            offsets.setdefault(key, []).append(offset)
        return {key: tuple(values) for key, values in offsets.items()}

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

    def recover_missing_indirect_object(self, lexer: SyntaxLexer, ref: PdfReference) -> object:
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

    def resolve_name_or_text(self, value: object, *, name_like: bool = False) -> str | None:
        """A value as a name, falling back to a text string.

        ``name_like`` also accepts a non-name value whose text is a valid name,
        which lenient readers allow for AcroForm field types.
        """
        text = self.resolve_name(value)
        if text is None and name_like:
            text = self.resolve_name_like_value(value)
        return text or self.resolve_str(value)

    def resolve_name_like_value(self, resolved: object) -> str | None:
        val = self.resolve(resolved)
        name = normalize_pdf_name(val)
        if name is not None:
            return name
        if type(val) is PdfString:
            return decode_pdf_text_string(val.data)
        return None

    def resolve_float(self, value: object, default: float | None = 0.0) -> float | None:
        if type(value) is int:
            return float(value)
        if type(value) is float:
            return value
        if type(value) is bool:
            return default
        return parse_float(self.resolve(value), default=default)

    def resolve_int(self, value: object, default: int | None = None) -> int | None:
        if type(value) is int:
            return value
        return parse_int(self.resolve(value), default)

    def resolve_box(self, value: object) -> tuple[float, float, float, float] | None:
        resolved = self.deep_resolve(value)
        if resolved is None:
            return None
        box = parse_box(resolved)
        if box is None:
            raise ValueError("invalid box value")
        return box

    def internal_decode_text(self, data: bytes) -> str:
        return decode_pdf_text_string(data)
