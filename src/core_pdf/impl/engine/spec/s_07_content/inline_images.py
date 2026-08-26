# SPDX-License-Identifier: AGPL-3.0-only
"""Native inline-image parsing and decode helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core_pdf.impl.engine.spec.s_07_filters.decode_spec import (
    normalize_stream_decode_spec,
)
from core_pdf.impl.engine.spec.s_07_syntax.coercion import (
    is_pdf_null,
    normalize_pdf_name,
)
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_07_syntax.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_syntax.tokens import (
    INLINE_IMAGE_KEY_MAP,
    WHITESPACE,
)
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.primitives import PdfName
from core_pdf.impl.types import PdfDict


class InlineImage:
    __slots__ = ("dictionary", "data")

    dictionary: dict[str, Any]
    data: bytes

    def __init__(self, dictionary: PdfDict, data: bytes) -> None:
        object.__setattr__(self, "dictionary", dictionary)
        object.__setattr__(self, "data", data)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"cannot assign to field {name!r}")


def normalize_inline_image_dictionary(dictionary: PdfDict) -> PdfDict:
    normalized: PdfDict = {}
    for key, value in dictionary.items():
        key_name = normalize_pdf_name(key)
        if key_name is None:
            raise PdfParseError("inline image keys must be names")
        mapped_key = INLINE_IMAGE_KEY_MAP.get(key_name, key_name)
        normalized[PdfName.of(mapped_key)] = value
    return normalized


def inline_image_unfiltered_data_length(dictionary: PdfDict) -> int | None:
    if not is_pdf_null(lookup_dict_key(dictionary, "Filter")):
        return None
    width = lookup_dict_key(dictionary, "Width")
    height = lookup_dict_key(dictionary, "Height")
    bits = lookup_dict_key(dictionary, "BitsPerComponent")
    image_mask = lookup_dict_key(dictionary, "ImageMask")
    if type(width) is not int or type(height) is not int:
        return None
    if width <= 0 or height <= 0:
        return None
    if image_mask is True:
        bits = 1
        colors = 1
    else:
        if type(bits) is not int or bits <= 0:
            return None
        color_space = normalize_pdf_name(lookup_dict_key(dictionary, "ColorSpace"))
        if color_space in {None, "G", "DeviceGray"}:
            colors = 1
        elif color_space in {"RGB", "DeviceRGB"}:
            colors = 3
        elif color_space in {"CMYK", "DeviceCMYK"}:
            colors = 4
        else:
            return None
    row_bits = width * colors * bits
    return ((row_bits + 7) // 8) * height


def skip_inline_image_separator(lexer: PdfLexer) -> bool:
    start = lexer.pos
    lexer.skip_eol()
    if lexer.pos < lexer.data_len and lexer.raw_data[lexer.pos] in WHITESPACE:
        lexer.advance(1)
    return lexer.pos > start


def filtered_inline_image_data_end(
    dictionary: PdfDict,
    data: bytes,
    start: int,
) -> int | None:
    try:
        filters = normalize_stream_decode_spec(dictionary).filters
    except PdfParseError:
        return None
    if not filters:
        return None

    first_filter = filters[0]

    if first_filter in {"ASCII85Decode", "A85"}:
        marker = data.find(b"~>", start)
        return None if marker < 0 else marker + 2
    if first_filter in {"ASCIIHexDecode", "AHx"}:
        marker = data.find(b">", start)
        return None if marker < 0 else marker + 1
    if first_filter in {"DCTDecode", "DCT"}:
        marker = data.find(b"\xff\xd9", start)
        return None if marker < 0 else marker + 2
    if first_filter in {"RunLengthDecode", "RL"}:
        pos = start
        while pos < len(data):
            length = data[pos]
            pos += 1
            if length == 128:
                return pos
            pos += length + 1 if length < 128 else 1
            if pos > len(data):
                return None
    return None


def parse_inline_image(lexer: PdfLexer) -> InlineImage:
    dictionary: PdfDict = {}
    while True:
        lexer.skip_ignored()
        if lexer.pos >= lexer.data_len:
            raise PdfParseError("unterminated inline image")
        if lexer.raw_data[lexer.pos : lexer.pos + 2] == b"ID":
            lexer.advance(2)
            break
        if lexer.raw_data[lexer.pos] != 47:
            raise PdfParseError("inline image keys must be names")
        key = PdfName.of(lexer.read_name())
        dictionary[key] = lexer.parse_object()

    if not skip_inline_image_separator(lexer):
        raise PdfParseError("expected inline image data separator")
    start = lexer.pos
    normalized = normalize_inline_image_dictionary(dictionary)
    raw_data = lexer.raw_data
    source_buffer = lexer.source_buffer
    source_bytes: bytes | None = source_buffer if type(source_buffer) is bytes else None

    exact_length = inline_image_unfiltered_data_length(normalized)
    if exact_length is not None and start + exact_length <= lexer.data_len:
        marker = start + exact_length
        while marker < lexer.data_len and raw_data[marker] in WHITESPACE:
            marker += 1
        if raw_data[marker : marker + 2] == b"EI":
            image_data = (
                source_bytes[start : start + exact_length]
                if source_bytes is not None
                else bytes(raw_data[start : start + exact_length])
            )
            lexer.pos = marker + 2
            return InlineImage(normalized, image_data)

    if source_bytes is not None:
        search_data = source_bytes
        data_start = start
        position_offset = 0
    else:
        search_data = bytes(raw_data[start:])
        data_start = 0
        position_offset = start

    hinted_end = filtered_inline_image_data_end(normalized, search_data, data_start)
    pos = hinted_end if hinted_end is not None else data_start
    while True:
        marker = search_data.find(b"EI", pos)
        if marker < 0:
            raise PdfParseError("unterminated inline image data")
        after = marker + 2
        prev_ok = marker == data_start or search_data[marker - 1] in WHITESPACE
        next_ok = (
            after >= len(search_data)
            or search_data[after] in WHITESPACE
            or search_data[after] in b"()<>[]{}/%"
        )
        if prev_ok and next_ok:
            image_data = search_data[data_start:marker].rstrip(WHITESPACE)
            lexer.pos = position_offset + after
            return InlineImage(normalized, image_data)
        pos = marker + 1


def recover_inline_image_position(
    lexer: PdfLexer,
    position: int,
    is_valid_operator: Callable[[bytes], bool] | None = None,
) -> int | None:
    data = lexer.raw_data
    data_len = lexer.data_len
    source_buffer = lexer.source_buffer
    source_bytes: bytes | None = source_buffer if type(source_buffer) is bytes else None
    search_data = source_bytes if source_bytes is not None else data.tobytes()
    pos = position
    while pos < data_len:
        marker = search_data.find(b"EI", pos)
        if marker < 0:
            return None
        after = marker + 2
        if (
            (marker == 0 or data[marker - 1] in WHITESPACE)
            and after < data_len
            and data[after] in WHITESPACE
        ):
            next_pos = lexer.skip_ignored_at(after)
            word = lexer.scan_word_at(next_pos, skip_ignored=False)
            if word is None:
                return after
            token, ignored = word
            if (
                is_valid_operator(bytes(token))
                if is_valid_operator is not None
                else token in (b"BT", b"ET", b"q", b"Q", b"cm", b"Do", b"BI")
            ):
                return next_pos
        pos = marker + 1
    return None


__all__ = (
    "InlineImage",
    "parse_inline_image",
    "recover_inline_image_position",
)
