# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from dataclasses import dataclass

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_filters.decode_spec import (
    normalize_stream_decode_spec,
)
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfObject
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    decoded_name,
    is_pdf_null,
)
from core_pdf_spec.s_07_syntax_primitives.scanning import (
    full_source_bytes,
    skip_comment,
    skip_hex_string,
    skip_literal_string,
    skip_name,
)
from core_pdf_spec.s_07_syntax_primitives.tokens import LexicalRules
from core_pdf_spec.standards import SemanticContext
from core_pdf_spec.types import PdfName

INLINE_IMAGE_KEY_MAP = {
    "BPC": "BitsPerComponent",
    "CS": "ColorSpace",
    "D": "Decode",
    "DP": "DecodeParms",
    "F": "Filter",
    "H": "Height",
    "IM": "ImageMask",
    "I": "Interpolate",
    "W": "Width",
}

INLINE_IMAGE_COLOR_SPACE_MAP = {
    "G": "DeviceGray",
    "RGB": "DeviceRGB",
    "CMYK": "DeviceCMYK",
    "I": "Indexed",
}


def internal_normalize_inline_color_space(value: PdfObject) -> PdfObject:
    name = decoded_name(value)
    if name in INLINE_IMAGE_COLOR_SPACE_MAP:
        return PdfName.of(INLINE_IMAGE_COLOR_SPACE_MAP[name])
    if isinstance(value, list) and value:
        values = list(value)
        values[0] = internal_normalize_inline_color_space(values[0])
        if decoded_name(values[0]) == "Indexed" and len(values) > 1:
            values[1] = internal_normalize_inline_color_space(values[1])
        return values
    return value


@dataclass(frozen=True, slots=True)
class InlineImage:
    dictionary: PdfDict
    data: bytes


class InlineImageDataLengthError(PdfParseError):
    def __init__(self, dictionary: PdfDict, data_start: int, expected_length: int) -> None:
        super().__init__("inline image data length does not match dimensions")
        self.dictionary = dictionary
        self.data_start = data_start
        self.expected_length = expected_length


def normalize_inline_image_dictionary(dictionary: PdfDict) -> PdfDict:
    normalized: PdfDict = {}
    for key, value in dictionary.items():
        key_name = decoded_name(key)
        if key_name is None:
            raise PdfParseError("inline image keys must be names")
        mapped_key = INLINE_IMAGE_KEY_MAP.get(key_name, key_name)
        if mapped_key == "ColorSpace":
            value = internal_normalize_inline_color_space(value)
        normalized[PdfName.of(mapped_key)] = value
    return normalized


def inline_image_unfiltered_data_length(dictionary: PdfDict) -> int | None:
    if not is_pdf_null(dictionary.get("Filter")):
        return None
    width = dictionary.get("Width")
    height = dictionary.get("Height")
    bits = dictionary.get("BitsPerComponent")
    image_mask = dictionary.get("ImageMask")
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
        color_space = decoded_name(dictionary.get("ColorSpace"))
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
    if (
        lexer.pos >= lexer.data_len
        or lexer.raw_data[lexer.pos] not in lexer.lexical_rules.whitespace
    ):
        return False
    if lexer.raw_data[lexer.pos : lexer.pos + 2] == b"\r\n":
        lexer.advance(2)
    else:
        lexer.advance(1)
    return True


def filtered_inline_image_data_end(
    dictionary: PdfDict,
    data: bytes,
    start: int,
) -> int | None:
    filters = normalize_stream_decode_spec(dictionary).steps
    if not filters:
        return None

    first_filter = filters[0].name

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
        while marker < lexer.data_len and raw_data[marker] in lexer.lexical_rules.whitespace:
            marker += 1
        if (
            marker > start + exact_length
            and raw_data[marker : marker + 2] == b"EI"
            and (
                marker + 2 == lexer.data_len
                or lexer.lexical_rules.separator_table[raw_data[marker + 2]]
            )
        ):
            image_data = (
                source_bytes[start : start + exact_length]
                if source_bytes is not None
                else bytes(raw_data[start : start + exact_length])
            )
            lexer.pos = marker + 2
            return InlineImage(normalized, image_data)

    if exact_length is not None:
        raise InlineImageDataLengthError(normalized, start, exact_length)
    return scan_inline_image_data(lexer, normalized, start)


def scan_inline_image_data(lexer: PdfLexer, dictionary: PdfDict, start: int) -> InlineImage:
    raw_data = lexer.raw_data
    source_buffer = lexer.source_buffer
    source_bytes: bytes | None = source_buffer if type(source_buffer) is bytes else None
    if source_bytes is not None:
        search_data = source_bytes
        data_start = start
        position_offset = 0
    else:
        search_data = bytes(raw_data[start:])
        data_start = 0
        position_offset = start

    hinted_end = filtered_inline_image_data_end(dictionary, search_data, data_start)
    pos = hinted_end if hinted_end is not None else data_start
    while True:
        marker = search_data.find(b"EI", pos)
        if marker < 0:
            raise PdfParseError("unterminated inline image data")
        after = marker + 2
        prev_ok = marker == data_start or search_data[marker - 1] in lexer.lexical_rules.whitespace
        next_ok = (
            after >= len(search_data) or lexer.lexical_rules.separator_table[search_data[after]]
        )
        if prev_ok and next_ok:
            data_end = marker - 1 if marker > data_start else marker
            image_data = search_data[data_start:data_end]
            lexer.pos = position_offset + after
            return InlineImage(dictionary, image_data)
        pos = marker + 1


internal_INLINE_IMAGE_MARKER_RE = re.compile(rb"[%(/<>\[\]]|BI")


def internal_next_inline_image(
    raw_bytes: bytes,
    pos: int,
    data_len: int,
    rules: LexicalRules,
) -> int | None:
    container_depth = 0
    while match := internal_INLINE_IMAGE_MARKER_RE.search(raw_bytes, pos):
        marker = match.start()
        token = match.group()
        if token == b"%":
            pos = skip_comment(raw_bytes, marker, data_len)
            continue
        if token == b"(":
            pos = skip_literal_string(raw_bytes, marker, data_len)
            continue
        if token == b"<":
            if marker + 1 < data_len and raw_bytes[marker + 1] == 60:
                container_depth += 1
                pos = marker + 2
            else:
                pos = skip_hex_string(raw_bytes, marker, data_len)
            continue
        if token == b">":
            if marker + 1 < data_len and raw_bytes[marker + 1] == 62:
                container_depth = max(0, container_depth - 1)
                pos = marker + 2
            else:
                pos = marker + 1
            continue
        if token == b"[":
            container_depth += 1
            pos = marker + 1
            continue
        if token == b"]":
            container_depth = max(0, container_depth - 1)
            pos = marker + 1
            continue
        if token == b"/":
            pos = skip_name(raw_bytes, marker, data_len, rules=rules)
            continue
        after = match.end()
        delimited = bool(
            (marker == 0 or rules.separator_table[raw_bytes[marker - 1]])
            and (after == data_len or rules.separator_table[raw_bytes[after]])
        )
        if not container_depth and delimited:
            return after
        pos = after
    return None


def validate_inline_images(
    data: bytes | memoryview, *, context: SemanticContext | None = None
) -> None:
    raw_bytes = full_source_bytes(data)
    if raw_bytes is None:
        raw_bytes = bytes(data)
    data_len = len(raw_bytes)
    pos = 0
    lexer = PdfLexer(raw_bytes, semantic_context=context)
    try:
        while (
            after := internal_next_inline_image(raw_bytes, pos, data_len, lexer.lexical_rules)
        ) is not None:
            lexer.pos = after
            parse_inline_image(lexer)
            pos = lexer.pos
    finally:
        lexer.close()


__all__ = (
    "InlineImage",
    "InlineImageDataLengthError",
    "scan_inline_image_data",
    "parse_inline_image",
    "validate_inline_images",
)
