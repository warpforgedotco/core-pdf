"""PDF text-operator projection for the LlamaIndex compatibility facade."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from core_pdf.impl.engine.spec.s_07_content.operations import iter_content_operations
from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_09_fonts.cmap_tokenizer import (
    cmap_tokens,
    decode_cmap_hex_token,
    iter_blocks,
)
from core_pdf.impl.engine.spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.engine.spec.s_09_fonts.data.base_encodings import (
    STANDARD_ENCODING,
)
from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.engine.spec.s_09_fonts.glyphs import glyph_name_to_unicode
from core_pdf.impl.objects import PdfStream, PdfString


def internal_mult(left: list[float], right: list[float]) -> list[float]:
    return [
        left[0] * right[0] + left[1] * right[2],
        left[0] * right[1] + left[1] * right[3],
        left[2] * right[0] + left[3] * right[2],
        left[2] * right[1] + left[3] * right[3],
        left[4] * right[0] + left[5] * right[2] + right[4],
        left[4] * right[1] + left[5] * right[3] + right[5],
    ]


def internal_orientation(matrix: list[float]) -> int:
    if matrix[3] > 1e-6:
        return 0
    if matrix[3] < -1e-6:
        return 180
    return 90 if matrix[1] > 0 else 270


def internal_neutral(character: str) -> bool:
    return any(
        start <= character <= end
        for start, end in (
            ("\x00", "\x2f"),
            ("\x3a", "\x40"),
            ("\u2000", "\u206f"),
            ("\u20a0", "\u21ff"),
        )
    )


def internal_rtl(character: str) -> bool:
    return any(
        start <= character <= end
        for start, end in (
            ("\u0590", "\u08ff"),
            ("\ufb1d", "\ufdff"),
            ("\ufe70", "\ufeff"),
        )
    )


def internal_byte_encoding(name: str) -> tuple[str, ...]:
    table: list[str] = []
    for code in range(256):
        try:
            table.append(bytes((code,)).decode(name))
        except UnicodeDecodeError:
            table.append(chr(code))
    return tuple(table)


internal_WIN_ANSI_ENCODING = internal_byte_encoding("cp1252")
internal_MAC_ROMAN_ENCODING = internal_byte_encoding("mac_roman")
internal_LEGACY_GLYPH_ALIASES = {
    "Ifractur": "ℑ",
    "Rfractur": "ℜ",
    "circlecopyrt": "©",
    "epsilon1": "ϵ",
    "f_f": "ﬀ",
    "f_f_i": "ﬃ",
    "f_f_l": "ﬄ",
    "negationslash": "⁄",
}


def internal_glyph_name_to_unicode(name: str) -> str:
    alias = internal_LEGACY_GLYPH_ALIASES.get(name)
    if alias is not None:
        return alias
    if "_" in name:
        return name
    return glyph_name_to_unicode(name)


@dataclass(frozen=True, slots=True)
class internal_Font:
    decoder: FontDecoder
    space_character: str
    space_width: float
    encoding: tuple[str, ...] | str
    character_map: Mapping[str, str]
    character_widths: Mapping[int, float]
    default_width: float

    def encoded(self, data: bytes) -> str:
        if isinstance(self.encoding, str):
            try:
                return data.decode(self.encoding, errors="surrogatepass")
            except (LookupError, UnicodeDecodeError):
                return data.decode(
                    "utf-16-be" if self.encoding == "charmap" else "latin-1",
                    errors="surrogatepass",
                )
        return "".join(self.encoding[code] or chr(code) for code in data)

    def decode(self, data: bytes) -> str:
        return "".join(self.display_chunks(data))

    def display_chunks(self, data: bytes) -> tuple[str, ...]:
        return tuple(
            self.character_map.get(character, character) for character in self.encoded(data)
        )

    def text_width(self, data: bytes) -> float:
        return sum(
            self.space_width
            if character == self.space_character
            else self.character_widths.get(ord(character), self.default_width)
            for character in self.encoded(data)
        )


class internal_TextState:
    def __init__(self, fonts: Mapping[str, internal_Font]) -> None:
        self.fonts = fonts
        self.font: internal_Font | None = None
        self.font_name: str | None = None
        self.font_size = 12.0
        self.half_space_width = 125.0
        self.leading = 0.0
        self.cm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.tm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.previous_cm = self.cm.copy()
        self.previous_tm = self.tm.copy()
        self.memo_cm = self.cm.copy()
        self.memo_tm = self.tm.copy()
        self.stack: list[tuple[list[float], internal_Font | None, str | None, float, float]] = []
        self.text = ""
        self.output = ""
        self.width = 0.0
        self.height = 0.0
        self.rtl = False

    def flush(self) -> None:
        self.output += self.text
        self.text = ""
        self.memo_cm = self.cm.copy()
        self.memo_tm = self.tm.copy()

    def positioned(self, string_width: float) -> None:
        previous = internal_mult(self.previous_tm, self.previous_cm)
        current = internal_mult(self.tm, self.cm)
        orientation = internal_orientation(current)
        delta_x = current[4] - previous[4]
        delta_y = current[5] - previous[5]
        previous_scale_x = math.hypot(self.previous_tm[0], self.previous_tm[1])
        previous_scale_y = math.hypot(self.previous_tm[2], self.previous_tm[3])
        current_scale_y = math.hypot(self.tm[2], self.tm[3])
        moved_height, moved_width = (
            (delta_y, delta_x) if orientation in (0, 180) else (delta_x, delta_y)
        )
        try:
            if abs(moved_height) > 0.8 * min(
                self.height * previous_scale_y, self.font_size * current_scale_y
            ):
                if (self.output + self.text)[-1] != "\n":
                    self.output += self.text + "\n"
                    self.text = ""
            elif (
                moved_width
                >= (self.font_size * self.half_space_width / 1000.0 + string_width)
                * previous_scale_x
                and (self.output + self.text)[-1] != " "
            ):
                self.text += " "
        except (IndexError, ValueError):
            pass
        self.previous_tm = self.tm.copy()
        self.previous_cm = self.cm.copy()
        if not self.text:
            self.memo_tm = self.tm.copy()
            self.memo_cm = self.cm.copy()

    def show(self, data: bytes) -> None:
        if self.font is None:
            chunks = ("�",) * len(data)
            width = 250.0 * len(data)
        else:
            chunks = self.font.display_chunks(data)
            width = self.font.text_width(data)
        for chunk in chunks:
            if len(chunk) != 1 or internal_neutral(chunk):
                self.text = chunk + self.text if self.rtl else self.text + chunk
            elif internal_rtl(chunk):
                if not self.rtl:
                    self.rtl = True
                    self.text = ""
                self.text = chunk + self.text
            else:
                if self.rtl:
                    self.rtl = False
                    self.text = ""
                self.text += chunk
        self.width += width * self.font_size
        self.height = self.font_size
        self.positioned(0.0)

    def insert_space(self) -> None:
        """Insert layout whitespace without passing it through the active font CMap."""
        self.text += " "
        width = (
            self.font.space_width
            if self.font is not None and self.font.space_character == " "
            else self.font.character_widths.get(32, self.font.default_width)
            if self.font is not None
            else 200.0
        )
        self.width += width * self.font_size
        self.height = self.font_size
        self.positioned(0.0)


class OperatorTextProjection:
    """Interpret page and form text operators using core-pdf's object and font engines."""

    def __init__(self, page: Any) -> None:
        self.page = page
        self.resolver = page.document.resolver

    def internal_fonts(self, resources: Mapping[object, object]) -> dict[str, internal_Font]:
        result: dict[str, internal_Font] = {}
        fonts = self.resolver.resolve(lookup_dict_key(resources, "Font"))
        if not isinstance(fonts, dict):
            return result
        for name, raw_font in fonts.items():
            font = self.resolver.resolve(raw_font)
            if not isinstance(font, dict):
                continue
            self.internal_validate_font_files(font)
            subtype = normalize_pdf_name(lookup_dict_key(font, "Subtype"))
            if subtype not in {"Type1", "MMType1", "TrueType", "Type3"} and not isinstance(
                self.resolver.resolve(lookup_dict_key(font, "DescendantFonts")),
                (list, tuple),
            ):
                raise KeyError("DescendantFonts")
            resolved = self.resolver.resolve_font_dict(font)
            decoder = FontDecoder(cast(dict[str, object], resolved))
            to_unicode = self.internal_to_unicode(resolved, decoder)
            widths, default_width = self.internal_widths(font)
            if subtype == "Type3" and not self.internal_type3_interpretable(font):
                widths, default_width = {}, 0.0
            encoding = self.internal_encoding(font, decoder, to_unicode)
            character_map = self.internal_character_map(decoder, to_unicode)
            space_code = (
                next((code for code, text in enumerate(encoding) if text == " "), 32)
                if not isinstance(encoding, str)
                else next(
                    (ord(code) for code, text in character_map.items() if text == " "),
                    32,
                )
            )
            declared_space_width = widths.get(space_code, 0.0)
            if default_width == 0:
                if declared_space_width:
                    default_width = declared_space_width * (
                        1.0 if self.internal_font_flags(font) & 1 else 2.0
                    )
                else:
                    positive = [width for width in widths.values() if width > 0]
                    default_width = sum(positive) // len(positive) if positive else 500.0
            result[str(name)] = internal_Font(
                decoder=decoder,
                space_character=chr(space_code),
                space_width=declared_space_width or 200.0,
                encoding=encoding,
                character_map=character_map,
                character_widths=widths,
                default_width=default_width,
            )
        return result

    def internal_type3_interpretable(self, font: Mapping[object, object]) -> bool:
        if lookup_dict_key(font, "ToUnicode") is not None:
            return True
        char_procs = self.resolver.resolve(lookup_dict_key(font, "CharProcs"))
        if not isinstance(char_procs, dict):
            return True
        return all(
            (glyph_name := normalize_pdf_name(name)) is not None
            and bool(mapped := internal_glyph_name_to_unicode(glyph_name))
            and mapped != glyph_name
            for name in char_procs
        )

    @staticmethod
    def internal_to_unicode(
        font: Mapping[object, object], decoder: FontDecoder
    ) -> ToUnicodeCMap | None:
        if decoder.to_unicode is not None:
            return decoder.to_unicode
        raw_cmap = lookup_dict_key(font, "ToUnicode")
        if not isinstance(raw_cmap, PdfStream):
            return None
        data = raw_cmap.data
        try:
            return ToUnicodeCMap(data)
        except ValueError:
            pass
        begin = b"begincodespacerange"
        end = b"endcodespacerange"
        ranges: list[tuple[bytes, bytes]] = []
        try:
            for block in iter_blocks(data, begin, end):
                tokens = cmap_tokens(block)
                ranges.extend(
                    (decode_cmap_hex_token(tokens[index]), decode_cmap_hex_token(tokens[index + 1]))
                    for index in range(0, len(tokens) - 1, 2)
                )
        except (UnicodeDecodeError, ValueError):
            return None
        if len(ranges) != 1:
            return None
        range_start, range_end = ranges[0]
        if (
            len(range_start) != len(range_end)
            or int.from_bytes(range_start, "big") > int.from_bytes(range_end, "big")
            or all(left <= right for left, right in zip(range_start, range_end, strict=True))
        ):
            return None
        while (start := data.find(begin)) >= 0:
            line_start = data.rfind(b"\n", 0, start) + 1
            stop = data.find(end, start + len(begin))
            if stop < 0:
                return None
            line_end = data.find(b"\n", stop + len(end))
            data = data[:line_start] + data[len(data) if line_end < 0 else line_end + 1 :]
        try:
            return ToUnicodeCMap(data)
        except ValueError:
            return None

    def internal_validate_font_files(self, font: Mapping[object, object]) -> None:
        owners: list[Mapping[object, object]] = [font]
        descendants = self.resolver.resolve(lookup_dict_key(font, "DescendantFonts"))
        if isinstance(descendants, (list, tuple)):
            owners.extend(
                descendant
                for raw_descendant in descendants
                if isinstance((descendant := self.resolver.resolve(raw_descendant)), dict)
            )
        for owner in owners:
            descriptor = self.resolver.resolve(lookup_dict_key(owner, "FontDescriptor"))
            if not isinstance(descriptor, dict):
                continue
            embedded_files = sum(
                lookup_dict_key(descriptor, key) is not None
                for key in ("FontFile", "FontFile2", "FontFile3")
            )
            if embedded_files > 1:
                raise ValueError("font descriptor declares more than one embedded font program")

    def internal_font_flags(self, font: Mapping[object, object]) -> int:
        descendants = self.resolver.resolve(lookup_dict_key(font, "DescendantFonts"))
        owner: Mapping[object, object] = font
        if isinstance(descendants, (list, tuple)) and descendants:
            descendant = self.resolver.resolve(descendants[0])
            if isinstance(descendant, dict):
                owner = descendant
        descriptor = self.resolver.resolve(lookup_dict_key(owner, "FontDescriptor"))
        flags = (
            self.resolver.resolve(lookup_dict_key(descriptor, "Flags"))
            if isinstance(descriptor, dict)
            else None
        )
        return int(flags) if isinstance(flags, (int, float)) else 0

    def internal_widths(self, font: Mapping[object, object]) -> tuple[dict[int, float], float]:
        widths: dict[int, float] = {}
        default_width = 0.0
        descendants = self.resolver.resolve(lookup_dict_key(font, "DescendantFonts"))
        if isinstance(descendants, (list, tuple)):
            for raw_descendant in descendants:
                descendant = self.resolver.resolve(raw_descendant)
                if not isinstance(descendant, dict):
                    continue
                raw_w = self.resolver.resolve(lookup_dict_key(descendant, "W"))
                if isinstance(raw_w, (list, tuple)):
                    index = 0
                    while index < len(raw_w):
                        start = self.resolver.resolve(raw_w[index])
                        if not isinstance(start, (int, float)) or index + 1 >= len(raw_w):
                            index += 1
                            continue
                        following = self.resolver.resolve(raw_w[index + 1])
                        if isinstance(following, (list, tuple)):
                            widths.update(
                                (int(start) + offset, float(self.resolver.resolve(value)))
                                for offset, value in enumerate(following)
                            )
                            index += 2
                            continue
                        if index + 2 < len(raw_w):
                            stop = self.resolver.resolve(raw_w[index + 1])
                            value = self.resolver.resolve(raw_w[index + 2])
                            if isinstance(stop, (int, float)) and isinstance(value, (int, float)):
                                widths.update(
                                    (code, float(value))
                                    for code in range(int(start), int(stop) + 1)
                                )
                                index += 3
                                continue
                        index += 1
                raw_default = self.resolver.resolve(lookup_dict_key(descendant, "DW"))
                if isinstance(raw_default, (int, float)):
                    default_width = float(raw_default)
        else:
            first_char = self.resolver.resolve(lookup_dict_key(font, "FirstChar"))
            raw_widths = self.resolver.resolve(lookup_dict_key(font, "Widths"))
            if isinstance(first_char, (int, float)) and isinstance(raw_widths, (list, tuple)):
                widths.update(
                    (int(first_char) + offset, float(self.resolver.resolve(value)))
                    for offset, value in enumerate(raw_widths)
                )
            descriptor = self.resolver.resolve(lookup_dict_key(font, "FontDescriptor"))
            if isinstance(descriptor, dict):
                missing = self.resolver.resolve(lookup_dict_key(descriptor, "MissingWidth"))
                if isinstance(missing, (int, float)):
                    default_width = float(missing)
        return widths, default_width

    def internal_encoding(
        self,
        font: Mapping[object, object],
        decoder: FontDecoder,
        to_unicode: ToUnicodeCMap | None,
    ) -> tuple[str, ...] | str:
        raw_encoding = self.resolver.resolve(lookup_dict_key(font, "Encoding"))
        encoding_name = normalize_pdf_name(raw_encoding)
        if raw_encoding is None:
            return "charmap"
        elif encoding_name is not None:
            name = encoding_name
            codecs = {
                "Identity-H": "utf-16-be",
                "Identity-V": "utf-16-be",
                "GB-EUC-H": "gbk",
                "GB-EUC-V": "gbk",
                "GBpc-EUC-H": "gb2312",
                "GBpc-EUC-V": "gb2312",
                "GBK-EUC-H": "gbk",
                "GBK-EUC-V": "gbk",
                "GBK2K-H": "gb18030",
                "GBK2K-V": "gb18030",
                "ETen-B5-H": "cp950",
                "ETen-B5-V": "cp950",
                "ETenms-B5-H": "cp950",
                "ETenms-B5-V": "cp950",
                "90ms-RKSJ-H": "cp932",
                "90ms-RKSJ-V": "cp932",
            }
            if name in codecs or "-UCS2-" in name:
                return codecs.get(name, "utf-16-be")
            table = list(
                internal_WIN_ANSI_ENCODING
                if name == "WinAnsiEncoding"
                else internal_MAC_ROMAN_ENCODING
                if name == "MacRomanEncoding"
                else STANDARD_ENCODING
            )
        elif isinstance(raw_encoding, dict):
            base = normalize_pdf_name(lookup_dict_key(raw_encoding, "BaseEncoding"))
            table = list(
                internal_WIN_ANSI_ENCODING
                if base == "WinAnsiEncoding"
                else internal_MAC_ROMAN_ENCODING
                if base == "MacRomanEncoding"
                else STANDARD_ENCODING
            )
        else:
            return "charmap"
        for code, glyph_name in decoder.differences.items():
            if not 0 <= code < 256:
                continue
            if glyph_name.startswith("a") and glyph_name[1:].isdigit():
                table[code] = chr(code)
                continue
            if glyph_name.isdigit():
                table[code] = f"/{glyph_name}"
                continue
            if (
                glyph_name.startswith("uni")
                and len(glyph_name) > 3
                and len(glyph_name[3:]) % 4 == 0
                and all(character in "0123456789abcdefABCDEF" for character in glyph_name[3:])
            ):
                table[code] = f"/{glyph_name}"
                continue
            mapped = internal_glyph_name_to_unicode(glyph_name)
            table[code] = (
                glyph_name
                if len(glyph_name) == 1
                else f"/{glyph_name}"
                if mapped == glyph_name
                else mapped
            )
        if to_unicode is not None:
            for source in to_unicode.mappings:
                code = int.from_bytes(source, "big")
                if 0 <= code < 256:
                    table[code] = chr(code)
        return tuple(table)

    @staticmethod
    def internal_character_map(
        decoder: FontDecoder, to_unicode: ToUnicodeCMap | None
    ) -> dict[str, str]:
        if to_unicode is None:
            result: dict[str, str] = {}
            if decoder.differences:
                return result
            for code, glyph_name in decoder.encoding_differences.items():
                mapped = internal_glyph_name_to_unicode(glyph_name)
                if mapped != glyph_name:
                    result[chr(code)] = mapped
            return result
        return {
            chr(int.from_bytes(source, "big")): (
                " " if int.from_bytes(source, "big") == 32 and text == "␣" else text
            )
            for source, text in to_unicode.mappings.items()
            if source
        }

    def internal_streams(self, value: object) -> tuple[PdfStream, ...]:
        resolved = self.resolver.resolve(value)
        if isinstance(resolved, PdfStream):
            return (self.resolver.resolve_stream(resolved),)
        if isinstance(resolved, (list, tuple)):
            return tuple(
                self.resolver.resolve_stream(stream)
                for item in resolved
                if isinstance((stream := self.resolver.resolve(item)), PdfStream)
            )
        return ()

    def internal_extract(
        self,
        streams: tuple[PdfStream, ...],
        resources: Mapping[object, object],
    ) -> str:
        state = internal_TextState(self.internal_fonts(resources))
        xobjects = self.resolver.resolve(lookup_dict_key(resources, "XObject"))
        content = b"\n".join(stream.data for stream in streams)
        for operator, raw_operands in iter_content_operations(PdfLexer(content)):
            operands = list(raw_operands)
            if operator == "BT":
                state.tm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                state.flush()
            elif operator == "ET":
                state.flush()
            elif operator == "q":
                state.stack.append(
                    (
                        state.cm.copy(),
                        state.font,
                        state.font_name,
                        state.font_size,
                        state.leading,
                    )
                )
            elif operator == "Q":
                if state.stack:
                    state.cm, state.font, state.font_name, state.font_size, state.leading = (
                        state.stack.pop()
                    )
                else:
                    state.cm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            elif operator == "cm":
                state.flush()
                try:
                    state.cm = internal_mult(
                        [float(cast(Any, value)) for value in operands[:6]], state.cm
                    )
                except (TypeError, ValueError):
                    state.cm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                state.memo_cm = state.cm.copy()
                state.memo_tm = state.tm.copy()
            elif operator == "TL":
                scale_x = math.hypot(state.tm[0], state.tm[2])
                state.leading = (
                    float(cast(Any, operands[0])) * state.font_size * scale_x if operands else 0.0
                )
            elif operator == "Tf":
                state.flush()
                if operands:
                    state.font_name = str(operands[0])
                    state.font = state.fonts.get(state.font_name)
                if len(operands) > 1:
                    state.font_size = float(cast(Any, operands[1]))
                state.half_space_width = (
                    state.font.space_width / 2.0 if state.font is not None else 125.0
                )
            elif operator in {"Td", "TD"}:
                tx = float(cast(Any, operands[0])) if operands else 0.0
                ty = float(cast(Any, operands[1])) if len(operands) > 1 else 0.0
                if operator == "TD":
                    scale_x = math.hypot(state.tm[0], state.tm[2])
                    state.leading = -ty * state.font_size * scale_x
                state.tm[4] += tx * state.tm[0] + ty * state.tm[2]
                state.tm[5] += tx * state.tm[1] + ty * state.tm[3]
                state.positioned(state.width / 1000.0)
                state.width = 0.0
            elif operator == "Tm":
                try:
                    matrix = [float(cast(Any, value)) for value in operands[:6]]
                except (TypeError, ValueError):
                    matrix = []
                state.tm = matrix if len(matrix) == 6 else [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                state.positioned(state.width / 1000.0)
                state.width = 0.0
            elif operator == "T*":
                state.tm[4] -= state.leading * state.tm[2]
                state.tm[5] -= state.leading * state.tm[3]
                state.positioned(state.width / 1000.0)
                state.width = 0.0
            elif operator in {"Tj", "'", '"'}:
                if operator in {"'", '"'}:
                    state.tm[4] -= state.leading * state.tm[2]
                    state.tm[5] -= state.leading * state.tm[3]
                    state.positioned(state.width / 1000.0)
                    state.width = 0.0
                value = operands[-1] if operands else None
                state.show(bytes(value.data) if isinstance(value, PdfString) else b"")
            elif operator == "TJ" and operands and isinstance(operands[0], (list, tuple)):
                threshold = state.half_space_width * 0.95
                for item in operands[0]:
                    if isinstance(item, PdfString):
                        state.show(bytes(item.data))
                    elif (
                        isinstance(item, (int, float))
                        and abs(float(item)) >= threshold
                        and state.text
                        and state.text[-1] != " "
                    ):
                        state.insert_space()
            elif operator == "Do" and operands and isinstance(xobjects, dict):
                state.flush()
                if state.output and not state.output.endswith("\n"):
                    state.output += "\n"
                form = self.resolver.resolve(xobjects.get(operands[0]))
                if not isinstance(form, PdfStream):
                    form = self.resolver.resolve(xobjects.get(str(operands[0])))
                if (
                    isinstance(form, PdfStream)
                    and str(lookup_dict_key(form.dictionary, "Subtype")) != "Image"
                ):
                    form_resources = self.resolver.resolve(
                        lookup_dict_key(form.dictionary, "Resources")
                    )
                    if isinstance(form_resources, dict):
                        nested = self.internal_extract(
                            (self.resolver.resolve_stream(form),), form_resources
                        )
                        state.output += nested
        state.flush()
        return state.output

    def extract_text(self) -> str:
        resources = self.page.resolve_resources()
        return self.internal_extract(tuple(self.page.content_streams), resources)
