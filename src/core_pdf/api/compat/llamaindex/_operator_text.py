"""PDF text-operator projection for the LlamaIndex compatibility facade."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from core_pdf.api.compat._text_state import (
    internal_append_directional_text,
    internal_ensure_line_break,
    internal_flush_text,
    internal_legacy_base_table,
    internal_positioned_text,
    internal_PREDEFINED_ENCODING_CODECS,
)
from core_pdf.impl.primitives import PdfName, PdfString
from core_pdf.impl.spec.s_07_content.operations import iter_content_operations
from core_pdf.impl.spec.s_07_filters.errors import FilterParseError
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf.impl.spec.s_08_graphics.matrix import multiply_affine
from core_pdf.impl.spec.s_09_fonts.cmap_tokenizer import (
    cmap_tokens,
    decode_cmap_hex_token,
    iter_blocks,
)
from core_pdf.impl.spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.spec.s_09_fonts.data.base_encodings import (
    STANDARD_ENCODING,
)
from core_pdf.impl.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.spec.s_09_fonts.glyphs import glyph_name_to_unicode

internal_WIN_ANSI_ENCODING = tuple(internal_legacy_base_table("WinAnsiEncoding"))
internal_MAC_ROMAN_ENCODING = tuple(internal_legacy_base_table("MacRomanEncoding"))
# Names whose engine translation differs from this facade's projection: the
# underscore ligature names would fall through untranslated, and negationslash
# is projected as the fraction slash.
internal_LEGACY_GLYPH_ALIASES = {
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


def internal_difference_text(glyph_name: str, code: int) -> str:
    """Translate one PDF Encoding Differences name using Adobe semantics."""
    if glyph_name == ".notdef":
        return "□"
    if glyph_name.startswith("a") and glyph_name[1:].isdigit():
        return chr(code)
    if glyph_name.isdigit():
        return f"/{glyph_name}"
    if (
        glyph_name.startswith("uni")
        and len(glyph_name) > 3
        and len(glyph_name[3:]) % 4 == 0
        and all(character in "0123456789abcdefABCDEF" for character in glyph_name[3:])
    ):
        return f"/{glyph_name}"
    mapped = internal_glyph_name_to_unicode(glyph_name)
    if len(glyph_name) == 1:
        return glyph_name
    return f"/{glyph_name}" if not mapped or mapped == glyph_name else mapped


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

    def decode_parts(self, data: bytes) -> tuple[tuple[str, ...], float]:
        encoded = self.encoded(data)
        chunks = tuple(self.character_map.get(character, character) for character in encoded)
        width = sum(
            self.space_width
            if character == self.space_character
            else self.character_widths.get(ord(character), self.default_width)
            for character in encoded
        )
        return chunks, width


class internal_TextState:
    def __init__(self, fonts: Mapping[str, internal_Font]) -> None:
        self.fonts = fonts
        self.font: internal_Font | None = None
        self.font_size = 12.0
        self.half_space_width = 125.0
        self.leading = 0.0
        self.cm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.tm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.previous_cm = self.cm.copy()
        self.previous_tm = self.tm.copy()
        self.stack: list[tuple[list[float], internal_Font | None, float, float]] = []
        self.text = ""
        # Accumulated output as parts plus its trailing character; joining or
        # copying the whole output per operator is quadratic.
        self.output_parts: list[str] = []
        self.output_last = ""
        self.width = 0.0
        self.height = 0.0
        self.rtl = False

    def flush(self) -> None:
        self.text, self.output_last = internal_flush_text(
            self.output_parts, self.text, self.output_last
        )

    def positioned(self, string_width: float) -> None:
        self.text, self.output_last = internal_positioned_text(
            self.output_parts,
            self.text,
            self.output_last,
            previous_text_matrix=self.previous_tm,
            previous_current_matrix=self.previous_cm,
            text_matrix=self.tm,
            current_matrix=self.cm,
            line_height=self.height,
            font_size=self.font_size,
            space_width=self.half_space_width,
            string_width=string_width,
        )
        self.previous_tm = self.tm.copy()
        self.previous_cm = self.cm.copy()

    def show(self, data: bytes) -> None:
        if self.font is None:
            chunks: tuple[str, ...] = ("�",) * len(data)
            width = 250.0 * len(data)
        else:
            chunks, width = self.font.decode_parts(data)
        for chunk in chunks:
            self.text, self.rtl = internal_append_directional_text(self.text, self.rtl, chunk)
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

    def show_name(self, value: PdfName) -> None:
        """Preserve a malformed name operand as its lexical PDF spelling."""
        text = f"/{value.value}"
        self.text += text
        self.width += 250.0 * len(text) * self.font_size
        self.height = self.font_size
        self.positioned(0.0)


class OperatorTextProjection:
    """Interpret page and form text operators using core-pdf's object and font engines."""

    def __init__(self, page: Any) -> None:
        self.page = page
        self.resolver = page.document.resolver
        self.active_forms: set[int] = set()

    def internal_fonts(self, resources: Mapping[object, object]) -> dict[str, internal_Font]:
        result: dict[str, internal_Font] = {}
        fonts = self.resolver.resolve(resources.get("Font"))
        if not isinstance(fonts, dict):
            return result
        for name, raw_font in fonts.items():
            font = self.resolver.resolve(raw_font)
            if not isinstance(font, dict):
                continue
            self.internal_validate_font_files(font)
            subtype = normalize_pdf_name(font.get("Subtype"))
            if subtype not in {"Type1", "MMType1", "TrueType", "Type3"} and not isinstance(
                self.resolver.resolve(font.get("DescendantFonts")),
                (list, tuple),
            ):
                raise KeyError("DescendantFonts")
            resolved = self.resolver.resolve_font_dict(font)
            decoder = FontDecoder(cast(dict[str, object], resolved))
            to_unicode = self.internal_to_unicode(resolved, decoder)
            widths, default_width = self.internal_widths(font, decoder)
            if subtype == "Type3" and not self.internal_type3_interpretable(font):
                widths, default_width = {}, 0.0
            encoding = self.internal_encoding(font, decoder, to_unicode)
            character_map = self.internal_character_map(decoder, to_unicode)
            builtin_mapping = (
                self.internal_type1_alternative(resolved)
                if subtype == "Type1" and to_unicode is None
                else {}
            )
            # Preserve the raw space identity before Type 1 recovery rewrites the
            # expanded extraction encoding. Explicit encodings take precedence;
            # otherwise ToUnicode identifies the encoded glyph most precisely.
            space_code: int | None = None
            has_explicit_encoding = self.resolver.resolve(font.get("Encoding")) is not None
            if has_explicit_encoding and not isinstance(encoding, str):
                space_code = next(
                    (code for code, text in enumerate(encoding) if text == " "),
                    None,
                )
            if space_code is None and to_unicode is not None:
                space_code = next(
                    (
                        int.from_bytes(source, "big")
                        for source, text in to_unicode.mappings.items()
                        if source and text == " "
                    ),
                    None,
                )
            if space_code is None:
                space_code = next(
                    (
                        code
                        for code, glyph_name in decoder.differences.items()
                        if internal_difference_text(glyph_name, code) == " "
                    ),
                    None,
                )
            if space_code is None:
                space_code = next(
                    (
                        code
                        for code, glyph_name in builtin_mapping.items()
                        if internal_glyph_name_to_unicode(glyph_name) == " "
                    ),
                    None,
                )
            if space_code is None:
                space_code = next(
                    (ord(code) for code, text in character_map.items() if text == " "),
                    None,
                )
            if space_code is None:
                space_code = (
                    next((code for code, text in enumerate(encoding) if text == " "), 32)
                    if not isinstance(encoding, str)
                    else 32
                )
            if subtype == "Type1" and to_unicode is None:
                # Many subset Type 1 fonts retain an explicit generic PDF encoding
                # while their embedded program carries the actual subset code map.
                # The program mapping describes the glyphs that will really render;
                # explicit /Differences remain the final PDF-level override.
                character_map = {}
                if not isinstance(encoding, str):
                    encoding_table = list(encoding)
                for code, glyph_name in builtin_mapping.items():
                    mapped = (
                        chr(int(glyph_name[1:]))
                        if glyph_name.startswith("a") and glyph_name[1:].isdigit()
                        else internal_glyph_name_to_unicode(glyph_name)
                    )
                    if mapped and (mapped != glyph_name or len(glyph_name) == 1):
                        if not isinstance(encoding, str) and 0 <= code < len(encoding_table):
                            encoding_table[code] = chr(code)
                        character_map[chr(code)] = mapped
                if not isinstance(encoding, str):
                    encoding = tuple(encoding_table)
            declared_space_width = widths.get(space_code, 0.0)
            flags = self.internal_font_flags(font)
            if default_width == 0:
                if declared_space_width:
                    default_width = declared_space_width * (1.0 if flags & 1 else 2.0)
                else:
                    positive = [width for width in widths.values() if width > 0]
                    default_width = (
                        float(sum(int(width) for width in positive) // len(positive))
                        if positive
                        else 500.0
                    )
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

    def internal_type1_alternative(self, font: Mapping[object, object]) -> dict[int, str]:
        """Read the conservative clear-text Type 1 encoding used by PDF readers."""
        descriptor = self.resolver.resolve(font.get("FontDescriptor"))
        if not isinstance(descriptor, dict):
            return {}
        font_file = self.resolver.resolve(descriptor.get("FontFile"))
        if not isinstance(font_file, PdfStream):
            return {}
        try:
            clear_text = font_file.data.split(b"eexec\n", 1)[0]
        except FilterParseError:
            return {}
        encoding_parts = clear_text.split(b"/Encoding", 1)
        if len(encoding_parts) != 2:
            return {}
        result: dict[int, str] = {}
        for line in encoding_parts[1].replace(b"\r", b"\n").split(b"\n"):
            if not line.startswith(b"dup"):
                continue
            words = [word for word in line.split(b" ") if word]
            if len(words) < 3 or (len(words) > 3 and words[3] != b"put"):
                continue
            try:
                code = int(words[1])
                glyph_name = words[2].removeprefix(b"/").decode("latin-1")
            except ValueError:
                continue
            if 0 <= code <= 255:
                result[code] = glyph_name
        return result

    def internal_type3_interpretable(self, font: Mapping[object, object]) -> bool:
        if font.get("ToUnicode") is not None:
            return True
        char_procs = self.resolver.resolve(font.get("CharProcs"))
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
        raw_cmap = font.get("ToUnicode")
        if not isinstance(raw_cmap, PdfStream):
            return None
        try:
            data = raw_cmap.data
        except FilterParseError:
            # A malformed optional ToUnicode map does not make the font unusable.
            # Continue with its declared/base encoding, as ISO 32000 requires readers
            # to do when this supplementary mapping cannot be interpreted.
            return None
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
        descendants = self.resolver.resolve(font.get("DescendantFonts"))
        if isinstance(descendants, (list, tuple)):
            owners.extend(
                descendant
                for raw_descendant in descendants
                if isinstance((descendant := self.resolver.resolve(raw_descendant)), dict)
            )
        for owner in owners:
            descriptor = self.resolver.resolve(owner.get("FontDescriptor"))
            if not isinstance(descriptor, dict):
                continue
            embedded_files = sum(
                descriptor.get(key) is not None for key in ("FontFile", "FontFile2", "FontFile3")
            )
            if embedded_files > 1:
                raise ValueError("font descriptor declares more than one embedded font program")

    def internal_font_flags(self, font: Mapping[object, object]) -> int:
        descendants = self.resolver.resolve(font.get("DescendantFonts"))
        owner: Mapping[object, object] = font
        if isinstance(descendants, (list, tuple)) and descendants:
            descendant = self.resolver.resolve(descendants[0])
            if isinstance(descendant, dict):
                owner = descendant
        descriptor = self.resolver.resolve(owner.get("FontDescriptor"))
        flags = (
            self.resolver.resolve(descriptor.get("Flags")) if isinstance(descriptor, dict) else None
        )
        return int(flags) if isinstance(flags, (int, float)) else 0

    def internal_widths(
        self,
        font: Mapping[object, object],
        decoder: FontDecoder,
    ) -> tuple[dict[int, float], float]:
        widths: dict[int, float] = {}
        default_width = 0.0
        descendants = self.resolver.resolve(font.get("DescendantFonts"))
        if isinstance(descendants, (list, tuple)):
            for raw_descendant in descendants:
                descendant = self.resolver.resolve(raw_descendant)
                if not isinstance(descendant, dict):
                    continue
                raw_w = self.resolver.resolve(descendant.get("W"))
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
                raw_default = self.resolver.resolve(descendant.get("DW"))
                if isinstance(raw_default, (int, float)):
                    default_width = float(raw_default)
        else:
            first_char = self.resolver.resolve(font.get("FirstChar"))
            raw_widths = self.resolver.resolve(font.get("Widths"))
            if isinstance(first_char, (int, float)) and isinstance(raw_widths, (list, tuple)):
                widths.update(
                    (
                        int(first_char) + offset,
                        float(int(float(self.resolver.resolve(value)))),
                    )
                    for offset, value in enumerate(raw_widths)
                )
            descriptor = self.resolver.resolve(font.get("FontDescriptor"))
            if isinstance(descriptor, dict):
                missing = self.resolver.resolve(descriptor.get("MissingWidth"))
                if isinstance(missing, (int, float)):
                    default_width = float(int(missing))
            if not widths:
                widths.update(
                    (code, width)
                    for code, width in decoder.widths.iter_explicit_widths()
                    if 0 <= code < 256 and width > 0
                )
        return widths, default_width

    def internal_encoding(
        self,
        font: Mapping[object, object],
        decoder: FontDecoder,
        to_unicode: ToUnicodeCMap | None,
    ) -> tuple[str, ...] | str:
        raw_encoding = self.resolver.resolve(font.get("Encoding"))
        encoding_name = normalize_pdf_name(raw_encoding)
        if raw_encoding is None:
            return "charmap"
        elif encoding_name is not None:
            name = encoding_name
            codecs = internal_PREDEFINED_ENCODING_CODECS
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
            base = normalize_pdf_name(raw_encoding.get("BaseEncoding"))
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
            table[code] = internal_difference_text(glyph_name, code)
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

    def internal_extract(
        self,
        streams: tuple[PdfStream, ...],
        resources: Mapping[object, object],
    ) -> str:
        state = internal_TextState(self.internal_fonts(resources))
        xobjects = self.resolver.resolve(resources.get("XObject"))
        decoded_streams: list[bytes] = []
        for stream in streams:
            try:
                decoded_streams.append(stream.data)
            except FilterParseError:
                # An undecodable content stream contributes no operators.  Other page
                # streams remain independently usable and must still be projected.
                continue
        content = b"\n".join(decoded_streams)
        for operator, raw_operands in iter_content_operations(PdfLexer(content)):
            operands = list(raw_operands)
            if operator == "BT":
                state.tm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                state.flush()
            elif operator == "ET":
                state.flush()
            elif operator == "q":
                state.stack.append((state.cm.copy(), state.font, state.font_size, state.leading))
            elif operator == "Q":
                if state.stack:
                    state.cm, state.font, state.font_size, state.leading = state.stack.pop()
                else:
                    state.cm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            elif operator == "cm":
                state.flush()
                try:
                    matrix = [float(cast(Any, value)) for value in operands[:6]]
                except (TypeError, ValueError):
                    matrix = []
                state.cm = (
                    multiply_affine(matrix, state.cm)
                    if len(matrix) == 6
                    else [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                )
            elif operator == "TL":
                scale_x = math.hypot(state.tm[0], state.tm[2])
                state.leading = (
                    float(cast(Any, operands[0])) * state.font_size * scale_x if operands else 0.0
                )
            elif operator == "Tf":
                state.flush()
                if operands:
                    state.font = state.fonts.get(str(operands[0]))
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
                value = (
                    operands[2]
                    if operator == '"' and len(operands) > 2
                    else operands[0]
                    if operands
                    else None
                )
                if isinstance(value, PdfString):
                    state.show(bytes(value.data))
                elif isinstance(value, PdfName):
                    state.show_name(value)
            elif operator == "TJ" and operands and isinstance(operands[0], (list, tuple)):
                threshold = state.half_space_width * 0.95
                for item in operands[0]:
                    if isinstance(item, PdfString):
                        state.show(bytes(item.data))
                    elif isinstance(item, PdfName):
                        state.show_name(item)
                    elif (
                        isinstance(item, (int, float))
                        and abs(float(item)) >= threshold
                        and state.text
                        and state.text[-1] != " "
                    ):
                        state.insert_space()
            elif operator == "Do" and operands and isinstance(xobjects, dict):
                state.flush()
                state.output_last = internal_ensure_line_break(
                    state.output_parts, state.output_last
                )
                form = self.resolver.resolve(xobjects.get(operands[0]))
                if not isinstance(form, PdfStream):
                    form = self.resolver.resolve(xobjects.get(str(operands[0])))
                if isinstance(form, PdfStream) and str(form.dictionary.get("Subtype")) != "Image":
                    form_resources = self.resolver.resolve(form.dictionary.get("Resources"))
                    if isinstance(form_resources, dict):
                        form_id = id(form)
                        if form_id in self.active_forms:
                            continue
                        self.active_forms.add(form_id)
                        try:
                            form_text = self.internal_extract(
                                (self.resolver.resolve_stream(form),), form_resources
                            )
                        finally:
                            self.active_forms.discard(form_id)
                        if form_text:
                            state.output_parts.append(form_text)
                            state.output_last = form_text[-1]
        state.flush()
        return "".join(state.output_parts)

    def extract_text(self) -> str:
        resources = self.page.resolve_resources()
        return self.internal_extract(tuple(self.page.content_streams), resources)
