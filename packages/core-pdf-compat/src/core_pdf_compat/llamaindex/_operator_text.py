from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

from core_pdf.impl.capture.recovery import iter_content_operations
from core_pdf.impl.document.recovery.lexer import PdfLexer
from core_pdf.impl.fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.fonts.decoder import FontDecoder
from core_pdf.impl.fonts.glyphs import glyph_name_to_unicode
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.types import PdfName, PdfString, Record, frozen_setattr
from core_pdf_compat._text_state import (
    IDENTITY_MATRIX,
    PREDEFINED_ENCODING_CODECS,
    TextMachine,
    append_directional_text,
    embedded_font_program_count,
    ensure_line_break,
    legacy_base_table,
    type1_encoding_entries,
)
from core_pdf_spec.s_07_filters.errors import FilterParseError
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_09_fonts.data.base_encodings import (
    STANDARD_ENCODING,
)

WIN_ANSI_ENCODING = tuple(legacy_base_table("WinAnsiEncoding"))
MAC_ROMAN_ENCODING = tuple(legacy_base_table("MacRomanEncoding"))
LEGACY_GLYPH_ALIASES = {
    "f_f": "ﬀ",
    "f_f_i": "ﬃ",
    "f_f_l": "ﬄ",
    "negationslash": "⁄",
}


def base_encoding_table(name: str | None) -> list[str]:
    return list(
        WIN_ANSI_ENCODING
        if name == "WinAnsiEncoding"
        else MAC_ROMAN_ENCODING
        if name == "MacRomanEncoding"
        else STANDARD_ENCODING
    )


def legacy_glyph_name_to_unicode(name: str) -> str:
    alias = LEGACY_GLYPH_ALIASES.get(name)
    if alias is not None:
        return alias
    if "_" in name:
        return name
    return glyph_name_to_unicode(name)


def difference_text(glyph_name: str, code: int) -> str:
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
    mapped = legacy_glyph_name_to_unicode(glyph_name)
    if len(glyph_name) == 1:
        return glyph_name
    return f"/{glyph_name}" if not mapped or mapped == glyph_name else mapped


class Font(Record):
    __slots__ = (
        "decoder",
        "space_character",
        "space_width",
        "encoding",
        "character_map",
        "character_widths",
        "default_width",
    )

    decoder: FontDecoder
    space_character: str
    space_width: float
    encoding: tuple[str, ...] | str
    character_map: Mapping[str, str]
    character_widths: Mapping[int, float]
    default_width: float

    __fields__: ClassVar[tuple[str, ...]] = (
        "decoder",
        "space_character",
        "space_width",
        "encoding",
        "character_map",
        "character_widths",
        "default_width",
    )
    __match_args__ = (
        "decoder",
        "space_character",
        "space_width",
        "encoding",
        "character_map",
        "character_widths",
        "default_width",
    )

    def __init__(
        self,
        decoder: FontDecoder,
        space_character: str,
        space_width: float,
        encoding: tuple[str, ...] | str,
        character_map: Mapping[str, str],
        character_widths: Mapping[int, float],
        default_width: float,
    ) -> None:
        frozen_setattr(self, "decoder", decoder)
        frozen_setattr(self, "space_character", space_character)
        frozen_setattr(self, "space_width", space_width)
        frozen_setattr(self, "encoding", encoding)
        frozen_setattr(self, "character_map", character_map)
        frozen_setattr(self, "character_widths", character_widths)
        frozen_setattr(self, "default_width", default_width)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.decoder == other.decoder
            and self.space_character == other.space_character
            and self.space_width == other.space_width
            and self.encoding == other.encoding
            and self.character_map == other.character_map
            and self.character_widths == other.character_widths
            and self.default_width == other.default_width
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.decoder,
                self.space_character,
                self.space_width,
                self.encoding,
                self.character_map,
                self.character_widths,
                self.default_width,
            )
        )

    def encoded(self, data: bytes) -> str:
        if isinstance(self.encoding, str):
            try:
                return data.decode(self.encoding, errors="surrogatepass")
            except LookupError, UnicodeDecodeError:
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


class TextState(TextMachine[Font]):
    def show(self, data: bytes) -> None:
        if self.font is None:
            chunks: tuple[str, ...] = ("�",) * len(data)
            width = 250.0 * len(data)
        else:
            chunks, width = self.font.decode_parts(data)
        for chunk in chunks:
            self.text, self.rtl = append_directional_text(self.text, self.rtl, chunk)
        self.accumulated_width += width * self.font_size
        self.actual_height = self.font_size
        self.positioned(0.0)

    def insert_space(self) -> None:
        self.text += " "
        width = (
            self.font.space_width
            if self.font is not None and self.font.space_character == " "
            else self.font.character_widths.get(32, self.font.default_width)
            if self.font is not None
            else 200.0
        )
        self.accumulated_width += width * self.font_size
        self.actual_height = self.font_size
        self.positioned(0.0)

    def show_name(self, value: PdfName) -> None:
        text = f"/{value.value}"
        self.text += text
        self.accumulated_width += 250.0 * len(text) * self.font_size
        self.actual_height = self.font_size
        self.positioned(0.0)


class OperatorTextProjection:
    def __init__(self, page: Any) -> None:
        self.page = page
        self.resolver = page.document.resolver
        self.active_forms: set[int] = set()

    def collect_fonts(self, resources: Mapping[object, object]) -> dict[str, Font]:
        result: dict[str, Font] = {}
        fonts = self.resolver.resolve(resources.get("Font"))
        if not isinstance(fonts, dict):
            return result
        for name, raw_font in fonts.items():
            font = self.resolver.resolve(raw_font)
            if not isinstance(font, dict):
                continue
            self.validate_font_files(font)
            subtype = recover_pdf_name(font.get("Subtype"))
            if subtype not in {"Type1", "MMType1", "TrueType", "Type3"} and not isinstance(
                self.resolver.resolve(font.get("DescendantFonts")),
                (list, tuple),
            ):
                raise KeyError("DescendantFonts")
            resolved = self.resolver.resolve_font_dict(font)
            decoder = FontDecoder(resolved)
            to_unicode = self.resolve_to_unicode(resolved, decoder)
            widths, default_width = self.resolve_widths(font, decoder)
            if subtype == "Type3" and not self.type3_interpretable(font):
                widths, default_width = {}, 0.0
            encoding = self.resolve_encoding(font, decoder, to_unicode)
            character_map = self.build_character_map(decoder, to_unicode)
            builtin_mapping = (
                self.type1_alternative(resolved)
                if subtype == "Type1" and to_unicode is None
                else {}
            )
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
                        if difference_text(glyph_name, code) == " "
                    ),
                    None,
                )
            if space_code is None:
                space_code = next(
                    (
                        code
                        for code, glyph_name in builtin_mapping.items()
                        if legacy_glyph_name_to_unicode(glyph_name) == " "
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
                character_map = {}
                if not isinstance(encoding, str):
                    encoding_table = list(encoding)
                for code, glyph_name in builtin_mapping.items():
                    mapped = (
                        chr(int(glyph_name[1:]))
                        if glyph_name.startswith("a") and glyph_name[1:].isdigit()
                        else legacy_glyph_name_to_unicode(glyph_name)
                    )
                    if mapped and (mapped != glyph_name or len(glyph_name) == 1):
                        if not isinstance(encoding, str) and 0 <= code < len(encoding_table):
                            encoding_table[code] = chr(code)
                        character_map[chr(code)] = mapped
                if not isinstance(encoding, str):
                    encoding = tuple(encoding_table)
            declared_space_width = widths.get(space_code, 0.0)
            flags = self.font_flags(font)
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
            result[str(name)] = Font(
                decoder=decoder,
                space_character=chr(space_code),
                space_width=declared_space_width or 200.0,
                encoding=encoding,
                character_map=character_map,
                character_widths=widths,
                default_width=default_width,
            )
        return result

    def type1_alternative(self, font: Mapping[object, object]) -> dict[int, str]:
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
        return dict(type1_encoding_entries(encoding_parts[1]))

    def type3_interpretable(self, font: Mapping[object, object]) -> bool:
        if font.get("ToUnicode") is not None:
            return True
        char_procs = self.resolver.resolve(font.get("CharProcs"))
        if not isinstance(char_procs, dict):
            return True
        return all(
            (glyph_name := recover_pdf_name(name)) is not None
            and bool(mapped := legacy_glyph_name_to_unicode(glyph_name))
            and mapped != glyph_name
            for name in char_procs
        )

    @staticmethod
    def resolve_to_unicode(
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
            return None
        try:
            return ToUnicodeCMap(data)
        except ValueError:
            return None

    def validate_font_files(self, font: Mapping[object, object]) -> None:
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
            if embedded_font_program_count(descriptor) > 1:
                raise ValueError("font descriptor declares more than one embedded font program")

    def font_flags(self, font: Mapping[object, object]) -> int:
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

    def resolve_widths(
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
                    for code, width in decoder.widths.items()
                    if 0 <= code < 256 and width > 0
                )
        return widths, default_width

    def resolve_encoding(
        self,
        font: Mapping[object, object],
        decoder: FontDecoder,
        to_unicode: ToUnicodeCMap | None,
    ) -> tuple[str, ...] | str:
        raw_encoding = self.resolver.resolve(font.get("Encoding"))
        encoding_name = recover_pdf_name(raw_encoding)
        if raw_encoding is None:
            return "charmap"
        if encoding_name is not None:
            name = encoding_name
            codecs = PREDEFINED_ENCODING_CODECS
            if name in codecs or "-UCS2-" in name:
                return codecs.get(name, "utf-16-be")
            table = base_encoding_table(name)
        elif isinstance(raw_encoding, dict):
            table = base_encoding_table(recover_pdf_name(raw_encoding.get("BaseEncoding")))
        else:
            return "charmap"
        for code, glyph_name in decoder.differences.items():
            if not 0 <= code < 256:
                continue
            table[code] = difference_text(glyph_name, code)
        if to_unicode is not None:
            for source in to_unicode.mappings:
                code = int.from_bytes(source, "big")
                if 0 <= code < 256:
                    table[code] = chr(code)
        return tuple(table)

    @staticmethod
    def build_character_map(
        decoder: FontDecoder, to_unicode: ToUnicodeCMap | None
    ) -> dict[str, str]:
        if to_unicode is None:
            result: dict[str, str] = {}
            if decoder.differences:
                return result
            for code, glyph_name in decoder.encoding_differences.items():
                mapped = legacy_glyph_name_to_unicode(glyph_name)
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

    def extract(
        self,
        streams: tuple[PdfStream, ...],
        resources: Mapping[object, object],
    ) -> str:
        state = TextState(self.collect_fonts(resources))
        xobjects = self.resolver.resolve(resources.get("XObject"))
        decoded_streams: list[bytes] = []
        for stream in streams:
            try:
                decoded_streams.append(stream.data)
            except FilterParseError:
                continue
        content = b"\n".join(decoded_streams)
        parsed = list(iter_content_operations(PdfLexer(content)))
        for operator, raw_operands in parsed:
            operands = list(raw_operands)
            if state.apply_state_operator(operator, operands):
                continue
            match operator:
                case "Tf":
                    state.flush()
                    if operands:
                        state.font = state.fonts.get(str(operands[0]))
                    if len(operands) > 1:
                        state.font_size = float(operands[1])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    state.half_space_width = (
                        state.font.space_width / 2.0 if state.font is not None else 125.0
                    )
                case "Td" | "TD":
                    tx = float(operands[0]) if operands else 0.0  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    ty = float(operands[1]) if len(operands) > 1 else 0.0  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    if operator == "TD":
                        scale_x = math.hypot(state.tm[0], state.tm[2])
                        state.leading = -ty * state.font_size * scale_x
                    state.tm[4] += tx * state.tm[0] + ty * state.tm[2]
                    state.tm[5] += tx * state.tm[1] + ty * state.tm[3]
                    state.positioned(state.accumulated_width / 1000.0)
                    state.accumulated_width = 0.0
                case "Tm":
                    try:
                        matrix = [float(value) for value in operands[:6]]  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    except TypeError, ValueError:
                        matrix = []
                    state.tm = matrix if len(matrix) == 6 else list(IDENTITY_MATRIX)
                    state.positioned(state.accumulated_width / 1000.0)
                    state.accumulated_width = 0.0
                case "T*":
                    state.tm[4] -= state.leading * state.tm[2]
                    state.tm[5] -= state.leading * state.tm[3]
                    state.positioned(state.accumulated_width / 1000.0)
                    state.accumulated_width = 0.0
                case "Tj" | "'" | '"':
                    if operator in {"'", '"'}:
                        state.tm[4] -= state.leading * state.tm[2]
                        state.tm[5] -= state.leading * state.tm[3]
                        state.positioned(state.accumulated_width / 1000.0)
                        state.accumulated_width = 0.0
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
                case "TJ" if operands and isinstance(operands[0], (list, tuple)):
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
                case "Do" if operands and isinstance(xobjects, dict):
                    state.flush()
                    state.output_last = ensure_line_break(state.output_parts, state.output_last)
                    form = self.resolver.resolve(xobjects.get(operands[0]))
                    if not isinstance(form, PdfStream):
                        form = self.resolver.resolve(xobjects.get(str(operands[0])))
                    if (
                        isinstance(form, PdfStream)
                        and str(form.dictionary.get("Subtype")) != "Image"
                    ):
                        form_resources = self.resolver.resolve(form.dictionary.get("Resources"))
                        if isinstance(form_resources, dict):
                            form_id = id(form)
                            if form_id in self.active_forms:
                                continue
                            self.active_forms.add(form_id)
                            try:
                                form_text = self.extract(
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
        resources = self.page.resources
        return self.extract(tuple(self.page.content_streams), resources)
