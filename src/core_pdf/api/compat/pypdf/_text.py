"""Content-stream-order text extraction for legacy compatibility projections."""

from __future__ import annotations

import math
import re
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast

from core_pdf.api.compat._shared import LIGATURES
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
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf.impl.spec.s_08_graphics.matrix import multiply_affine
from core_pdf.impl.spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.spec.s_09_fonts.cmap_widths import FontWidthMap, SparseFontWidthMap
from core_pdf.impl.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.spec.s_09_fonts.glyphs import (
    TEX_GLYPH_ALIASES,
    ensure_glyph_map,
)
from core_pdf.impl.spec.s_09_fonts.widths import parse_font_widths

Matrix = list[float]


@dataclass(slots=True)
class LegacyFont:
    decoder: FontDecoder
    cmap: ToUnicodeCMap | None
    widths: FontWidthMap
    default_width: float
    space_width: float
    synthetic_space_width: float
    space_code_bytes: bytes
    encoding_table: tuple[str, ...] | None
    encoding_codec: str | None
    character_map: dict[str, str]
    difference_fallbacks: dict[bytes, str]
    width_uses_source_code: bool

    def decode_parts(self, data: bytes) -> tuple[tuple[str, ...], float]:
        glyphs = self.decoder.decode_glyphs(data)
        width = sum(self.internal_glyph_width(glyph) for glyph in glyphs)
        mapped_every_glyph = False
        if not self.decoder.is_cid_font and self.encoding_table is not None:
            parts = tuple(self.internal_simple_glyph_text(glyph) for glyph in glyphs)
        elif self.cmap is not None:
            mappings = self.cmap.mappings
            parts = tuple(
                mappings.get(
                    glyph.code_bytes,
                    self.internal_glyph_text(glyph),
                )
                for glyph in glyphs
            )
            mapped_every_glyph = bool(glyphs) and all(
                glyph.code_bytes in mappings for glyph in glyphs
            )
            if not glyphs:
                parts = tuple(self.cmap.decode(data, preserve_nulls=True))
        elif self.encoding_codec is not None:
            errors = "surrogatepass" if self.encoding_codec.startswith("utf-") else "replace"
            parts = tuple(data.decode(self.encoding_codec, errors=errors))
        else:
            parts = tuple(self.internal_glyph_text(glyph) for glyph in glyphs)
        # A ToUnicode CMap may deliberately map codes to no text at all: Arabic
        # shaping glyphs whose letters are emitted by a different run do exactly
        # that. Reading the bytes back as latin-1 would then paste raw CIDs into
        # the page, so only treat an empty result as a failed decode when the
        # CMap had nothing to say about it.
        if not any(parts) and not mapped_every_glyph:
            parts = tuple(data.decode("latin-1"))
        return parts, width

    def internal_simple_glyph_text(self, glyph: Any) -> str:
        mapped = self.cmap.mappings.get(glyph.code_bytes) if self.cmap is not None else None
        if mapped is not None:
            return mapped
        if len(glyph.code_bytes) != 1:
            return self.internal_glyph_text(glyph)
        table = self.encoding_table
        if table is None:
            return self.internal_glyph_text(glyph)
        encoded = table[glyph.code_bytes[0]]
        return "".join(self.character_map.get(character, character) for character in encoded)

    def internal_glyph_width(self, glyph: Any) -> float:
        table = self.encoding_table
        if not self.decoder.is_cid_font and table is not None and len(glyph.code_bytes) == 1:
            code = glyph.code_bytes[0]
            encoded = (
                chr(code)
                if self.cmap is not None and glyph.code_bytes in self.cmap.mappings
                else table[code]
            )
            # pypdf computes widths before applying ToUnicode. Its space
            # shortcut therefore applies only when the encoding itself emits
            # the encoded space character, not merely when ToUnicode maps the
            # source byte to U+0020.
            if glyph.code_bytes == self.space_code_bytes and encoded == chr(code):
                return self.space_width
            return float(
                sum(
                    int(self.widths.width_for(ord(character), self.default_width))
                    for character in encoded
                )
            )
        if glyph.code_bytes == self.space_code_bytes:
            return self.space_width
        width_code = glyph.char_code if self.width_uses_source_code else glyph.width_code
        width = self.widths.width_for(width_code, self.default_width)
        return float(width if self.decoder.is_cid_font else int(width))

    def internal_glyph_text(self, glyph: Any) -> str:
        fallback = self.difference_fallbacks.get(glyph.code_bytes)
        if fallback is not None and glyph.unicode in {"", "\ufffd"}:
            return fallback
        if glyph.unicode_source == "undefined" and len(glyph.code_bytes) == 1:
            return glyph.code_bytes.decode("latin-1")
        if glyph.split_unicode:
            return LIGATURES.get(glyph.unicode, glyph.unicode)
        return glyph.unicode


class LegacyTextExtractor:
    """Interpret text operators while retaining their original state boundaries."""

    def __init__(
        self,
        page: Any,
        resources: object | None = None,
        known_forms: set[int] | None = None,
        form_text_cache: dict[int, tuple[PdfStream, str]] | None = None,
    ) -> None:
        self.page = page
        self.document = page.document
        self.resources = resources if resources is not None else page.resolve_resources()
        self.known_forms = known_forms if known_forms is not None else set()
        # Extracted text per form XObject (keyed by identity, holding the stream
        # alive), shared across nested extractors for one page extraction. A
        # form's output ignores the outer graphics state, so repeating a Do of
        # the same form repeats the same text. Entries are recorded only for
        # top-level invocations: while ancestors are active, a recursive
        # reference is skipped and the result would not be reusable.
        self.form_text_cache = form_text_cache if form_text_cache is not None else {}
        self.fonts = self.internal_fonts(self.resources)
        self.cm: Matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.tm: Matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.previous_cm = self.cm.copy()
        self.previous_tm = self.tm.copy()
        self.stack: list[tuple[Matrix, LegacyFont | None, float, float]] = []
        self.font: LegacyFont | None = None
        self.font_size = 12.0
        self.half_space_width = 125.0
        self.leading = 0.0
        self.text = ""
        # Accumulated page output as parts plus its trailing character; joining
        # or copying the whole output per operator is quadratic.
        self.output_parts: list[str] = []
        self.output_last = ""
        self.rtl = False
        self.accumulated_width = 0.0
        self.actual_height = 0.0

    def internal_fonts(self, resources: object) -> dict[str, LegacyFont]:
        resolved_resources = self.document.resolver.resolve(resources)
        if not isinstance(resolved_resources, dict):
            return {}
        raw_fonts = self.document.resolver.resolve(resolved_resources.get("Font"))
        if not isinstance(raw_fonts, dict):
            return {}
        fonts: dict[str, LegacyFont] = {}
        for resource_name, raw_font in raw_fonts.items():
            font = self.document.resolver.resolve(raw_font)
            if not isinstance(font, dict):
                continue
            subtype = normalize_pdf_name(font.get("Subtype") or "")
            if subtype not in {"Type1", "MMType1", "TrueType", "Type3"}:
                descendants = self.document.resolver.resolve(font.get("DescendantFonts"))
                if not isinstance(descendants, (list, tuple)) or not descendants:
                    raise KeyError("DescendantFonts")
                descriptor_font = self.document.resolver.resolve(descendants[0])
            else:
                descriptor_font = font
            descriptor = (
                self.document.resolver.resolve(descriptor_font.get("FontDescriptor"))
                if isinstance(descriptor_font, dict)
                else None
            )
            if (
                isinstance(descriptor, dict)
                and sum(
                    descriptor.get(key) is not None
                    for key in ("FontFile", "FontFile2", "FontFile3")
                )
                > 1
            ):
                raise ValueError("font descriptor contains more than one font program")
            try:
                decoder = FontDecoder(self.document.resolver.resolve_font_dict(font))
            except (TypeError, ValueError):
                continue
            cmap: ToUnicodeCMap | None = None
            to_unicode = self.document.resolver.resolve(font.get("ToUnicode"))
            if isinstance(to_unicode, PdfStream):
                try:
                    cmap = ToUnicodeCMap(self.document.resolver.resolve_stream(to_unicode).data)
                except ValueError:
                    cmap = None
            encoding_table, encoding_codec, character_map = self.internal_legacy_encoding(
                font, decoder
            )
            raw_encoding = self.document.resolver.resolve(font.get("Encoding"))
            width_uses_source_code = decoder.is_cid_font and isinstance(raw_encoding, PdfStream)
            encoding_is_mapping = self.internal_encoding_is_mapping(font)
            widths, default_width, space_width = self.internal_font_widths(
                font, decoder, cmap, encoding_table, encoding_is_mapping
            )
            if decoder.is_cid_font:
                space_code_bytes = next(
                    (
                        code
                        for code, mapped in (cmap.mappings.items() if cmap is not None else ())
                        if mapped == " "
                    ),
                    b"\x00 " if encoding_codec == "utf-16-be" else b" ",
                )
            else:
                space_code_bytes = bytes(
                    (self.internal_space_code(cmap, encoding_table, encoding_is_mapping),)
                )
            synthetic_space_width = self.internal_synthetic_space_width(
                decoder,
                cmap,
                encoding_table,
                encoding_is_mapping,
                widths,
                default_width,
                space_width,
            )
            fonts[str(resource_name)] = LegacyFont(
                decoder,
                cmap,
                widths,
                default_width,
                space_width,
                synthetic_space_width,
                space_code_bytes,
                encoding_table,
                encoding_codec,
                character_map,
                self.internal_difference_fallbacks(font),
                width_uses_source_code,
            )
        return fonts

    def internal_synthetic_space_width(
        self,
        decoder: FontDecoder,
        cmap: ToUnicodeCMap | None,
        encoding_table: tuple[str, ...] | None,
        encoding_is_mapping: bool,
        widths: FontWidthMap,
        default_width: float,
        space_width: float,
    ) -> float:
        if decoder.is_cid_font:
            return space_width
        space_code = self.internal_space_code(cmap, encoding_table, encoding_is_mapping)
        if space_code == 32:
            return space_width
        return float(int(widths.width_for(32, default_width)))

    @staticmethod
    def internal_space_code(
        cmap: ToUnicodeCMap | None,
        encoding_table: tuple[str, ...] | None,
        encoding_is_mapping: bool,
    ) -> int:
        cmap_codes = set(cmap.mappings) if cmap is not None else set()
        if encoding_is_mapping and encoding_table is not None:
            encoded = next(
                (
                    code
                    for code, value in enumerate(encoding_table)
                    if value == " " and bytes((code,)) not in cmap_codes
                ),
                None,
            )
            if encoded is not None:
                return encoded
        if cmap is not None:
            mapped = next(
                (
                    code[0]
                    for code, value in cmap.mappings.items()
                    if value == " " and len(code) == 1
                ),
                None,
            )
            if mapped is not None:
                return mapped
        return 32

    def internal_encoding_is_mapping(self, font: dict[object, object]) -> bool:
        encoding = self.document.resolver.resolve(font.get("Encoding"))
        if isinstance(encoding, dict):
            return True
        encoding_name = normalize_pdf_name(encoding or "")
        if encoding_name in {
            "StandardEncoding",
            "WinAnsiEncoding",
            "MacRomanEncoding",
            "PDFDocEncoding",
            "Symbol",
            "ZapfDingbats",
        }:
            return True
        if encoding is not None:
            return False
        base_font = normalize_pdf_name(font.get("BaseFont") or "") or ""
        base_font = base_font.split("+", 1)[-1]
        return base_font in {
            "Courier",
            "Courier-Bold",
            "Courier-BoldOblique",
            "Courier-Oblique",
            "Helvetica",
            "Helvetica-Bold",
            "Helvetica-BoldOblique",
            "Helvetica-Oblique",
            "Times-Bold",
            "Times-BoldItalic",
            "Times-Italic",
            "Times-Roman",
            "Symbol",
            "ZapfDingbats",
        }

    def internal_legacy_encoding(
        self, font: dict[object, object], decoder: FontDecoder
    ) -> tuple[tuple[str, ...] | None, str | None, dict[str, str]]:
        if decoder.is_cid_font:
            encoding_name = normalize_pdf_name(font.get("Encoding") or "") or ""
            codec = internal_PREDEFINED_ENCODING_CODECS.get(encoding_name)
            if codec is None and "-UCS2-" in encoding_name:
                codec = "utf-16-be"
            return None, codec, {}
        encoding_obj = self.document.resolver.resolve(font.get("Encoding"))
        base_font = normalize_pdf_name(font.get("BaseFont"))
        if encoding_obj is None and base_font not in {"Symbol", "ZapfDingbats"}:
            table = [chr(code) for code in range(256)]
        else:
            table = internal_legacy_base_table(decoder.base_encoding or "StandardEncoding")
        character_map: dict[str, str] = {}

        for code, name in decoder.differences.items():
            if 0 <= code <= 255:
                table[code] = self.internal_legacy_glyph_name(name)

        subtype = normalize_pdf_name(font.get("Subtype") or "")
        descriptor = self.document.resolver.resolve(font.get("FontDescriptor"))
        has_type1_font_file = (
            isinstance(descriptor, dict) and descriptor.get("FontFile") is not None
        )
        if (
            subtype in {"Type1", "MMType1"}
            and font.get("ToUnicode") is None
            and has_type1_font_file
        ):
            character_map.update(self.internal_type1_character_map(descriptor))
            for character in character_map:
                table[ord(character)] = character
        return tuple(table), None, character_map

    def internal_type1_character_map(self, descriptor: dict[object, object]) -> dict[str, str]:
        """Project the clear-text Type 1 encoding accepted by pypdf."""
        raw_font_file = self.document.resolver.resolve(descriptor.get("FontFile"))
        if not isinstance(raw_font_file, PdfStream):
            return {}
        data = self.document.resolver.resolve_stream(raw_font_file).data
        clear_text = data.split(b"eexec\n", 1)[0]
        encoding_parts = clear_text.split(b"/Encoding")
        if len(encoding_parts) < 2:
            return {}
        result: dict[str, str] = {}
        for line in encoding_parts[1].replace(b"\r", b"\n").split(b"\n"):
            if not line.startswith(b"dup"):
                continue
            words = [word for word in line.split(b" ") if word]
            if len(words) < 3 or (len(words) > 3 and words[3] != b"put"):
                continue
            with suppress(ValueError):
                code = int(words[1])
                if not 0 <= code <= 255:
                    continue
                name = words[2].removeprefix(b"/").decode("latin-1")
                mapped = self.internal_legacy_glyph_name(name, unknown="")
                if not mapped and name.startswith("uni"):
                    mapped = chr(int(name[3:], 16))
                if mapped:
                    result[chr(code)] = mapped
        return result

    @staticmethod
    def internal_legacy_glyph_name(name: str, *, unknown: str | None = None) -> str:
        if name == "negationslash":
            return "⁄"
        # The engine reads cmex's wide tilde accents as U+02DC, which is the
        # accent rather than the ASCII punctuation. pypdf reports a plain tilde,
        # and this facade reports what pypdf reports. Its wide circumflexes
        # already agree, so only the tildes need saying.
        if name in {"tildewide", "tildewider", "tildewidest"}:
            return "~"
        if name == ".notdef":
            return "□"
        if name.startswith("a") and name[1:].isdecimal():
            code = int(name[1:])
            if 0 <= code <= 255:
                return chr(code)
        mapped = ensure_glyph_map().get(name)
        if mapped is None:
            mapped = TEX_GLYPH_ALIASES.get(name)
        if mapped is not None:
            return mapped
        return f"/{name}" if unknown is None else unknown

    def internal_difference_fallbacks(self, font: dict[object, object]) -> dict[bytes, str]:
        encoding = self.document.resolver.resolve(font.get("Encoding"))
        differences = (
            self.document.resolver.resolve(encoding.get("Differences"))
            if isinstance(encoding, dict)
            else None
        )
        if not isinstance(differences, (list, tuple)):
            return {}
        result: dict[bytes, str] = {}
        code: int | None = None
        for item in differences:
            if isinstance(item, (int, float)):
                code = int(item)
                continue
            if code is None or not 0 <= code <= 255:
                continue
            name = str(item)
            if name.isdecimal():
                result[bytes((code,))] = f"/{name}"
            code += 1
        return result

    def internal_font_widths(
        self,
        font: dict[object, object],
        decoder: FontDecoder,
        cmap: ToUnicodeCMap | None,
        encoding_table: tuple[str, ...] | None,
        encoding_is_mapping: bool,
    ) -> tuple[FontWidthMap, float, float]:
        subtype = normalize_pdf_name(font.get("Subtype") or "")
        widths = decoder.widths
        if decoder.is_type3:
            char_procs = self.document.resolver.resolve(font.get("CharProcs"))
            if (
                cmap is None
                and isinstance(char_procs, dict)
                and any(
                    not self.internal_legacy_glyph_name(
                        normalize_pdf_name(name) or str(name), unknown=""
                    )
                    for name in char_procs
                )
            ):
                # pypdf declares such a Type3 font uninterpretable and skips
                # its Widths array, but native text mode still emits its
                # declared encoding with the placeholder metrics.
                return SparseFontWidthMap(), 500.0, 200.0
            # The legacy API compares unscaled Widths values. The canonical
            # engine scales Type3 metrics through FontMatrix for geometry.
            with suppress(ValueError):
                widths = parse_font_widths(cast(Any, font), subtype).widths

        default_width = decoder.default_width
        if not decoder.is_cid_font:
            descriptor = self.document.resolver.resolve(font.get("FontDescriptor"))
            missing_width = (
                self.document.resolver.resolve(descriptor.get("MissingWidth"))
                if isinstance(descriptor, dict)
                else None
            )
            flags = (
                self.document.resolver.resolve(descriptor.get("Flags"))
                if isinstance(descriptor, dict)
                else 0
            )
            positive_widths = [
                int(width) for _, width in widths.iter_explicit_widths() if int(width) > 0
            ]
            space_code = self.internal_space_code(cmap, encoding_table, encoding_is_mapping)
            raw_space = widths.get(space_code)
            space = int(raw_space) if raw_space is not None else 0
            if isinstance(missing_width, (int, float)) and missing_width:
                default_width = float(int(missing_width))
            elif space:
                default_width = float(space if int(flags or 0) & 1 else 2 * space)
            elif positive_widths:
                default_width = float(sum(positive_widths) // len(positive_widths))
            else:
                default_width = 500.0

        if decoder.is_cid_font:
            space_codes = [
                int.from_bytes(code, "big")
                for code, mapped in (cmap.mappings.items() if cmap is not None else ())
                if mapped == " "
            ] or [32]
        else:
            space_codes = [self.internal_space_code(cmap, encoding_table, encoding_is_mapping)]
        for code in space_codes:
            # pypdf uses the explicit width attached to the encoded space for
            # both simple and CID fonts. Only a genuinely missing/zero entry
            # falls back to the extraction API's 200-unit default.
            width = widths.get(code)
            legacy_width = width if decoder.is_cid_font else int(width or 0)
            if legacy_width:
                return widths, default_width, float(legacy_width)
        # extract_text's public ``space_width=200`` override replaces the
        # synthesized Font.space_width whenever the encoded space has no
        # explicit width entry. Keep that extraction-layer behavior separate
        # from the font's default glyph width above.
        return widths, default_width, 200.0

    def flush(self) -> None:
        self.text, self.output_last = internal_flush_text(
            self.output_parts, self.text, self.output_last
        )

    def add_text(self, value: str) -> None:
        for character in value:
            self.add_text_unit(character)

    def add_text_unit(self, value: str) -> None:
        # Native pypdf sends a preceding directional run only to visitor_text
        # when direction changes. The default API intentionally discards it.
        self.text, self.rtl = internal_append_directional_text(self.text, self.rtl, value)

    def check_position(self, string_width: float) -> None:
        self.text, self.output_last = internal_positioned_text(
            self.output_parts,
            self.text,
            self.output_last,
            previous_text_matrix=self.previous_tm,
            previous_current_matrix=self.previous_cm,
            text_matrix=self.tm,
            current_matrix=self.cm,
            line_height=self.actual_height,
            font_size=self.font_size,
            space_width=self.current_space_width,
            string_width=string_width,
        )
        self.previous_tm = self.tm.copy()
        self.previous_cm = self.cm.copy()

    @property
    def current_space_width(self) -> float:
        return self.half_space_width

    def show(self, data: bytes) -> None:
        if self.font is None:
            parts, width = tuple(data.decode("latin-1")), len(data) * 500.0
        else:
            parts, width = self.font.decode_parts(data)
        for part in parts:
            self.add_text_unit(part)
        self.accumulated_width += width * self.font_size
        self.actual_height = self.font_size
        self.check_position(0.0)

    def move_text(self, tx: float, ty: float) -> None:
        self.tm[4] += tx * self.tm[0] + ty * self.tm[2]
        self.tm[5] += tx * self.tm[1] + ty * self.tm[3]
        self.check_position(self.accumulated_width / 1000.0)
        self.accumulated_width = 0.0

    def process(self, operator: str, operands: tuple[object, ...]) -> None:  # noqa: C901
        if operator == "BT":
            self.tm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            self.flush()
        elif operator == "ET":
            self.flush()
        elif operator == "q":
            self.stack.append((self.cm.copy(), self.font, self.font_size, self.leading))
        elif operator == "Q":
            if self.stack:
                self.cm, self.font, self.font_size, self.leading = self.stack.pop()
            else:
                self.cm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        elif operator == "cm":
            self.flush()
            try:
                values = [float(cast(Any, value)) for value in operands[:6]]
            except (TypeError, ValueError):
                values = []
            self.cm = (
                multiply_affine(values, self.cm)
                if len(values) == 6
                else [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            )
        elif operator == "Tf":
            self.flush()
            if operands:
                self.font = self.fonts.get(str(operands[0]))
                self.half_space_width = (
                    self.font.space_width if self.font is not None else 250.0
                ) / 2.0
            if len(operands) > 1:
                self.font_size = float(cast(Any, operands[1]))
        elif operator == "TL":
            scale_x = math.hypot(self.tm[0], self.tm[2])
            self.leading = (
                float(cast(Any, operands[0])) * self.font_size * scale_x if operands else 0.0
            )
        elif operator in {"Td", "TD"}:
            tx = float(cast(Any, operands[0])) if operands else 0.0
            ty = float(cast(Any, operands[1])) if len(operands) > 1 else 0.0
            if operator == "TD":
                self.process("TL", (-ty,))
            self.move_text(tx, ty)
        elif operator == "Tm":
            values = [float(cast(Any, value)) for value in operands[:6]]
            self.tm = values if len(values) == 6 else [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            self.check_position(self.accumulated_width / 1000.0)
            self.accumulated_width = 0.0
        elif operator == "T*":
            self.move_text(0.0, -self.leading)
        elif operator == "Tj":
            value = operands[0] if operands else None
            self.show(bytes(value.data) if isinstance(value, PdfString) else b"")
        elif operator == "TJ":
            threshold = self.current_space_width * 0.95
            for item in cast(list[object], operands[0] if operands else []):
                if isinstance(item, PdfString):
                    self.show(bytes(item.data))
                elif isinstance(item, PdfName):
                    # Malformed producers occasionally place a name where TJ
                    # requires a string. pypdf preserves that operand's PDF
                    # spelling in extracted text.
                    self.add_text(f"/{item.value}")
                elif (
                    isinstance(item, (int, float))
                    and abs(float(item)) >= threshold
                    and self.text
                    and not self.text.endswith(" ")
                ):
                    self.add_text(" ")
                    self.accumulated_width += (
                        self.font.synthetic_space_width if self.font is not None else 250.0
                    ) * self.font_size
                    self.actual_height = self.font_size
                    self.check_position(0.0)
        elif operator == "'":
            self.process("T*", ())
            self.process("Tj", operands)
        elif operator == '"':
            self.process("T*", ())
            self.process("Tj", operands[2:3])

    def extract(self, streams: tuple[PdfStream, ...] | None = None) -> str:
        content_streams = streams if streams is not None else self.page.content_streams
        # A page Contents array is one logical content stream. Operators and
        # their operands may legally straddle physical stream boundaries, so
        # preserve the token state by joining them with a whitespace separator.
        data = b"\n".join(stream.data for stream in content_streams)
        inline_image = re.search(rb"(?<!\S)BI(?=\s)", data)
        if (
            inline_image is not None
            and re.search(rb"(?<!\S)EI(?:\s|$)", data[inline_image.end() :]) is None
        ):
            raise ValueError("unexpected end of inline image stream")
        for operator, operands in iter_content_operations(PdfLexer(data)):
            if operator == "Do":
                self.flush()
                self.output_last = internal_ensure_line_break(self.output_parts, self.output_last)
                self.internal_form(operands)
                self.text = ""
            else:
                self.process(operator, operands)
        self.flush()
        return "".join(self.output_parts)

    def internal_form(self, operands: tuple[object, ...]) -> None:
        if not operands:
            return
        resources = self.document.resolver.resolve(self.resources)
        if not isinstance(resources, dict):
            return
        xobjects = self.document.resolver.resolve(resources.get("XObject"))
        if not isinstance(xobjects, dict):
            return
        xobject = self.document.resolver.resolve(xobjects.get(operands[0]))
        if not isinstance(xobject, PdfStream):
            return
        if str(xobject.dictionary.get("Subtype")) == "Image":
            return
        form_id = id(xobject)
        if form_id in self.known_forms:
            return
        top_level = not self.known_forms
        form_resources = xobject.dictionary.get("Resources")
        if form_resources is None:
            return
        cached = self.form_text_cache.get(form_id) if top_level else None
        if cached is not None:
            child_text = cached[1]
        else:
            stream = self.document.resolver.resolve_stream(xobject)
            self.known_forms.add(form_id)
            try:
                child = LegacyTextExtractor(
                    self.page, form_resources, self.known_forms, self.form_text_cache
                )
                child_text = child.extract((stream,))
            finally:
                self.known_forms.discard(form_id)
            if top_level:
                self.form_text_cache[form_id] = (xobject, child_text)
        if child_text:
            self.output_parts.append(child_text)
            self.output_last = child_text[-1]


def extract_legacy_text(page: Any) -> str:
    """Extract page/form text in content-stream operation order."""
    return LegacyTextExtractor(page).extract()
