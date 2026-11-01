from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from contextlib import suppress
from functools import cache
from importlib.util import find_spec
from typing import Any

from core_adobe_fonts.afm.core14 import FONT_DATA
from core_adobe_fonts.agl.glyph_list import GLYPH_DATA
from core_adobe_fonts.cff.font import CFF_STANDARD_STRINGS, CFFFont
from core_pdf.impl.capture_recovery import iter_content_operations
from core_pdf.impl.fonts_cmap import ToUnicodeCMap
from core_pdf.impl.fonts_decoder import FontDecoder
from core_pdf.impl.fonts_helpers import TEX_GLYPH_ALIASES, recover_strip_subset_tag
from core_pdf.impl.fonts_metrics import LIGATURE_TEXT_TO_CHAR
from core_pdf.impl.pdf_values import recover_pdf_name
from core_pdf.impl.recovery_lexer import PdfLexer
from core_pdf.impl.types import PdfName, PdfString, RecordType, ReplaceFields, ReprFields
from core_pdf_compat._text_state import (
    IDENTITY_MATRIX,
    TextMachine,
    embedded_font_program_count,
    ensure_line_break,
    legacy_base_table,
    number_operand,
    orientation,
    predefined_encoding_codec,
    type1_encoding_entries,
)
from core_pdf_compat.pypdf._cmap_scan import scan_to_unicode
from core_pdf_compat.pypdf._layout import layout_text
from core_pdf_compat.pypdf.generic import to_generic
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.matrix import multiply_affine
from core_records import GeneratedRecord


class LegacyFont(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
    decoder: FontDecoder
    cmap: ToUnicodeCMap | None
    widths: Mapping[int, float]
    default_width: float
    space_width: float
    natural_space_width: float
    synthetic_space_width: float
    space_code_bytes: bytes
    encoding_table: tuple[str, ...] | None
    encoding_codec: str | None
    character_map: dict[str, str]
    difference_fallbacks: dict[bytes, str]
    width_uses_source_code: bool

    __hash__ = None  # type: ignore[assignment]

    def decode_parts(self, data: bytes) -> tuple[tuple[str, ...], float]:
        glyphs = self.decoder.decode_glyphs(data)
        width = sum(self.glyph_width(glyph) for glyph in glyphs)
        mapped_every_glyph = False
        if not self.decoder.is_cid_font and self.encoding_table is not None:
            parts = tuple(self.simple_glyph_text(glyph) for glyph in glyphs)
        elif self.cmap is not None:
            mappings = self.cmap.mappings
            parts = tuple(
                mappings.get(
                    glyph.code_bytes,
                    self.glyph_text(glyph),
                )
                for glyph in glyphs
            )
            mapped_every_glyph = bool(glyphs) and all(
                glyph.code_bytes in mappings for glyph in glyphs
            )
            if not glyphs:
                parts = tuple(self.cmap.decode(data, preserve_nulls=True))
        elif self.encoding_codec is not None:
            try:
                decoded = data.decode(self.encoding_codec, "surrogatepass")
            except UnicodeDecodeError:
                # pypdf's fallback for odd byte counts and unmapped codes.
                decoded = data.decode(self.encoding_codec, "surrogateescape")
            parts = tuple(decoded)
        else:
            parts = tuple(self.glyph_text(glyph) for glyph in glyphs)
        if not any(parts) and not mapped_every_glyph:
            parts = tuple(data.decode("latin-1"))
        return parts, width

    def simple_glyph_text(self, glyph: Any) -> str:
        mapped = self.cmap.mappings.get(glyph.code_bytes) if self.cmap is not None else None
        if mapped is not None:
            return mapped
        if len(glyph.code_bytes) != 1:
            return self.glyph_text(glyph)
        table = self.encoding_table
        if table is None:
            return self.glyph_text(glyph)
        encoded = table[glyph.code_bytes[0]]
        return "".join(self.character_map.get(character, character) for character in encoded)

    def glyph_width(self, glyph: Any) -> float:
        if (
            not self.decoder.is_cid_font
            and self.encoding_table is not None
            and len(glyph.code_bytes) == 1
        ):
            # pypdf sums the widths of the raw codes, not of the decoded text.
            if glyph.code_bytes == self.space_code_bytes:
                return self.space_width
            return float(int(self.widths.get(glyph.code_bytes[0], self.default_width)))
        if glyph.code_bytes == self.space_code_bytes:
            return self.space_width
        width_code = glyph.char_code if self.width_uses_source_code else glyph.width_code
        width = self.widths.get(width_code, self.default_width)
        return float(width if self.decoder.is_cid_font else int(width))

    def glyph_text(self, glyph: Any) -> str:
        fallback = self.difference_fallbacks.get(glyph.code_bytes)
        if fallback is not None and glyph.unicode in {"", "\ufffd"}:
            return fallback
        if glyph.unicode_source == "undefined" and len(glyph.code_bytes) == 1:
            return glyph.code_bytes.decode("latin-1")
        if glyph.split_unicode:
            return LIGATURE_TEXT_TO_CHAR.get(glyph.unicode, glyph.unicode)
        return glyph.unicode


# The fixed-pitch fonts among pypdf's built-in core font metrics, whose flags it
# uses for a font without /Widths or /FontDescriptor.
REFERENCE_FIXED_PITCH_CORE_FONTS = frozenset(
    {
        "Courier",
        "Courier-Bold",
        "Courier-BoldOblique",
        "Courier-Oblique",
        "CourierNew",
        "CourierNew,Bold",
        "CourierNew,BoldItalic",
        "CourierNew,Italic",
    }
)


def cmap_key_character(code: bytes) -> str:
    """The character pypdf keys a ToUnicode source code by."""
    return code.decode("charmap" if len(code) == 1 else "utf-16-be", "surrogatepass")


@cache
def reference_reads_cff_encodings() -> bool:
    """pypdf reads a CFF font program's built-in encoding only with fontTools."""
    return find_spec("fontTools") is not None


def reference_cff_encoding(data: bytes) -> dict[int, str]:
    """The custom built-in encoding fontTools reads from a CFF font program.

    pypdf maps the encoding fontTools decodes, so this follows fontTools rather
    than the CFF specification: later codes overwrite earlier ones, a format 0
    encoding never assigns code 0, and any malformed table yields nothing. A
    predefined Standard or Expert encoding is not mapped either.
    """
    try:
        cff = CFFFont(data)
        position = cff.dict_offset(16, default=0)
    except ValueError:
        return {}
    if cff.is_cid_keyed or position in {0, 1}:
        return {}
    names = dict(enumerate(CFF_STANDARD_STRINGS)) | {
        sid: name for name, sid in cff.custom_string_sids.items()
    }
    charset = {gid: names.get(sid) for sid, gid in cff.cid_to_gid.items()}
    program = cff.data
    encoding: dict[int, str | None] = {}
    try:
        raw_format = program[position]
        count = program[position + 1]
        position += 2
        if raw_format & 127 == 0:
            for gid in range(1, count + 1):
                code = program[position]
                position += 1
                if code != 0:
                    encoding[code] = charset[gid]
        elif raw_format & 127 == 1:
            gid = 1
            for _ in range(count):
                first, left = program[position], program[position + 1]
                position += 2
                if first + left > 255:
                    return {}
                for offset in range(left + 1):
                    encoding[first + offset] = charset[gid + offset]
                gid += left + 1
        else:
            return {}
        if raw_format & 128:
            supplements = program[position]
            position += 1
            for _ in range(supplements):
                code = program[position]
                sid = int.from_bytes(program[position + 1 : position + 3], "big")
                position += 3
                encoding[code] = names[sid]
    except IndexError, KeyError:
        return {}
    return {code: name for code, name in encoding.items() if name is not None}


logger = logging.getLogger(__name__)

ALL_ORIENTATIONS = (0, 90, 180, 270)
DEFAULT_SPACE_WIDTH = 200.0

# pypdf stops descending into form XObjects after this many per page extraction
# (pypdf Configuration.xform_maximum_invocations_per_extraction).
MAX_FORM_INVOCATIONS = 5_000


class FormInvocations:
    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0


class LegacyTextCaches:
    __slots__ = ("fonts", "forms")

    def __init__(self) -> None:
        self.fonts: dict[int, tuple[object, LegacyFont | None]] = {}
        self.forms: dict[int, tuple[PdfStream, str]] = {}


class TextOptions(GeneratedRecord):
    """The extract_text arguments. ``reader`` turns visited objects into pypdf's."""

    orientations: tuple[int, ...] = ALL_ORIENTATIONS
    visitor_operand_before: Callable[[Any, Any, Any, Any], None] | None = None
    visitor_operand_after: Callable[[Any, Any, Any, Any], None] | None = None
    visitor_text: Callable[[Any, Any, Any, Any, Any], None] | None = None
    reader: Any = None

    @property
    def is_plain(self) -> bool:
        return self == TextOptions(reader=self.reader)


class LegacyTextExtractor(TextMachine[LegacyFont]):
    def __init__(
        self,
        page: Any,
        resources: object | None = None,
        known_forms: set[int] | None = None,
        caches: LegacyTextCaches | None = None,
        invocations: FormInvocations | None = None,
        options: TextOptions | None = None,
    ) -> None:
        self.page = page
        self.options = TextOptions() if options is None else options
        self.memo_cm = list(IDENTITY_MATRIX)
        self.memo_tm = list(IDENTITY_MATRIX)
        self.font_name: str | None = None
        self.document = page.document
        self.resources = resources if resources is not None else page.resources
        self.known_forms = known_forms if known_forms is not None else set()
        self.caches = caches if caches is not None else LegacyTextCaches()
        self.invocations = invocations if invocations is not None else FormInvocations()
        super().__init__(self.collect_fonts(self.resources))

    def collect_fonts(self, resources: object) -> dict[str, LegacyFont]:
        resolver = self.document.resolver
        resolved_resources = resolver.as_dict(resources)
        if resolved_resources is None:
            return {}
        raw_fonts = resolver.dict_at(resolved_resources, "Font")
        if raw_fonts is None:
            return {}
        fonts: dict[str, LegacyFont] = {}
        font_cache = self.caches.fonts
        as_dict = resolver.as_dict
        for resource_name, raw_font in raw_fonts.items():
            font = as_dict(raw_font)
            if font is None:
                continue
            known = font_cache.get(id(font))
            if known is None or known[0] is not font:
                known = font_cache[id(font)] = (font, self.legacy_font(font))
            if known[1] is not None:
                fonts[str(resource_name)] = known[1]
        return fonts

    def legacy_font(self, font: dict[object, object]) -> LegacyFont | None:
        subtype = recover_pdf_name(font.get("Subtype") or "")
        if subtype not in {"Type1", "MMType1", "TrueType", "Type3"}:
            descendants = self.document.resolver.resolve(font.get("DescendantFonts"))
            # pypdf ignores a missing or malformed /DescendantFonts array.
            descriptor_font = (
                self.document.resolver.resolve(descendants[0])
                if isinstance(descendants, (list, tuple)) and descendants
                else None
            )
        else:
            descriptor_font = font
        descriptor = (
            self.document.resolver.resolve(descriptor_font.get("FontDescriptor"))
            if isinstance(descriptor_font, dict)
            else None
        )
        if isinstance(descriptor, dict) and embedded_font_program_count(descriptor) > 1:
            raise ValueError("font descriptor contains more than one font program")
        try:
            decoder = FontDecoder(self.document.resolver.resolve_font_dict(font))
        except TypeError, ValueError:
            return None
        cmap: ToUnicodeCMap | None = None
        to_unicode = self.document.resolver.resolve(font.get("ToUnicode"))
        if isinstance(to_unicode, PdfStream):
            data = self.document.resolver.resolve_stream(to_unicode).data
            scan_to_unicode(data)
            try:
                cmap = ToUnicodeCMap(data)
            except ValueError:
                cmap = None
        encoding_table, encoding_codec, character_map = self.legacy_encoding(font, decoder)
        raw_encoding = self.document.resolver.resolve(font.get("Encoding"))
        width_uses_source_code = decoder.is_cid_font and isinstance(raw_encoding, PdfStream)
        encoding_is_mapping = self.encoding_is_mapping(font)
        space_code = (
            self.resolve_cid_space_code(cmap)
            if decoder.is_cid_font
            else self.resolve_space_code(cmap, character_map, encoding_table, encoding_is_mapping)
        )
        widths, default_width, space_width, fixed_pitch = self.font_widths(
            font, decoder, cmap, space_code
        )
        # pypdf overrides a zero space width for plain extraction only.
        natural_space_width = space_width
        if space_width == DEFAULT_SPACE_WIDTH:
            natural_space_width = default_width if fixed_pitch else float(default_width // 2)
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
            space_code_bytes = bytes((space_code,)) if space_code is not None else b""
        synthetic_space_width = self.resolve_synthetic_space_width(
            space_code, widths, default_width, space_width, cid=decoder.is_cid_font
        )
        return LegacyFont(
            decoder,
            cmap,
            widths,
            default_width,
            space_width,
            natural_space_width,
            synthetic_space_width,
            space_code_bytes,
            encoding_table,
            encoding_codec,
            character_map,
            self.difference_fallbacks(font),
            width_uses_source_code,
        )

    @staticmethod
    def resolve_synthetic_space_width(
        space_code: int | None,
        widths: Mapping[int, float],
        default_width: float,
        space_width: float,
        *,
        cid: bool,
    ) -> float:
        # pypdf inserts a literal " ": it measures as the font's space only
        # when that is code 32, and otherwise as the width of code 32.
        if space_code == 32:
            return space_width
        width = widths.get(32, default_width)
        return float(width if cid else int(width))

    @staticmethod
    def resolve_cid_space_code(cmap: ToUnicodeCMap | None) -> int:
        # pypdf's space is the first ToUnicode entry for one, else code 32.
        mapped = next(
            (
                code
                for code, value in (cmap.mappings.items() if cmap is not None else ())
                if value == " "
            ),
            None,
        )
        return int.from_bytes(mapped, "big") if mapped is not None else 32

    @staticmethod
    def resolve_space_code(
        cmap: ToUnicodeCMap | None,
        character_map: Mapping[str, str],
        encoding_table: tuple[str, ...] | None,
        encoding_is_mapping: bool,
    ) -> int | None:
        # pypdf takes the first entry for a space in its character map (the
        # ToUnicode CMap, or else the map read from a Type1 font program), then
        # the encoding.
        if cmap is not None:
            mapped = next((code for code, value in cmap.mappings.items() if value == " "), None)
            if mapped is not None:
                # Simple fonts compare the key with single-byte codes.
                character = cmap_key_character(mapped)
                return ord(character) if len(character) == 1 and ord(character) < 256 else None
        else:
            key = next((key for key, value in character_map.items() if value == " "), None)
            if key is not None:
                return ord(key)
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
        return 32

    def encoding_is_mapping(self, font: dict[object, object]) -> bool:
        encoding = self.document.resolver.resolve(font.get("Encoding"))
        if isinstance(encoding, dict):
            return True
        encoding_name = recover_pdf_name(encoding or "")
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
        base_font = recover_strip_subset_tag(recover_pdf_name(font.get("BaseFont") or "") or "")
        return base_font in FONT_DATA

    def legacy_encoding(
        self, font: dict[object, object], decoder: FontDecoder
    ) -> tuple[tuple[str, ...] | None, str | None, dict[str, str]]:
        if decoder.is_cid_font:
            encoding_name = recover_pdf_name(font.get("Encoding") or "") or ""
            return None, predefined_encoding_codec(encoding_name), {}
        encoding_obj = self.document.resolver.resolve(font.get("Encoding"))
        base_font = recover_pdf_name(font.get("BaseFont"))
        if encoding_obj is None and base_font not in {"Symbol", "ZapfDingbats"}:
            # pypdf falls back to StandardEncoding when a font has no /Encoding.
            table = legacy_base_table("StandardEncoding")
        else:
            table = legacy_base_table(decoder.encoding.base_encoding or "StandardEncoding")
        character_map: dict[str, str] = {}

        for code, name in decoder.encoding.differences.items():
            if 0 <= code <= 255:
                table[code] = self.legacy_glyph_name(name)

        subtype = recover_pdf_name(font.get("Subtype") or "")
        descriptor = self.document.resolver.resolve(font.get("FontDescriptor"))
        if (
            subtype in {"Type1", "MMType1"}
            and font.get("ToUnicode") is None
            and isinstance(descriptor, dict)
        ):
            character_map.update(self.font_file_character_map(descriptor))
            for character in character_map:
                table[ord(character)] = character
        return tuple(table), None, character_map

    def font_file_character_map(self, descriptor: dict[object, object]) -> dict[str, str]:
        # pypdf reads the first of /FontFile and a /FontFile3 of subtype Type1C
        # that is a stream, and stops there even when it maps nothing.
        resolve = self.document.resolver.resolve
        font_file = resolve(descriptor.get("FontFile"))
        if isinstance(font_file, PdfStream):
            return self.type1_character_map(font_file)
        font_file = resolve(descriptor.get("FontFile3"))
        if (
            isinstance(font_file, PdfStream)
            and recover_pdf_name(font_file.dictionary.get("Subtype") or "") == "Type1C"
            and reference_reads_cff_encodings()
        ):
            return self.cff_character_map(font_file)
        return {}

    def cff_character_map(self, font_file: PdfStream) -> dict[str, str]:
        encoding = reference_cff_encoding(self.document.resolver.resolve_stream(font_file).data)
        result: dict[str, str] = {}
        for code, name in sorted(encoding.items()):
            if not name or name == ".notdef":
                continue
            mapped = self.legacy_glyph_name(name, unknown="")
            if not mapped and name.startswith("uni"):
                with suppress(ValueError):
                    mapped = chr(int(name[3:], 16))
            if mapped:
                result[chr(code)] = mapped
        return result

    def type1_character_map(self, font_file: PdfStream) -> dict[str, str]:
        data = self.document.resolver.resolve_stream(font_file).data
        clear_text = data.split(b"eexec\n", 1)[0]
        encoding_parts = clear_text.split(b"/Encoding")
        if len(encoding_parts) < 2:
            return {}
        result: dict[str, str] = {}
        for code, name in type1_encoding_entries(encoding_parts[1]):
            with suppress(ValueError):
                mapped = self.legacy_glyph_name(name, unknown="")
                if not mapped and name.startswith("uni"):
                    mapped = chr(int(name[3:], 16))
                if mapped:
                    result[chr(code)] = mapped
        return result

    @staticmethod
    def legacy_glyph_name(name: str, *, unknown: str | None = None) -> str:
        if name == "negationslash":
            return "⁄"
        if name in {"tildewide", "tildewider", "tildewidest"}:
            return "~"
        if name == ".notdef":
            return "□"
        if name.startswith("a") and name[1:].isdecimal():
            code = int(name[1:])
            if 0 <= code <= 255:
                return chr(code)
        mapped = GLYPH_DATA.get(name)
        if mapped is None:
            mapped = TEX_GLYPH_ALIASES.get(name)
        if mapped is not None:
            return mapped
        return f"/{name}" if unknown is None else unknown

    def difference_fallbacks(self, font: dict[object, object]) -> dict[bytes, str]:
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

    def font_widths(
        self,
        font: dict[object, object],
        decoder: FontDecoder,
        cmap: ToUnicodeCMap | None,
        space_code: int | None,
    ) -> tuple[Mapping[int, float], float, float, bool]:
        widths = decoder.metrics.widths
        fixed_pitch = False
        if decoder.is_type3:
            char_procs = self.document.resolver.resolve(font.get("CharProcs"))
            if (
                cmap is None
                and isinstance(char_procs, dict)
                and any(
                    not self.legacy_glyph_name(recover_pdf_name(name) or str(name), unknown="")
                    for name in char_procs
                )
            ):
                return {}, 500.0, DEFAULT_SPACE_WIDTH, False
            # pypdf uses the raw /Widths of a Type3 font, unscaled by /FontMatrix.
            widths = self.raw_simple_widths(font)

        default_width = decoder.metrics.default_width
        if not decoder.is_cid_font:
            descriptor = self.document.resolver.resolve(font.get("FontDescriptor"))
            missing_width = (
                self.document.resolver.resolve(descriptor.get("MissingWidth"))
                if isinstance(descriptor, dict)
                else None
            )
            if isinstance(descriptor, dict):
                flags = self.document.resolver.resolve(descriptor.get("Flags"))
            else:
                base_font = recover_pdf_name(font.get("BaseFont") or "")
                fixed = font.get("Widths") is None and base_font in REFERENCE_FIXED_PITCH_CORE_FONTS
                flags = 1 if fixed else 0
            fixed_pitch = bool(int(flags or 0) & 1)
            positive_widths = [int(width) for _, width in widths.items() if int(width) > 0]
            raw_space = widths.get(space_code) if space_code is not None else None
            space = int(raw_space) if raw_space is not None else 0
            if isinstance(missing_width, (int, float)) and missing_width:
                default_width = float(int(missing_width))
            elif space:
                default_width = float(space if fixed_pitch else 2 * space)
            elif positive_widths:
                default_width = float(sum(positive_widths) // len(positive_widths))
            else:
                default_width = 500.0

        if decoder.is_cid_font and space_code is not None:
            default_width = self.cid_default_width(font, widths, space_code)
        width = widths.get(space_code) if space_code is not None else None
        legacy_width = width if decoder.is_cid_font else int(width or 0)
        if legacy_width:
            return widths, default_width, float(legacy_width), fixed_pitch
        return widths, default_width, DEFAULT_SPACE_WIDTH, fixed_pitch

    def raw_simple_widths(self, font: dict[object, object]) -> dict[int, float]:
        resolve = self.document.resolver.resolve
        raw_widths = resolve(font.get("Widths"))
        first_char = resolve(font.get("FirstChar", 0))
        if (
            not isinstance(raw_widths, (list, tuple))
            or isinstance(first_char, bool)
            or not isinstance(first_char, (int, float))
            or first_char < 0
        ):
            return {}
        widths: dict[int, float] = {}
        for code, raw_width in enumerate(raw_widths, start=int(first_char)):
            if code > 255:
                break
            width = resolve(raw_width)
            if isinstance(width, (int, float)) and not isinstance(width, bool):
                widths[code] = float(int(width))
        return widths

    def cid_default_width(
        self, font: dict[object, object], widths: Mapping[int, float], space_code: int
    ) -> float:
        # pypdf takes a nonzero /DW, else derives the default from the space
        # width (doubled unless fixed pitch) or the average width.
        resolve = self.document.resolver.resolve
        descendants = resolve(font.get("DescendantFonts"))
        descendant = (
            self.document.resolver.as_dict(descendants[0])
            if isinstance(descendants, (list, tuple)) and descendants
            else None
        )
        if descendant is None:
            return 500.0
        default = resolve(descendant.get("DW"))
        if isinstance(default, (int, float)) and not isinstance(default, bool) and default:
            return float(default)
        if not widths:
            return 500.0
        descriptor = resolve(descendant.get("FontDescriptor"))
        flags = resolve(descriptor.get("Flags")) if isinstance(descriptor, dict) else 0
        space = widths.get(space_code, 0)
        if space:
            return float(space if int(flags or 0) & 1 else int(2 * space))
        positive = [width for width in widths.values() if width > 0]
        return float(sum(positive) // len(positive)) if positive else 500.0

    def add_text(self, value: str) -> None:
        for character in value:
            self.add_text_unit(character)

    def add_text_unit(self, value: str) -> None:
        self.add_directional(value)

    def font_resource(self) -> Any:
        """The current font as pypdf's dictionary, which visitors receive."""
        if self.font_name is None:
            return None
        resolver = self.document.resolver
        resources = resolver.as_dict(self.resources)
        fonts = None if resources is None else resolver.dict_at(resources, "Font")
        font = None if fonts is None else resolver.as_dict(fonts.get(self.font_name))
        return to_generic(font, self.options.reader) if font is not None else None

    def visit(
        self, text: str, cm: list[float] | None = None, tm: list[float] | None = None
    ) -> None:
        visitor = self.options.visitor_text
        if visitor is not None:
            cm, tm = cm or self.memo_cm, tm or self.memo_tm
            visitor(text, cm, tm, self.font_resource(), self.font_size)

    def remember_position(self) -> None:
        self.memo_cm = self.cm.copy()
        self.memo_tm = self.tm.copy()

    def flush(self) -> None:
        self.visit(self.text)
        super().flush()
        self.remember_position()

    def add_directional(self, value: str) -> None:
        pending = self.text
        rtl = self.rtl
        super().add_directional(value)
        if self.rtl != rtl:
            self.visit(pending, self.cm, self.tm)

    def visible(self) -> bool:
        current = multiply_affine(self.tm, self.cm)
        return orientation(current) in self.options.orientations

    def positioned(self, string_width: float) -> None:
        if not self.visible():
            return
        text, parts = self.text, len(self.output_parts)
        super().positioned(string_width)
        if len(self.output_parts) > parts:
            self.visit(text + "\n")
        if not self.text:
            self.remember_position()

    def show(self, data: bytes) -> None:
        if not self.visible():
            self.actual_height = self.font_size
            self.positioned(0.0)
            return
        if self.font is None:
            parts, width = tuple(data.decode("latin-1")), len(data) * 500.0
        else:
            parts, width = self.font.decode_parts(data)
        for part in parts:
            self.add_text_unit(part)
        self.accumulated_width += width * self.font_size
        self.actual_height = self.font_size
        self.positioned(0.0)

    def move_text(self, tx: float, ty: float) -> None:
        self.tm[4] += tx * self.tm[0] + ty * self.tm[2]
        self.tm[5] += tx * self.tm[1] + ty * self.tm[3]
        self.positioned(self.accumulated_width / 1000.0)
        self.accumulated_width = 0.0

    def process(self, operator: str, operands: tuple[object, ...]) -> None:
        if self.apply_state_operator(operator, operands):
            return
        match operator:
            case "Tf":
                if self.text:
                    self.flush()
                else:
                    self.remember_position()
                if operands:
                    self.font = self.fonts.get(str(operands[0]))
                    self.font_name = str(operands[0]) if self.font is not None else None
                    self.half_space_width = (
                        self.font.space_width if self.font is not None else 250.0
                    ) / 2.0
                if len(operands) > 1:
                    self.font_size = float(operands[1])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            case "Td":
                self.move_text(number_operand(operands, 0, 0.0), number_operand(operands, 1, 0.0))
            case "TD" if len(operands) >= 2:
                # pypdf negates the raw operand, so a non-number fails extraction.
                ty = operands[1]
                if isinstance(ty, bool) or not isinstance(ty, (int, float)):
                    raise TypeError(f"bad operand type for unary -: {type(ty).__name__!r}")
                self.process("TL", (-ty,))
                self.process("Td", operands)
            case "Tm":
                try:
                    values = [float(value) for value in operands[:6]]  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                except TypeError, ValueError:
                    values = []
                self.tm = values if len(values) == 6 else list(IDENTITY_MATRIX)
                self.positioned(self.accumulated_width / 1000.0)
                self.accumulated_width = 0.0
            case "T*":
                self.move_text(0.0, -self.leading)
            case "Tj":
                value = operands[0] if operands else None
                if isinstance(value, PdfString):
                    self.show(bytes(value.data))
                elif isinstance(value, PdfName):
                    self.show(b"")
                else:
                    # pypdf shows nothing for a missing or non-string operand
                    # but still applies its positioning check.
                    self.positioned(0.0)
            case "TJ" if operands and isinstance(operands[0], list):
                threshold = self.half_space_width * 0.95
                for item in operands[0]:
                    match item:
                        case PdfString(data=data):
                            self.show(bytes(data))
                        case bool():
                            pass
                        case int() | float() if (
                            abs(float(item)) >= threshold
                            and self.text
                            and not self.text.endswith(" ")
                        ):
                            self.add_text(" ")
                            self.accumulated_width += (
                                self.font.synthetic_space_width if self.font is not None else 250.0
                            ) * self.font_size
                            self.actual_height = self.font_size
                            self.positioned(0.0)
            case "'":
                self.process("T*", ())
                self.process("Tj", operands)
            case '"' if len(operands) >= 3:
                self.process("T*", ())
                self.process("Tj", operands[2:3])

    def extract(self, streams: tuple[PdfStream, ...] | None = None) -> str:
        content_streams = streams if streams is not None else self.page.content_streams
        data = b"\n".join(stream.data for stream in content_streams)
        inline_image = re.search(rb"(?<!\S)BI(?=\s)", data)
        if (
            inline_image is not None
            and re.search(rb"(?<!\S)EI(?:\s|$)", data[inline_image.end() :]) is None
        ):
            raise ValueError("unexpected end of inline image stream")
        parsed = list(iter_content_operations(PdfLexer(data)))
        before, after = self.options.visitor_operand_before, self.options.visitor_operand_after
        for operator, operands in parsed:
            if before is not None or after is not None:
                visited_operator = operator.encode("latin-1")
                visited_operands = [to_generic(item, self.options.reader) for item in operands]
            if before is not None:
                before(visited_operator, visited_operands, self.cm, self.tm)
            if operator == "Do" and operands:
                self.flush()
                previous = self.output_last
                self.output_last = ensure_line_break(self.output_parts, self.output_last)
                if self.output_last != previous:
                    self.visit("\n")
                self.paint_form_xobject(operands)
                self.text = ""
                self.remember_position()
            else:
                self.process(operator, operands)
            if after is not None:
                after(visited_operator, visited_operands, self.cm, self.tm)
        if self.text:
            self.flush()
        return "".join(self.output_parts)

    def paint_form_xobject(self, operands: tuple[object, ...]) -> None:
        if not operands:
            return
        resources = self.document.resolver.as_dict(self.resources)
        if resources is None:
            return
        xobjects = self.document.resolver.dict_at(resources, "XObject")
        if xobjects is None:
            return
        xobject = self.document.resolver.resolve(xobjects.get(operands[0]))
        if not isinstance(xobject, PdfStream):
            return
        if str(xobject.dictionary.get("Subtype")) == "Image":
            return
        form_id = id(xobject)
        if form_id in self.known_forms:
            logger.warning(
                "Detected cyclic form XObject reference, skipping /%s.",
                recover_pdf_name(operands[0]),
            )
            return
        if self.invocations.count >= MAX_FORM_INVOCATIONS:
            return
        self.invocations.count += 1
        top_level = not self.known_forms
        form_resources = xobject.dictionary.get("Resources")
        if form_resources is None:
            return
        top_level = top_level and self.options.is_plain
        cached = self.caches.forms.get(form_id) if top_level else None
        if cached is not None:
            child_text = cached[1]
        else:
            stream = self.document.resolver.resolve_stream(xobject)
            self.known_forms.add(form_id)
            try:
                child = LegacyTextExtractor(
                    self.page,
                    form_resources,
                    self.known_forms,
                    self.caches,
                    self.invocations,
                    self.options,
                )
                child_text = child.extract((stream,))
            finally:
                self.known_forms.discard(form_id)
            if top_level:
                self.caches.forms[form_id] = (xobject, child_text)
        if child_text:
            self.output_parts.append(child_text)
            self.output_last = child_text[-1]
        self.visit(child_text)


def extract_legacy_text(
    page: Any, caches: LegacyTextCaches | None = None, options: TextOptions | None = None
) -> str:
    return LegacyTextExtractor(page, caches=caches, options=options).extract()


def extract_layout_text(page: Any, caches: LegacyTextCaches, **arguments: Any) -> str:
    fonts = LegacyTextExtractor(page, caches=caches).fonts
    data = b"\n".join(stream.data for stream in page.content_streams)
    return layout_text(iter_content_operations(PdfLexer(data)), fonts, **arguments)
