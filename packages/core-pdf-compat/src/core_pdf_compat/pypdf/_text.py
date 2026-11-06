from __future__ import annotations

import logging
import re
import string
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager, suppress
from typing import Any, Literal, cast

from core_adobe_fonts.agl.glyph_list import GLYPH_DATA
from core_pdf.impl.capture_recovery import ContentFallback, content_scanner
from core_pdf.impl.fonts_cmap import ToUnicodeCMap
from core_pdf.impl.fonts_decoder import FontDecoder
from core_pdf.impl.fonts_metrics import LIGATURE_TEXT_TO_CHAR
from core_pdf.impl.pdf_values import recover_pdf_name
from core_pdf.impl.recovery_lexer import PdfLexer
from core_pdf.impl.types import (
    PdfName,
    PdfReference,
    PdfString,
    RecordType,
    ReplaceFields,
    ReprFields,
)
from core_pdf_compat._text_state import (
    IDENTITY_MATRIX,
    TextMachine,
    ensure_line_break,
    legacy_base_table,
    number_operand,
    orientation,
    predefined_encoding_codec,
    type1_encoding_entries,
)
from core_pdf_compat.pypdf._charsets import SYMBOL_ENCODING, ZAPF_DINGBATS_ENCODING
from core_pdf_compat.pypdf._cmap_scan import scan_to_unicode
from core_pdf_compat.pypdf._configuration import get_configuration
from core_pdf_compat.pypdf._core_metrics import core_font_widths
from core_pdf_compat.pypdf._font_scan import scan_cid_widths, scan_encoding
from core_pdf_compat.pypdf._generic_content import inline_image_spans
from core_pdf_compat.pypdf._glyph_names import extra_glyphs
from core_pdf_compat.pypdf._layout import layout_text
from core_pdf_compat.pypdf.errors import LimitReachedError, PdfReadError
from core_pdf_compat.pypdf.generic import FloatObject, IndirectObject, NumberObject, to_generic
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.operations import ContentOperation, parse_content_token
from core_pdf_spec.s_07_filters.errors import FilterError
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.matrix import multiply_affine
from core_records import GeneratedRecord


def pypdf_data(stream: PdfStream) -> bytes:
    """A stream's decoded data, tolerant where pypdf is.

    core-pdf decodes strictly. A filter error falls back to the facade's filters, which follow
    pypdf: a broken Flate stream, for one, gives what decompresses, with pypdf's warning.
    """
    filters = stream.dictionary.get("Filter")
    first = filters[0] if isinstance(filters, list) and filters else filters
    if isinstance(first, PdfName) and recover_pdf_name(first) in {"LZWDecode", "LZW"}:
        from core_pdf_compat.pypdf.filters import decode_stream_data  # noqa: PLC0415

        # pypdf limits what LZW decompresses to; core-pdf does not.
        return decode_stream_data(to_generic(stream))
    try:
        data = stream.data
    except FilterError:
        from core_pdf_compat.pypdf.filters import decode_stream_data  # noqa: PLC0415

        return decode_stream_data(to_generic(stream))
    if isinstance(first, PdfName) and recover_pdf_name(first) in {"ASCII85Decode", "A85"}:
        from core_pdf_compat.pypdf.filters import warn_missing_ascii85_end  # noqa: PLC0415

        # core-pdf supplies a missing end marker silently; pypdf warns.
        warn_missing_ascii85_end(bytes(stream.raw_data))
    return data


FONT_PROGRAM_KEYS = frozenset({"FontFile", "FontFile2", "FontFile3"})


def without_program(descriptor_holder: object) -> object:
    """A font dictionary whose descriptor carries no font program."""
    if not isinstance(descriptor_holder, dict):
        return descriptor_holder
    descriptor = descriptor_holder.get("FontDescriptor")
    if not isinstance(descriptor, dict):
        return descriptor_holder
    descriptor = {key: value for key, value in descriptor.items() if key not in FONT_PROGRAM_KEYS}
    return {**descriptor_holder, PdfName(b"FontDescriptor"): descriptor}


def without_font_programs(font: dict[object, object]) -> dict[object, object]:
    """A resolved font, and its descendants, whose descriptors carry no font program."""
    stripped = cast(dict[object, object], without_program(font))
    descendants = font.get("DescendantFonts")
    if not isinstance(descendants, (list, tuple)):
        return stripped
    return {
        **stripped,
        PdfName(b"DescendantFonts"): [without_program(item) for item in descendants],
    }


INLINE_IMAGE_START = re.compile(rb"(?<!\S)BI(?=\s)")


def content_operations(data: bytes) -> Iterator[ContentOperation]:
    """core-pdf's operations, read as pypdf's content stream parser reads them.

    An inline image ends where pypdf's parser ends it (only a stream with a BI is searched,
    and an image pypdf cannot read raises its error), and a malformed number is pypdf's.
    """
    return pypdf_numbers(core_operations(data))


def core_operations(data: bytes) -> Iterator[ContentOperation]:
    if INLINE_IMAGE_START.search(data) is None:
        yield from lexed_operations(data)
        return
    position = 0
    for start, end, image in inline_image_spans(data):
        yield from lexed_operations(data[position:start])
        yield "BI", (image,)
        position = end
    yield from lexed_operations(data[position:])


# Delimiters pypdf's parser cannot start an object with, which core's skips.
STRAY_DELIMITERS = frozenset(b")]>}{")


class PypdfContentFallback(ContentFallback):
    """core-pdf's fallback for a token its scanner hands back, raising where pypdf does on a
    delimiter that starts no object, rather than skipping it."""

    __slots__ = ()

    def __call__(self, cursor: int) -> ContentOperation | Literal[False] | None:
        lexer = self.lexer
        start = lexer.skip_ignored_at(cursor)
        if start < lexer.data_len and lexer.raw_data[start] in STRAY_DELIMITERS:
            delimiter = bytes(lexer.raw_data[start : start + 1])
            raise PdfReadError(f"Invalid Elementary Object starting with {delimiter!r}")
        return super().__call__(cursor)


# pypdf reads a number in chunks of 16, then 32, then 64 bytes, and stops at 64.
NUMBER_LENGTH_LIMIT = 64
LONG_NUMBER = re.compile(rb"[+,\-.0-9]{48,}")


def check_number_lengths(data: bytes) -> None:
    """Raise pypdf's LimitReachedError for a number it would read past 64 bytes.

    pypdf has read 48 bytes of a number without its end when it reads the next 64, which
    crosses the limit unless the data ends first. Only data with so long a run of number
    characters is read token by token, to see that the run starts a token.
    """
    if LONG_NUMBER.search(data) is None:
        return
    lexer = PdfLexer(data)
    try:
        while True:
            token = parse_content_token(lexer)
            if token is None:
                return
            start = token.start
            run = LONG_NUMBER.match(data, start)
            if run is not None and len(data) - start >= NUMBER_LENGTH_LIMIT:
                read = 48 + min(64, len(data) - start - 48)
                raise LimitReachedError(
                    f"Read stream length of {read} exceeds maximum allowed length of "
                    f"{NUMBER_LENGTH_LIMIT}."
                )
    except PdfParseError:
        return
    finally:
        lexer.close()


def lexed_operations(data: bytes) -> Iterator[ContentOperation]:
    """iter_content_operations, with PypdfContentFallback and pypdf's number length limit."""
    check_number_lengths(data)
    lexer = PdfLexer(data)
    scanner = content_scanner(lexer)
    step = PypdfContentFallback(lexer, scanner.operands, None, None)
    scan = scanner.next_operation
    while True:
        scanner.pos = lexer.pos
        result = scan()
        if isinstance(result, tuple):
            lexer.pos = scanner.pos
            yield result
            continue
        operation = step(result)
        if operation is None:
            return
        if operation is not False:
            yield operation


NUMBER_START = frozenset("+-.0123456789")
# What pypdf's content stream parser reads an operator or a number from.
OPERATOR_START = frozenset(string.ascii_letters + "'\"") | NUMBER_START
# pypdf reads a number as the characters up to the first outside "+,-." and the digits.
NUMBER_PREFIX = re.compile(r"[+,\-.0-9]*")


def pypdf_number(text: str) -> int | float:
    """pypdf's NumberObject or FloatObject of the text: 0, with its warning, if malformed."""
    value = text.encode("latin-1")
    return float(FloatObject(value)) if b"." in value else int(NumberObject(value))


def pypdf_numbers(operations: Iterator[ContentOperation]) -> Iterator[ContentOperation]:
    """Split the operators core reads from a malformed number, as pypdf's parser does.

    Where pypdf reads a number, and then an operator after it if letters follow, core reads
    one operator: "5Tm", "1.2.3", "7-5". The number joins the operands of what follows.
    """
    pending: list[Any] = []
    for operator, operands in operations:
        if operator and operator[0] not in OPERATOR_START:
            # pypdf reads an object there, and none starts with that character.
            raise PdfReadError(
                f"Invalid Elementary Object starting with {operator[0].encode('latin-1')!r}"
            )
        if operator and operator[0] in NUMBER_START:
            prefix = NUMBER_PREFIX.match(operator)
            length = prefix.end() if prefix is not None else 0
            pending.extend(operands)
            pending.append(pypdf_number(operator[:length]))
            operator = operator[length:]
            if not operator:
                continue
            operands, pending = tuple(pending), []
        elif pending:
            operands, pending = (*pending, *operands), []
        yield operator, operands


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
    # pypdf's layout mode skips the text of a Type3 font it cannot map to characters.
    interpretable: bool = True
    # A simple font's /Encoding name pypdf has no table for, which its layout mode fails
    # to decode with.
    unknown_encoding: str | None = None

    __hash__ = None  # type: ignore[assignment]

    def decode_parts(self, data: bytes) -> tuple[tuple[str, ...], float]:
        glyphs = self.decoder.decode_glyphs(data)
        width = sum(self.glyph_width(glyph) for glyph in glyphs)
        mapped_every_glyph = False
        if not self.decoder.is_cid_font and self.encoding_table is not None:
            # pypdf takes the characters the encoding gives one by one, so a glyph name it
            # does not know ("/afii62766") reads as letters, each with its own direction.
            parts = tuple(part for glyph in glyphs for part in self.simple_glyph_parts(glyph))
            mapped_every_glyph = True
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
                decoded = data.decode("latin-1")
            parts = tuple(decoded)
            # pypdf measures each character it decoded, looked up as a code in /W.
            width = sum(
                self.space_width
                if character == " "
                else float(self.widths.get(ord(character), self.default_width))
                for character in decoded
            )
        else:
            parts = tuple(self.glyph_text(glyph) for glyph in glyphs)
        if not any(parts) and not mapped_every_glyph:
            parts = tuple(data.decode("latin-1"))
        return parts, width

    def layout_word(self, data: bytes) -> tuple[str, str]:
        """The characters pypdf's layout mode decodes the bytes to before its character map,
        which it measures and counts, and the one it takes for a space."""
        if not self.decoder.is_cid_font and self.encoding_table is not None:
            space = chr(self.space_code_bytes[0]) if len(self.space_code_bytes) == 1 else ""
            return "".join(self.encoding_table[code] for code in data), space
        space = cmap_key_character(self.space_code_bytes) if self.space_code_bytes else " "
        try:
            return data.decode(self.encoding_codec or "utf-16-be", "surrogatepass"), space
        except UnicodeDecodeError:
            return data.decode("utf-8", "replace"), space

    def simple_glyph_parts(self, glyph: Any) -> tuple[str, ...]:
        if len(glyph.code_bytes) != 1:
            return (self.glyph_text(glyph),)
        table = self.encoding_table
        if table is None:
            return (self.glyph_text(glyph),)
        encoded = table[glyph.code_bytes[0]]
        return tuple(self.character_map.get(character, character) for character in encoded)

    def glyph_width(self, glyph: Any) -> float:
        if (
            not self.decoder.is_cid_font
            and self.encoding_table is not None
            and len(glyph.code_bytes) == 1
        ):
            # pypdf sums the widths of the characters its encoding gives, each looked up
            # as a code.
            characters = self.encoding_table[glyph.code_bytes[0]]
            space = self.space_code_bytes[0] if len(self.space_code_bytes) == 1 else None
            return sum(
                self.space_width
                if ord(character) == space
                else float(int(self.widths.get(ord(character), self.default_width)))
                for character in characters
            )
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
# The names pypdf keeps encodings under; a font without /Encoding whose /BaseFont is one
# of them is mapped through that table.
PYPDF_CHARSET_NAMES = frozenset(
    {
        "StandardEncoding",
        "WinAnsiEncoding",
        "MacRomanEncoding",
        "PDFDocEncoding",
        "Symbol",
        "ZapfDingbats",
    }
)


def pypdf_charset(name: str) -> list[str]:
    """pypdf's charset_encoding table of that name."""
    if name == "Symbol":
        return list(SYMBOL_ENCODING)
    if name == "ZapfDingbats":
        return list(ZAPF_DINGBATS_ENCODING)
    table = legacy_base_table(name)
    if name == "PDFDocEncoding":
        # pypdf's table leaves these codes undefined, and spells the ligatures as one.
        for code in (22, 127, 159, 173):
            table[code] = "\x00"
        table[147], table[148] = "\ufb01", "\ufb02"
    return table


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


logger = logging.getLogger(__name__)

ALL_ORIENTATIONS = (0, 90, 180, 270)
DEFAULT_SPACE_WIDTH = 200.0


class FormInvocations:
    """How many form XObjects one page extraction has entered, across nesting levels.

    pypdf stops descending into them past Configuration.xform_maximum_invocations_per_extraction,
    and says so once.
    """

    __slots__ = ("count", "has_logged")

    def __init__(self) -> None:
        self.count = 0
        self.has_logged = False


FACADE_LOGGER = logging.getLogger("core_pdf_compat.pypdf")


class LogRecorder(logging.Handler):
    """Keeps the records logged while a font is read, to repeat them when it is reused."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class LegacyTextCaches:
    __slots__ = ("fonts", "forms")

    def __init__(self) -> None:
        self.fonts: dict[int, tuple[object, LegacyFont | None, tuple[logging.LogRecord, ...]]] = {}
        self.forms: dict[int, tuple[PdfStream, str]] = {}


class FontLeftOut(Exception):  # noqa: N818
    """A font pypdf leaves out of the page's fonts, without caching it."""


class TextOptions(GeneratedRecord):
    """The extract_text arguments. ``reader`` turns visited objects into pypdf's."""

    orientations: tuple[int, ...] = ALL_ORIENTATIONS
    visitor_operand_before: Callable[[Any, Any, Any, Any], None] | None = None
    visitor_operand_after: Callable[[Any, Any, Any, Any], None] | None = None
    visitor_text: Callable[[Any, Any, Any, Any, Any], None] | None = None
    reader: Any = None
    layout: bool = False

    @property
    def is_plain(self) -> bool:
        return self == TextOptions(reader=self.reader)


class UnknownFont:
    """pypdf's "Unknown" font, for a Tf naming no font it read: every code shows as U+FFFD."""

    space_width = 250.0
    synthetic_space_width = 250.0
    default_width = 500.0
    space_code_bytes = b" "
    widths: Mapping[int, float] = {}

    def decode_parts(self, data: bytes) -> tuple[tuple[str, ...], float]:
        return ("\ufffd",) * len(data), 500.0 * len(data)


UNKNOWN_FONT = UnknownFont()


class LegacyTextExtractor(TextMachine[LegacyFont | UnknownFont]):
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
                if self.options.layout:
                    # pypdf's layout mode reads every font, failing on what is not one.
                    value = self.document.resolver.resolve(raw_font)
                    name = "NoneType" if value is None else type(to_generic(value)).__name__
                    raise AttributeError(f"'{name}' object has no attribute 'get'")
                continue
            known = font_cache.get(id(font))
            if known is None or known[0] is not font:
                recorder = LogRecorder()
                FACADE_LOGGER.addHandler(recorder)
                try:
                    legacy = self.legacy_font(font)
                except FontLeftOut:
                    continue
                finally:
                    FACADE_LOGGER.removeHandler(recorder)
                known = font_cache[id(font)] = (font, legacy, tuple(recorder.records))
            else:
                # pypdf reads every font afresh for each extraction, warning each time.
                for record in known[2]:
                    logging.getLogger(record.name).handle(record)
            if known[1] is not None:
                fonts[str(resource_name)] = known[1]
        return fonts

    def legacy_font(self, font: dict[object, object]) -> LegacyFont | None:
        resolve = self.document.resolver.resolve
        scan_encoding(font, resolve)
        subtype = recover_pdf_name(font.get("Subtype") or "")
        if subtype not in {"Type1", "MMType1", "TrueType", "Type3"}:
            # pypdf ignores a missing or malformed /DescendantFonts array.
            descendants = self.document.resolver.resolve(font.get("DescendantFonts"))
        try:
            self.read_font_programs(font, subtype)
        except (AttributeError, TypeError) as error:
            # pypdf's plain extraction leaves out a font it fails to read so, and reads it
            # again for the next page; its layout mode fails.
            if self.options.layout:
                raise
            raise FontLeftOut from error
        try:
            resolved_font = self.document.resolver.resolve_font_dict(font)
            try:
                decoder = FontDecoder(resolved_font)
            except FilterError:
                # pypdf does not read a composite font's program, so one that cannot be
                # decoded leaves the font usable, read without its program.
                if subtype in {"Type1", "MMType1", "TrueType", "Type3"}:
                    raise
                decoder = FontDecoder(without_font_programs(resolved_font))
            except TypeError, ValueError:
                raise
            except Exception:  # noqa: BLE001
                # The font program is one fontTools cannot read; pypdf never reads it.
                decoder = FontDecoder(without_font_programs(resolved_font))
        except TypeError, ValueError:
            return None
        cmap: ToUnicodeCMap | None = None
        to_unicode = self.document.resolver.resolve(font.get("ToUnicode"))
        if isinstance(to_unicode, PdfStream):
            data = pypdf_data(self.document.resolver.resolve_stream(to_unicode))
            scan_to_unicode(data)
            try:
                cmap = ToUnicodeCMap(data)
            except ValueError:
                cmap = None
        if subtype not in {"Type1", "MMType1", "TrueType", "Type3"} and isinstance(
            descendants, (list, tuple)
        ):
            reader = self.options.reader
            walked: dict[int, float] = {}
            irregular = False
            for descendant in descendants:
                resolved = resolve(descendant)
                if isinstance(resolved, dict):
                    irregular |= scan_cid_widths(
                        resolved, resolve, lambda value: to_generic(value, reader), walked
                    )
            # A malformed /W is read as pypdf walks it, which core-pdf does differently.
            reference_widths = walked if irregular else None
        else:
            reference_widths = None
        encoding_table, encoding_codec, character_map = self.legacy_encoding(font, decoder)
        if encoding_table is not None and cmap is not None:
            # pypdf's character map is the ToUnicode CMap, keyed by each code read as text,
            # and its encoding maps each code the CMap maps (up to 255) to the code itself.
            table = list(encoding_table)
            for code in cmap.mappings:
                number = int.from_bytes(code, "big")
                if number <= 255:
                    table[number] = chr(number)
            encoding_table = tuple(table)
            character_map = {
                cmap_key_character(code): value for code, value in cmap.mappings.items()
            }
        raw_encoding = self.document.resolver.resolve(font.get("Encoding"))
        width_uses_source_code = decoder.is_cid_font and isinstance(raw_encoding, PdfStream)
        encoding_is_mapping = self.encoding_is_mapping(font)
        space_code = (
            self.resolve_cid_space_code(cmap)
            if decoder.is_cid_font
            else self.resolve_space_code(character_map, encoding_table, encoding_is_mapping)
        )
        if not decoder.is_cid_font and encoding_table is not None:
            reference_widths = self.pypdf_simple_widths(font, list(encoding_table))
        widths, default_width, space_width, fixed_pitch = self.font_widths(
            font, decoder, cmap, space_code, reference_widths
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
            self.interpretable(font, subtype),
            None if decoder.is_cid_font else self.unknown_encoding_name(font),
        )

    def read_font_programs(self, font: dict[object, object], subtype: str | None) -> None:
        """Read the font programs as pypdf's Font.from_font_resource reads them.

        pypdf gets each /FontFile* object of the font's descriptors (the descendants' for a
        composite font), so an object its reader cannot read fails the extraction, and a
        descriptor with two programs fails it with PdfReadError.
        """
        resolve = self.document.resolver.resolve
        if subtype in {"Type1", "MMType1", "TrueType", "Type3"}:
            if not self.interpretable(font, subtype):
                return
            holders: list[object] = [font]
        else:
            descendants = resolve(font.get("DescendantFonts"))
            holders = list(descendants) if isinstance(descendants, (list, tuple)) else []
        for holder in holders:
            resolved = resolve(holder)
            descriptor = (
                resolve(resolved.get("FontDescriptor")) if isinstance(resolved, dict) else None
            )
            if isinstance(descriptor, dict):
                self.read_descriptor_programs(descriptor)

    def read_descriptor_programs(self, descriptor: dict[object, object]) -> None:
        reader = self.options.reader
        found = False
        for key in ("FontFile", "FontFile2", "FontFile3"):
            if key not in descriptor:
                continue
            if found:
                shown = to_generic(descriptor, reader) if reader is not None else descriptor
                raise PdfReadError(f"More than one /FontFile found in {shown}")
            value = descriptor[key]
            if reader is not None and isinstance(value, PdfReference):
                try:
                    reader.get_object(
                        IndirectObject(value.object_number, value.generation_number, reader)
                    )
                except PdfReadError:
                    # pypdf warns and reads the font without its program.
                    continue
            found = True

    def interpretable(self, font: dict[object, object], subtype: str | None) -> bool:
        """pypdf's test: a Type3 font without /ToUnicode needs only glyph names it knows."""
        if subtype != "Type3" or "ToUnicode" in font:
            return True
        char_procs = self.document.resolver.resolve(font.get("CharProcs"))
        if not isinstance(char_procs, dict):
            return True
        return all(
            self.legacy_glyph_name(recover_pdf_name(name) or str(name), unknown="")
            for name in char_procs
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
        character_map: Mapping[str, str],
        encoding_table: tuple[str, ...] | None,
        encoding_is_mapping: bool,
    ) -> int | None:
        # pypdf takes the first code its encoding maps to a space, then the first entry for
        # a space in its character map: the ToUnicode CMap, or else the map read from a
        # Type1 font program.
        if encoding_is_mapping and encoding_table is not None:
            encoded = next(
                (code for code, value in enumerate(encoding_table) if value == " "), None
            )
            if encoded is not None:
                return encoded
        key = next((key for key, value in character_map.items() if value == " "), None)
        if key is not None:
            # Simple fonts compare the key with single-byte codes.
            return ord(key) if len(key) == 1 and ord(key) < 256 else None
        return 32

    def encoding_is_mapping(self, font: dict[object, object]) -> bool:
        """Whether pypdf's encoding of the simple font is a table (else a codec name)."""
        encoding = self.document.resolver.resolve(font.get("Encoding"))
        if encoding is None:
            # Without /Encoding, pypdf maps codes through a table only for a font named
            # after one of its encodings (Symbol, ZapfDingbats); others decode as "charmap".
            return recover_pdf_name(font.get("BaseFont") or "") in PYPDF_CHARSET_NAMES
        if isinstance(encoding, PdfName):
            return recover_pdf_name(encoding) in PYPDF_CHARSET_NAMES
        # pypdf takes a string for a name it has no table for; a dictionary has its base
        # table, and anything else gets StandardEncoding.
        return not isinstance(encoding, PdfString)

    def legacy_encoding(
        self, font: dict[object, object], decoder: FontDecoder
    ) -> tuple[tuple[str, ...] | None, str | None, dict[str, str]]:
        if decoder.is_cid_font:
            encoding_name = recover_pdf_name(font.get("Encoding") or "") or ""
            return None, predefined_encoding_codec(encoding_name), {}
        table = self.base_table(font)
        character_map: dict[str, str] = {}

        for code, name in decoder.encoding.differences.items():
            if 0 <= code <= 255:
                table[code] = self.legacy_glyph_name(name)

        subtype = recover_pdf_name(font.get("Subtype") or "")
        descriptor = self.document.resolver.resolve(font.get("FontDescriptor"))
        if subtype == "Type1" and font.get("ToUnicode") is None and isinstance(descriptor, dict):
            character_map.update(self.font_file_character_map(descriptor))
            for character in character_map:
                table[ord(character)] = character
        return tuple(table), None, character_map

    def unknown_encoding_name(self, font: dict[object, object]) -> str | None:
        encoding = self.document.resolver.resolve(font.get("Encoding"))
        if not isinstance(encoding, PdfName):
            return None
        name = recover_pdf_name(encoding) or ""
        if name in PYPDF_CHARSET_NAMES or predefined_encoding_codec(name) is not None:
            return None
        return f"/{name}"

    def base_table(self, font: dict[object, object]) -> list[str]:
        """The table pypdf's _parse_encoding starts a simple font from, before /Differences."""
        resolve = self.document.resolver.resolve
        encoding = resolve(font.get("Encoding"))
        if encoding is None:
            # Without /Encoding, a font named after one of pypdf's encodings uses it, and any
            # other decodes as "charmap": each code is itself.
            base_font = recover_pdf_name(font.get("BaseFont") or "")
            if base_font in PYPDF_CHARSET_NAMES:
                return pypdf_charset(str(base_font))
            return [chr(code) for code in range(256)]
        if isinstance(encoding, PdfString):
            return [chr(code) for code in range(256)]  # pypdf reads it as a name it lacks.
        if isinstance(encoding, PdfName):
            name = recover_pdf_name(encoding)
            if name in PYPDF_CHARSET_NAMES:
                return pypdf_charset(str(name))
            # A name pypdf has no table for decodes as "charmap".
            return [chr(code) for code in range(256)]
        base = resolve(encoding.get("BaseEncoding")) if isinstance(encoding, dict) else None
        name = recover_pdf_name(base) if isinstance(base, PdfName) else None
        return pypdf_charset(str(name) if name in PYPDF_CHARSET_NAMES else "StandardEncoding")

    def font_file_character_map(self, descriptor: dict[object, object]) -> dict[str, str]:
        # pypdf reads the built-in encoding of a /FontFile (Type 1) program only.
        font_file = self.document.resolver.resolve(descriptor.get("FontFile"))
        if isinstance(font_file, PdfStream):
            return self.type1_character_map(font_file)
        return {}

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
        # pypdf's adobe_glyphs: the Adobe Glyph List and its additions.
        if name == ".notdef":
            return "□"
        if name.startswith("a") and name[1:].isdecimal():
            code = int(name[1:])
            if 0 <= code <= 255:
                return chr(code)
        mapped = extra_glyphs().get(name)
        if mapped is None:
            mapped = GLYPH_DATA.get(name)
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
        reference_widths: Mapping[int, float] | None = None,
    ) -> tuple[Mapping[int, float], float, float, bool]:
        widths = decoder.metrics.widths if reference_widths is None else reference_widths
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
        if decoder.is_cid_font:
            fixed_pitch = self.cid_fixed_pitch(font)
        width = widths.get(space_code) if space_code is not None else None
        legacy_width = width if decoder.is_cid_font else int(width or 0)
        if legacy_width:
            return widths, default_width, float(legacy_width), fixed_pitch
        return widths, default_width, DEFAULT_SPACE_WIDTH, fixed_pitch

    def pypdf_simple_widths(self, font: dict[object, object], table: list[str]) -> dict[int, float]:
        """A simple font's widths as pypdf reads them: its /Widths, else those of the core font
        it names, by the characters its encoding gives; keyed by character code.
        """
        if font.get("Widths") is not None:
            return self.raw_simple_widths(font)
        core = core_font_widths(recover_pdf_name(font.get("BaseFont") or "") or "")
        if core is None:
            return {}
        return {
            code: float(core[character])
            for code, character in enumerate(table)
            if character in core
        }

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
            width = resolve(raw_width)
            if isinstance(width, (int, float)) and not isinstance(width, bool):
                widths[code] = float(int(width))
        return widths

    def cid_fixed_pitch(self, font: dict[object, object]) -> bool:
        """pypdf's fixed pitch flag of a composite font: its last descendant's descriptor's."""
        resolve = self.document.resolver.resolve
        descendants = resolve(font.get("DescendantFonts"))
        flags: object = 32
        for descendant in descendants if isinstance(descendants, (list, tuple)) else ():
            resolved = resolve(descendant)
            descriptor = (
                resolve(resolved.get("FontDescriptor")) if isinstance(resolved, dict) else None
            )
            flags = resolve(descriptor.get("Flags", 32)) if isinstance(descriptor, dict) else 32
        return isinstance(flags, int) and bool(flags & 1)

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
        if not text or not self.text:
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

    def show_name(self, name: PdfName) -> None:
        """pypdf shows a name operand as its own text, slash included, without the font."""
        if not self.visible():
            self.actual_height = self.font_size
            self.positioned(0.0)
            return
        text = "/" + (recover_pdf_name(name) or "")
        width = 0.0
        if self.font is None:
            width = len(text) * 500.0
        else:
            font = self.font
            space = font.space_code_bytes.decode("latin-1")
            width = sum(
                font.space_width
                if character == space
                else float(int(font.widths.get(ord(character), font.default_width)))
                for character in text
            )
        self.text += text
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
                font = self.fonts.get(str(operands[0])) if operands else None
                # pypdf shows the text of a font it cannot find with its "Unknown" font.
                self.font = UNKNOWN_FONT if font is None else font
                self.font_name = str(operands[0]) if font is not None else None
                self.half_space_width = self.font.space_width / 2.0
                with suppress(IndexError, TypeError, ValueError):
                    self.font_size = float(operands[1])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            case "Td":
                self.move_text(number_operand(operands, 0, 0.0), number_operand(operands, 1, 0.0))
            case "TD" if len(operands) >= 2:
                # A non-numeric ty leaves the leading as it is.
                ty = operands[1]
                if isinstance(ty, (int, float)) and not isinstance(ty, bool):
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
                    self.show_name(value)
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
                            # pypdf shows a str " ", which it appends whatever the direction.
                            if self.visible():
                                self.text += " "
                                self.accumulated_width += (
                                    self.font.synthetic_space_width
                                    if self.font is not None
                                    else 250.0
                                ) * self.font_size
                            self.actual_height = self.font_size
                            self.positioned(0.0)
            case "'":
                self.process("T*", ())
                self.process("Tj", operands)
            case '"' if len(operands) >= 3:
                self.process("T*", ())
                self.process("Tj", operands[2:3])

    def extract(
        self, streams: tuple[PdfStream, ...] | None = None, data: bytes | None = None
    ) -> str:
        if data is None:
            content_streams = streams if streams is not None else self.page.content_streams
            data = b"\n".join(pypdf_data(stream) for stream in content_streams)
        parsed = list(content_operations(data))
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
        limit = get_configuration().xform_maximum_invocations_per_extraction
        if self.invocations.count >= limit:
            if not self.invocations.has_logged:
                self.invocations.has_logged = True
                logger.warning(
                    "Exceeded %d form XObject invocations while extracting text; "
                    "further form content is skipped.",
                    limit,
                )
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
        # The visitor saw each piece of the form's text while it was extracted.


VENDORED_FONTTOOLS_LOGGER = logging.getLogger("core_pdf._vendor.fontTools")
VENDORED_FONTTOOLS_LOGGER.addHandler(logging.NullHandler())


@contextmanager
def font_programs_quiet() -> Generator[None]:
    """Keep the messages of core-pdf's bundled fontTools to itself in the block.

    core-pdf reads font programs to extract text, where pypdf does not read them at all, so
    what fontTools says about one is nothing pypdf would log.
    """
    propagate = VENDORED_FONTTOOLS_LOGGER.propagate
    VENDORED_FONTTOOLS_LOGGER.propagate = False
    try:
        yield
    finally:
        VENDORED_FONTTOOLS_LOGGER.propagate = propagate


@contextmanager
def irregular_dictionaries_reported(reader: Any) -> Generator[None]:
    """Read as pypdf the objects with an irregular dictionary core first parsed in the block.

    pypdf reads the objects it extracts text from through its parser, which warns about a
    repeated key or a broken entry; core-pdf's text extraction reads them itself, so they
    are read again here.
    """
    resolver = getattr(getattr(reader, "_pdf", None), "resolver", None)
    noted = getattr(resolver, "irregular_dictionaries", None)
    before = frozenset(noted) if noted else frozenset()
    yield
    if not noted or len(noted) == len(before):
        return
    from core_pdf_compat.pypdf.generic import IndirectObject  # noqa: PLC0415

    for number, generation in sorted(noted - before):
        if (generation, number) not in reader.resolved_objects:
            with suppress(Exception):
                reader.get_object(IndirectObject(number, generation, reader))


def irregular_contents(page: Any, reader: Any, source: Any) -> bool:
    """Whether core noted an irregular dictionary among the page's content streams.

    pypdf reads those with its parser, which can find something other than a stream there,
    so their data is then taken as pypdf's page reads it.
    """
    resolver = getattr(getattr(reader, "_pdf", None), "resolver", None)
    noted = getattr(resolver, "irregular_dictionaries", None)
    if noted is None:
        return False
    core_streams = source.content_streams  # Core reads them, and notes what is irregular.
    contents = page.get("/Contents")
    if contents is not None and not core_streams:
        # Core could not read a content stream that pypdf's reader repairs.
        resolved = contents.get_object()
        if isinstance(resolved, list) or hasattr(resolved, "get_data"):
            return True
    if not noted:
        return False
    references = list(contents) if isinstance(contents, list) else [contents]
    if hasattr(contents, "idnum"):
        resolved = contents.get_object()
        if isinstance(resolved, list):
            references.extend(resolved)
    return any(
        (reference.idnum, reference.generation) in noted
        for reference in references
        if hasattr(reference, "idnum")
    )


def extract_legacy_text(
    page: Any,
    caches: LegacyTextCaches | None = None,
    options: TextOptions | None = None,
    data: bytes | None = None,
) -> str:
    return LegacyTextExtractor(page, caches=caches, options=options).extract(data=data)


def extract_form_text(
    page: Any, form: PdfStream, caches: LegacyTextCaches, options: TextOptions
) -> str:
    resources = form.dictionary.get("Resources")
    extractor = LegacyTextExtractor(page, resources, caches=caches, options=options)
    return extractor.extract((page.document.resolver.resolve_stream(form),))


def extract_layout_text(
    page: Any,
    caches: LegacyTextCaches,
    data: bytes | None = None,
    reader: Any = None,
    **arguments: Any,
) -> str:
    options = TextOptions(reader=reader, layout=True)
    fonts = LegacyTextExtractor(page, caches=caches, options=options).fonts
    if data is None:
        data = b"\n".join(pypdf_data(stream) for stream in page.content_streams)
    return layout_text(content_operations(data), fonts, **arguments)
