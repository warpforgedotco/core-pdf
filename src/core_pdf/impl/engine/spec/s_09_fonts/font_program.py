"""CFF font-program parsing and glyph geometry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from math import inf
from threading import RLock

from core_pdf.impl.engine.spec.s_09_fonts.feature_distance_kernel import (
    feature_distance as compiled_feature_distance,
)
from core_pdf.impl.engine.spec.s_09_fonts.feature_distance_kernel import (
    feature_distance_matrix as compiled_feature_distance_matrix,
)
from core_pdf.impl.engine.spec.s_09_fonts.raster_kernel import rasterize_contours


@dataclass(frozen=True)
class CFFGlyphFeature:
    cells: tuple[tuple[int, int], ...]
    aspect: float
    contours: int
    bitmap: tuple[int, ...] = ()


EMPTY_FEATURE = CFFGlyphFeature((), 0.0, 0, ())
FEATURE_GRID_WIDTH = 18
FEATURE_GRID_HEIGHT = 24
STANDARD_GLYPH_SIDS = {
    name: sid
    for sid, name in enumerate(
        (
            ".notdef",
            "space",
            "exclam",
            "quotedbl",
            "numbersign",
            "dollar",
            "percent",
            "ampersand",
            "quoteright",
            "parenleft",
            "parenright",
            "asterisk",
            "plus",
            "comma",
            "hyphen",
            "period",
            "slash",
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "colon",
            "semicolon",
            "less",
            "equal",
            "greater",
            "question",
            "at",
            *tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            "bracketleft",
            "backslash",
            "bracketright",
            "asciicircum",
            "underscore",
            "quoteleft",
            *tuple("abcdefghijklmnopqrstuvwxyz"),
            "braceleft",
            "bar",
            "braceright",
            "asciitilde",
        )
    )
}
CFF_STANDARD_STRING_COUNT = 391
# Type 2 charstrings may call subroutines, which may call further subroutines. The spec
# allows 10 levels; deeper than that means a malformed or maliciously recursive font.
TYPE2_MAX_SUBR_DEPTH = 10
internal_STANDARD_GLYPH_NAMES_AFTER_ASCII = """
exclamdown cent sterling fraction yen florin section currency quotesingle quotedblleft
guillemotleft guilsinglleft guilsinglright fi fl endash dagger daggerdbl periodcentered
paragraph bullet quotesinglbase quotedblbase quotedblright guillemotright ellipsis perthousand
questiondown grave acute circumflex tilde macron breve dotaccent dieresis ring cedilla
hungarumlaut ogonek caron emdash AE ordfeminine Lslash Oslash OE ordmasculine ae dotlessi
lslash oslash oe germandbls onesuperior logicalnot mu trademark Eth onehalf plusminus Thorn
onequarter divide brokenbar degree thorn threequarters twosuperior registered minus eth
multiply threesuperior copyright Aacute Acircumflex Adieresis Agrave Aring Atilde Ccedilla
Eacute Ecircumflex Edieresis Egrave Iacute Icircumflex Idieresis Igrave Ntilde Oacute
Ocircumflex Odieresis Ograve Otilde Scaron Uacute Ucircumflex Udieresis Ugrave Yacute
Ydieresis Zcaron aacute acircumflex adieresis agrave aring atilde ccedilla eacute
ecircumflex edieresis egrave iacute icircumflex idieresis igrave ntilde oacute ocircumflex
odieresis ograve otilde scaron uacute ucircumflex udieresis ugrave yacute ydieresis zcaron
exclamsmall Hungarumlautsmall dollaroldstyle dollarsuperior ampersandsmall Acutesmall
parenleftsuperior parenrightsuperior twodotenleader onedotenleader zerooldstyle oneoldstyle
twooldstyle threeoldstyle fouroldstyle fiveoldstyle sixoldstyle sevenoldstyle eightoldstyle
nineoldstyle commasuperior threequartersemdash periodsuperior questionsmall asuperior bsuperior
centsuperior dsuperior esuperior isuperior lsuperior msuperior nsuperior osuperior rsuperior
ssuperior tsuperior ff ffi ffl parenleftinferior parenrightinferior Circumflexsmall
hyphensuperior Gravesmall Asmall Bsmall Csmall Dsmall Esmall Fsmall Gsmall Hsmall Ismall Jsmall
Ksmall Lsmall Msmall Nsmall Osmall Psmall Qsmall Rsmall Ssmall Tsmall Usmall Vsmall Wsmall
Xsmall Ysmall Zsmall colonmonetary onefitted rupiah Tildesmall exclamdownsmall centoldstyle
Lslashsmall Scaronsmall Zcaronsmall Dieresissmall Brevesmall Caronsmall Dotaccentsmall
Macronsmall figuredash hypheninferior Ogoneksmall Ringsmall Cedillasmall questiondownsmall
oneeighth threeeighths fiveeighths seveneighths onethird twothirds zerosuperior foursuperior
fivesuperior sixsuperior sevensuperior eightsuperior ninesuperior zeroinferior oneinferior
twoinferior threeinferior fourinferior fiveinferior sixinferior seveninferior eightinferior
nineinferior centinferior dollarinferior periodinferior commainferior Agravesmall Aacutesmall
Acircumflexsmall Atildesmall Adieresissmall Aringsmall AEsmall Ccedillasmall Egravesmall
Eacutesmall Ecircumflexsmall Edieresissmall Igravesmall Iacutesmall Icircumflexsmall
Idieresissmall Ethsmall Ntildesmall Ogravesmall Oacutesmall Ocircumflexsmall Otildesmall
Odieresissmall OEsmall Oslashsmall Ugravesmall Uacutesmall Ucircumflexsmall Udieresissmall
Yacutesmall Thornsmall Ydieresissmall 001.000 001.001 001.002 001.003 Black Bold Book Light
Medium Regular Roman Semibold
""".split()  # noqa: SIM905 - compact ordered copy of the CFF specification table
STANDARD_GLYPH_SIDS.update(
    {name: sid for sid, name in enumerate(internal_STANDARD_GLYPH_NAMES_AFTER_ASCII, start=96)}
)
assert len(STANDARD_GLYPH_SIDS) == CFF_STANDARD_STRING_COUNT


class CFFFont:
    __slots__ = (
        "data",
        "top_dict",
        "charstrings",
        "cid_to_gid",
        "custom_string_sids",
        "is_cid_keyed",
        "global_subrs",
        "local_subrs",
        "fd_select",
        "internal_glyph_geometry_cache",
    )

    def __init__(self, data: bytes | memoryview | None) -> None:
        if data is None:
            self.data = b""
            self.top_dict = {}
            self.charstrings: list[bytes] = []
            self.cid_to_gid = {}
            self.custom_string_sids = {}
            self.is_cid_keyed = False
            self.global_subrs: tuple[bytes, ...] = ()
            self.local_subrs: tuple[tuple[bytes, ...], ...] = ()
            self.fd_select: tuple[int, ...] = ()
            self.internal_glyph_geometry_cache: dict[
                int,
                tuple[
                    tuple[tuple[tuple[float, float], ...], ...],
                    tuple[float, float, float, float] | None,
                ],
            ] = {}
            return
        # Keep a caller-owned read-only view when one is provided.  INDEX
        # entries are still materialized as bytes below because they escape
        # the parser and are used as stable cache keys.
        self.data = data
        if len(self.data) < 4 or self.data[0] != 1:
            raise ValueError("invalid CFF font program")
        pos = self.data[2]
        ignored_names, pos = self.internal_read_index(pos)
        top_index, pos = self.internal_read_index(pos)
        custom_strings, pos = self.internal_read_index(pos)
        self.custom_string_sids = {
            value.decode("latin-1"): CFF_STANDARD_STRING_COUNT + index
            for index, value in enumerate(custom_strings)
        }
        global_subrs, pos = self.internal_read_index(pos)
        self.global_subrs = tuple(global_subrs)
        if not top_index:
            raise ValueError("invalid CFF top dict")
        self.top_dict = self.internal_parse_dict(top_index[0])
        self.is_cid_keyed = (12, 30) in self.top_dict
        charstrings_off = self.top_dict.get(17, [None])[0]
        if not isinstance(charstrings_off, (int, float)):
            raise ValueError("invalid CFF CharStrings offset")
        self.charstrings, ignored_pos = self.internal_read_index(int(charstrings_off))
        charset_off = self.top_dict.get(15, [0])[0]
        self.cid_to_gid = self.internal_read_charset(
            int(charset_off) if isinstance(charset_off, (int, float)) else 0,
            len(self.charstrings),
        )
        self.fd_select = self.internal_read_fd_select()
        self.local_subrs = self.internal_read_local_subrs()
        self.internal_glyph_geometry_cache = {}

    def internal_read_index(self, pos: int) -> tuple[list[bytes], int]:
        data = memoryview(self.data)
        if pos + 2 > len(data):
            raise ValueError("invalid CFF INDEX")
        count = int.from_bytes(data[pos : pos + 2], "big")
        pos += 2
        if count == 0:
            return ([], pos)
        if pos >= len(data):
            raise ValueError("invalid CFF INDEX")
        off_size = data[pos]
        pos += 1
        if off_size < 1 or off_size > 4:
            raise ValueError("invalid CFF INDEX")
        offsets_end = pos + (count + 1) * off_size
        if offsets_end > len(data):
            raise ValueError("invalid CFF INDEX")
        offsets = [
            int.from_bytes(data[pos + i * off_size : pos + (i + 1) * off_size], "big")
            for i in range(count + 1)
        ]
        pos = offsets_end
        if offsets[0] != 1 or any(b < a for a, b in zip(offsets, offsets[1:])):
            raise ValueError("invalid CFF INDEX")
        base = pos
        end = base + offsets[-1] - 1
        if end > len(data):
            raise ValueError("invalid CFF INDEX")
        return (
            [bytes(data[base + offsets[i] - 1 : base + offsets[i + 1] - 1]) for i in range(count)],
            end,
        )

    @staticmethod
    def internal_parse_number(
        item: bytes, pos: int, *, dict_number: bool = False
    ) -> tuple[float, int]:
        b0 = item[pos]
        if 32 <= b0 <= 246:
            return (float(b0 - 139), pos + 1)
        if 247 <= b0 <= 250:
            if pos + 1 >= len(item):
                raise ValueError("invalid CFF number")
            return (float((b0 - 247) * 256 + item[pos + 1] + 108), pos + 2)
        if 251 <= b0 <= 254:
            if pos + 1 >= len(item):
                raise ValueError("invalid CFF number")
            return (float(-(b0 - 251) * 256 - item[pos + 1] - 108), pos + 2)
        if b0 == 28:
            if pos + 3 > len(item):
                raise ValueError("invalid CFF number")
            return (
                float(int.from_bytes(item[pos + 1 : pos + 3], "big", signed=True)),
                pos + 3,
            )
        if b0 == 29 and dict_number:
            if pos + 5 > len(item):
                raise ValueError("invalid CFF number")
            return (
                float(int.from_bytes(item[pos + 1 : pos + 5], "big", signed=True)),
                pos + 5,
            )
        if b0 == 30 and dict_number:
            return CFFFont.internal_parse_real_number(item, pos + 1)
        if b0 == 255 and not dict_number:
            if pos + 5 > len(item):
                raise ValueError("invalid Type 2 number")
            return (
                int.from_bytes(item[pos + 1 : pos + 5], "big", signed=True) / 65536.0,
                pos + 5,
            )
        raise ValueError("invalid CFF number")

    @staticmethod
    def internal_parse_real_number(item: bytes, pos: int) -> tuple[float, int]:
        parts: list[str] = []
        while pos < len(item):
            byte = item[pos]
            pos += 1
            for nibble in (byte >> 4, byte & 15):
                if nibble == 15:
                    text = "".join(parts) or "0"
                    return (float(text), pos)
                if nibble <= 9:
                    parts.append(str(nibble))
                elif nibble == 10:
                    parts.append(".")
                elif nibble == 11:
                    parts.append("e")
                elif nibble == 12:
                    parts.append("e-")
                elif nibble == 14:
                    parts.append("-")
        raise ValueError("invalid CFF real number")

    def internal_parse_dict(self, item: bytes) -> dict[int | tuple[int, int], list[float]]:
        result: dict[int | tuple[int, int], list[float]] = {}
        stack: list[float] = []
        pos = 0
        while pos < len(item):
            byte = item[pos]
            if byte <= 21:
                if byte == 12:
                    pos += 1
                    if pos >= len(item):
                        raise ValueError("invalid CFF dict operator")
                    op: int | tuple[int, int] = (12, item[pos])
                else:
                    op = byte
                result[op] = stack
                stack = []
                pos += 1
            else:
                value, pos = self.internal_parse_number(item, pos, dict_number=True)
                stack.append(value)
        return result

    def internal_read_charset(self, pos: int, glyph_count: int) -> dict[int, int]:
        if glyph_count <= 0:
            return {}
        if glyph_count == 1 or pos in {0, 1, 2}:
            return {gid: gid for gid in range(glyph_count)}
        data = self.data
        if pos >= len(data):
            return {gid: gid for gid in range(glyph_count)}
        fmt = data[pos]
        pos += 1
        cid_to_gid = {0: 0}
        gid = 1
        try:
            if fmt == 0:
                while gid < glyph_count:
                    if pos + 2 > len(data):
                        break
                    cid = int.from_bytes(data[pos : pos + 2], "big")
                    pos += 2
                    cid_to_gid.setdefault(cid, gid)
                    gid += 1
            elif fmt in {1, 2}:
                while gid < glyph_count:
                    if pos + 2 > len(data):
                        break
                    first = int.from_bytes(data[pos : pos + 2], "big")
                    pos += 2
                    if fmt == 1:
                        if pos >= len(data):
                            break
                        left = data[pos]
                        pos += 1
                    else:
                        if pos + 2 > len(data):
                            break
                        left = int.from_bytes(data[pos : pos + 2], "big")
                        pos += 2
                    for offset in range(left + 1):
                        if gid >= glyph_count:
                            break
                        cid_to_gid.setdefault(first + offset, gid)
                        gid += 1
            else:
                return {gid: gid for gid in range(glyph_count)}
        except IndexError:
            pass
        return cid_to_gid

    def internal_read_encoding_codes(self, pos: int) -> dict[int, int]:
        """Read a custom CFF Encoding into a code -> glyph id map.

        Section 12 of the CFF specification defines two layouts, both assigning
        codes to glyph ids in order from glyph 1 (glyph 0 is .notdef and is
        always unencoded). Setting the high bit of the format byte appends
        supplements, which give a second code to an already encoded glyph.
        """
        data = self.data
        if pos <= 0 or pos >= len(data):
            return {}
        raw_format = data[pos]
        fmt = raw_format & 0x7F
        pos += 1
        glyph_count = len(self.charstrings)
        codes: dict[int, int] = {}
        if fmt == 0:
            if pos >= len(data):
                return {}
            n_codes = data[pos]
            pos += 1
            for index in range(n_codes):
                if pos >= len(data):
                    return codes
                gid = index + 1
                if gid < glyph_count:
                    codes.setdefault(data[pos], gid)
                pos += 1
        elif fmt == 1:
            if pos >= len(data):
                return {}
            n_ranges = data[pos]
            pos += 1
            gid = 1
            for _ in range(n_ranges):
                if pos + 2 > len(data):
                    return codes
                first = data[pos]
                n_left = data[pos + 1]
                pos += 2
                for offset in range(n_left + 1):
                    code = first + offset
                    if code > 255:
                        break
                    if gid < glyph_count:
                        codes.setdefault(code, gid)
                    gid += 1
        else:
            return {}

        if raw_format & 0x80:
            if pos >= len(data):
                return codes
            n_sups = data[pos]
            pos += 1
            for _ in range(n_sups):
                if pos + 3 > len(data):
                    break
                code = data[pos]
                sid = int.from_bytes(data[pos + 1 : pos + 3], "big")
                pos += 3
                # Supplements are code to SID, so route them through the
                # charset rather than treating the value as a glyph id.
                supplement_gid = self.cid_to_gid.get(sid)
                if supplement_gid is not None and supplement_gid < glyph_count:
                    codes[code] = supplement_gid
        return codes

    def builtin_encoding(self) -> dict[int, str]:
        """Return the font program's own code -> glyph name encoding.

        9.6.6.1 makes this the encoding in force when the PDF font dictionary
        supplies none. An empty result means the font uses one of the
        predefined encodings, which the caller already applies by name.
        """
        if self.is_cid_keyed:
            # A CIDFont specifies no encoding (CFF specification, section 12).
            return {}
        operand = self.top_dict.get(16, [0])[0]
        if not isinstance(operand, (int, float)):
            return {}
        offset = int(operand)
        if offset in (0, 1):
            # Predefined: 0 is Standard and 1 is Expert.
            return {}
        sid_to_name = {sid: name for name, sid in STANDARD_GLYPH_SIDS.items()}
        sid_to_name.update({sid: name for name, sid in self.custom_string_sids.items()})
        gid_to_name = {
            gid: sid_to_name[sid] for sid, gid in self.cid_to_gid.items() if sid in sid_to_name
        }
        encoding: dict[int, str] = {}
        for code, gid in self.internal_read_encoding_codes(offset).items():
            name = gid_to_name.get(gid)
            if name is not None and name != ".notdef":
                encoding[code] = name
        return encoding

    def glyph_id_for_cid(self, cid: int) -> int:
        if self.is_cid_keyed:
            return self.cid_to_gid.get(cid, 0)
        return cid

    def glyph_id_for_name(self, name: str) -> int:
        sid = STANDARD_GLYPH_SIDS.get(name)
        if sid is None:
            sid = self.custom_string_sids.get(name)
        if sid is None:
            return 0
        return self.cid_to_gid.get(sid, 0)

    def has_glyph_id(self, gid: int) -> bool:
        return 0 <= gid < len(self.charstrings)

    def internal_read_fd_select(self) -> tuple[int, ...]:
        glyph_count = len(self.charstrings)
        fdselect_off = self.top_dict.get((12, 37), [None])[0]
        if not isinstance(fdselect_off, (int, float)):
            return (0,) * glyph_count
        pos = int(fdselect_off)
        data = self.data
        if pos >= len(data):
            return (0,) * glyph_count
        fmt = data[pos]
        pos += 1
        fd_select = [0] * glyph_count
        try:
            if fmt == 0:
                if pos + glyph_count <= len(data):
                    return tuple(data[pos : pos + glyph_count])
            elif fmt == 3:
                if pos + 2 > len(data):
                    return tuple(fd_select)
                range_count = int.from_bytes(data[pos : pos + 2], "big")
                pos += 2
                ranges: list[tuple[int, int]] = []
                for ignored in range(range_count):
                    if pos + 3 > len(data):
                        return tuple(fd_select)
                    first = int.from_bytes(data[pos : pos + 2], "big")
                    fd = data[pos + 2]
                    pos += 3
                    ranges.append((first, fd))
                if pos + 2 > len(data):
                    return tuple(fd_select)
                sentinel = int.from_bytes(data[pos : pos + 2], "big")
                for idx, (first, fd) in enumerate(ranges):
                    end = ranges[idx + 1][0] if idx + 1 < len(ranges) else sentinel
                    for gid in range(max(0, first), min(glyph_count, end)):
                        fd_select[gid] = fd
        except IndexError:
            pass
        return tuple(fd_select)

    def internal_read_local_subrs(self) -> tuple[tuple[bytes, ...], ...]:
        fdarray_off = self.top_dict.get((12, 36), [None])[0]
        if isinstance(fdarray_off, (int, float)):
            try:
                fd_dicts_raw, ignored_pos = self.internal_read_index(int(fdarray_off))
            except ValueError:
                fd_dicts_raw = []
            return tuple(
                tuple(self.internal_read_private_subrs(self.internal_parse_dict(fd_dict_raw)))
                for fd_dict_raw in fd_dicts_raw
            )
        return (tuple(self.internal_read_private_subrs(self.top_dict)),)

    def internal_read_private_subrs(
        self, font_dict: dict[int | tuple[int, int], list[float]]
    ) -> list[bytes]:
        private = font_dict.get(18)
        if not isinstance(private, list) or len(private) < 2:
            return []
        size, offset = private[:2]
        if not isinstance(size, (int, float)) or not isinstance(offset, (int, float)):
            return []
        private_off = int(offset)
        private_size = int(size)
        if private_off < 0 or private_size <= 0 or private_off + private_size > len(self.data):
            return []
        private_dict = self.internal_parse_dict(
            bytes(self.data[private_off : private_off + private_size])
        )
        subrs_off = private_dict.get(19, [None])[0]
        if not isinstance(subrs_off, (int, float)):
            return []
        try:
            subrs, ignored_pos = self.internal_read_index(private_off + int(subrs_off))
        except ValueError:
            return []
        return subrs

    def local_subrs_for_glyph(self, glyph_id: int) -> tuple[bytes, ...]:
        fd_index = self.fd_select[glyph_id] if glyph_id < len(self.fd_select) else 0
        if 0 <= fd_index < len(self.local_subrs):
            return self.local_subrs[fd_index]
        return ()

    def internal_glyph_geometry_for_gid(
        self, glyph_id: int
    ) -> tuple[
        tuple[tuple[tuple[float, float], ...], ...],
        tuple[float, float, float, float] | None,
    ]:
        cached = self.internal_glyph_geometry_cache.get(glyph_id)
        if cached is not None:
            return cached
        geometry: tuple[
            tuple[tuple[tuple[float, float], ...], ...],
            tuple[float, float, float, float] | None,
        ]
        try:
            charstring = self.charstrings[glyph_id]
        except IndexError:
            geometry = ((), None)
        else:
            contours, bbox = internal_type2_glyph_geometry(
                charstring,
                local_subrs=self.local_subrs_for_glyph(glyph_id),
                global_subrs=self.global_subrs,
                collect_contours=True,
            )
            geometry = (tuple(tuple(contour) for contour in contours), bbox)
        if len(self.internal_glyph_geometry_cache) >= 512:
            self.internal_glyph_geometry_cache.clear()
        self.internal_glyph_geometry_cache[glyph_id] = geometry
        return geometry

    def glyph_feature(self, glyph_id: int) -> CFFGlyphFeature:
        geometry = self.internal_glyph_geometry_for_gid(glyph_id)
        contours = geometry[0]
        if not contours:
            return EMPTY_FEATURE
        return internal_feature_from_contours(contours)

    def glyph_feature_for_cid(self, cid: int) -> CFFGlyphFeature:
        return self.glyph_feature(self.glyph_id_for_cid(cid))

    def glyph_bitmap(self, cid: int, width: int = 24, height: int = 32) -> tuple[int, ...]:
        return self.glyph_bitmap_for_gid(
            self.glyph_id_for_cid(cid),
            width=width,
            height=height,
        )

    def glyph_bitmap_for_gid(
        self, glyph_id: int, width: int = 24, height: int = 32
    ) -> tuple[int, ...]:
        geometry = self.internal_glyph_geometry_for_gid(glyph_id)
        contours = geometry[0]
        if not contours:
            return ()
        return rasterize_contours(contours, width=width, height=height)

    def glyph_bbox(self, cid: int) -> tuple[float, float, float, float] | None:
        return self.glyph_bbox_for_gid(self.glyph_id_for_cid(cid))

    def glyph_bbox_for_gid(self, glyph_id: int) -> tuple[float, float, float, float] | None:
        geometry = self.internal_glyph_geometry_for_gid(glyph_id)
        return geometry[1]


def internal_feature_from_contours(
    contours: tuple[tuple[tuple[float, float], ...], ...] | list[list[tuple[float, float]]],
) -> CFFGlyphFeature:
    if not contours:
        return EMPTY_FEATURE

    points = [point for contour in contours for point in contour]
    if not points:
        return EMPTY_FEATURE
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    cells = {
        (
            max(0, min(17, round((px - min_x) / width * 17))),
            max(0, min(23, round((py - min_y) / height * 23))),
        )
        for px, py in points
    }
    bitmap = rasterize_contours(contours, width=18, height=24)
    return CFFGlyphFeature(tuple(sorted(cells)), round(width / height, 2), len(contours), bitmap)


def internal_type2_glyph_geometry_impl(
    charstring: bytes,
    *,
    local_subrs: tuple[bytes, ...],
    global_subrs: tuple[bytes, ...],
    collect_contours: bool,
) -> tuple[list[list[tuple[float, float]]], tuple[float, float, float, float] | None]:
    stack: list[float] = []
    contours: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    current_min_x = inf
    current_min_y = inf
    current_max_x = -inf
    current_max_y = -inf
    bbox_min_x = inf
    bbox_min_y = inf
    bbox_max_x = -inf
    bbox_max_y = -inf
    current_has_points = False
    bbox_has_points = False
    x = 0.0
    y = 0.0
    stem_count = 0
    subr_bias = internal_type2_subr_bias(len(local_subrs))
    gsubr_bias = internal_type2_subr_bias(len(global_subrs))

    def flush_contour() -> None:
        nonlocal current
        nonlocal current_min_x, current_min_y, current_max_x, current_max_y
        nonlocal bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y
        nonlocal current_has_points, bbox_has_points
        if current:
            contours.append(current)
            current = []
        if current_has_points:
            bbox_min_x = min(bbox_min_x, current_min_x)
            bbox_min_y = min(bbox_min_y, current_min_y)
            bbox_max_x = max(bbox_max_x, current_max_x)
            bbox_max_y = max(bbox_max_y, current_max_y)
            bbox_has_points = True
            current_min_x = inf
            current_min_y = inf
            current_max_x = -inf
            current_max_y = -inf
            current_has_points = False

    def record_point(px: float, py: float) -> None:
        nonlocal current_min_x, current_min_y, current_max_x, current_max_y
        nonlocal current_has_points
        if collect_contours:
            current.append((px, py))
        current_min_x = min(current_min_x, px)
        current_min_y = min(current_min_y, py)
        current_max_x = max(current_max_x, px)
        current_max_y = max(current_max_y, py)
        current_has_points = True

    def move(dx: float, dy: float) -> None:
        nonlocal x, y
        flush_contour()
        x += dx
        y += dy
        record_point(x, y)

    def line(dx: float, dy: float) -> None:
        nonlocal x, y
        x += dx
        y += dy
        record_point(x, y)

    def curve(dx1: float, dy1: float, dx2: float, dy2: float, dx3: float, dy3: float) -> None:
        nonlocal x, y
        x0, y0 = x, y
        x1, y1 = x + dx1, y + dy1
        x2, y2 = x1 + dx2, y1 + dy2
        x3, y3 = x2 + dx3, y2 + dy3
        for t in (0.25, 0.5, 0.75, 1.0):
            mt = 1.0 - t
            record_point(
                mt**3 * x0 + 3 * mt * mt * t * x1 + 3 * mt * t * t * x2 + t**3 * x3,
                mt**3 * y0 + 3 * mt * mt * t * y1 + 3 * mt * t * t * y2 + t**3 * y3,
            )
        x, y = x3, y3

    def clear_stack() -> None:
        del stack[:]

    def execute(program: bytes, depth: int = 0) -> bool:
        """Interpret one Type 2 charstring, appending to the enclosing contour state.

        Returns whether the caller still owns an unflushed contour: ``True`` when the
        program simply ran out (so the caller must call ``flush_contour``), ``False``
        when ``endchar`` already flushed it or the charstring was malformed. A
        subroutine returning ``False`` aborts its caller the same way.

        The branches below are keyed by raw Type 2 operator bytes; each carries the
        operator's spec name. Operands are values above 31, plus 28 (a two-byte
        integer) and 255 (a 16.16 fixed-point number).
        """
        nonlocal stem_count
        if depth > TYPE2_MAX_SUBR_DEPTH:
            return False
        pos = 0
        try:
            while pos < len(program):
                byte = program[pos]
                if byte > 31 or byte in {28, 255}:
                    value, pos = CFFFont.internal_parse_number(program, pos)
                    stack.append(value)
                    continue
                pos += 1
                if byte in (1, 3, 18, 23):  # hstem, vstem, hstemhm, vstemhm
                    stem_count += len(stack) // 2
                    clear_stack()
                elif byte in (19, 20):  # hintmask, cntrmask -- skip the trailing mask bytes
                    stem_count += len(stack) // 2
                    clear_stack()
                    pos += (stem_count + 7) // 8
                elif byte == 4:  # vmoveto
                    if len(stack) > 1:
                        del stack[:-1]
                    move(0.0, stack[-1] if stack else 0.0)
                    clear_stack()
                elif byte == 21:  # rmoveto
                    if len(stack) > 2:
                        del stack[:-2]
                    dx = stack[-2] if len(stack) >= 2 else 0.0
                    dy = stack[-1] if stack else 0.0
                    move(dx, dy)
                    clear_stack()
                elif byte == 22:  # hmoveto
                    if len(stack) > 1:
                        del stack[:-1]
                    move(stack[-1] if stack else 0.0, 0.0)
                    clear_stack()
                elif byte == 5:  # rlineto
                    for i in range(0, len(stack) - 1, 2):
                        line(stack[i], stack[i + 1])
                    clear_stack()
                elif byte == 6:  # hlineto -- alternates horizontal/vertical
                    horizontal = True
                    for value in stack:
                        line(value, 0.0) if horizontal else line(0.0, value)
                        horizontal = not horizontal
                    clear_stack()
                elif byte == 7:  # vlineto -- alternates vertical/horizontal
                    vertical = True
                    for value in stack:
                        line(0.0, value) if vertical else line(value, 0.0)
                        vertical = not vertical
                    clear_stack()
                elif byte == 8:  # rrcurveto
                    for i in range(0, len(stack) - 5, 6):
                        curve(*stack[i : i + 6])
                    clear_stack()
                elif byte == 10:  # callsubr (local)
                    if stack:
                        subr_index = int(stack.pop()) + subr_bias
                        if 0 <= subr_index < len(local_subrs) and not execute(
                            local_subrs[subr_index], depth + 1
                        ):
                            return False
                elif byte == 11:  # return -- leave this subroutine, caller keeps going
                    return True
                elif byte == 14:  # endchar -- glyph complete
                    flush_contour()
                    return False
                elif byte == 24:  # rcurveline -- lines then one curve
                    line_count = len(stack) - 6
                    for i in range(0, line_count - 1, 2):
                        line(stack[i], stack[i + 1])
                    if line_count >= 0:
                        curve(*stack[line_count : line_count + 6])
                    clear_stack()
                elif byte == 25:  # rlinecurve -- lines then curves
                    line_count = len(stack) % 6
                    for i in range(0, line_count - 1, 2):
                        line(stack[i], stack[i + 1])
                    for i in range(line_count, len(stack) - 5, 6):
                        curve(*stack[i : i + 6])
                    clear_stack()
                elif byte == 26:  # vvcurveto
                    if len(stack) % 2:
                        line(stack.pop(0), 0.0)
                    for i in range(0, len(stack) - 3, 4):
                        curve(0.0, stack[i], stack[i + 1], stack[i + 2], 0.0, stack[i + 3])
                    clear_stack()
                elif byte == 27:  # hhcurveto
                    if len(stack) % 2:
                        line(0.0, stack.pop(0))
                    for i in range(0, len(stack) - 3, 4):
                        curve(stack[i], 0.0, stack[i + 1], stack[i + 2], stack[i + 3], 0.0)
                    clear_stack()
                elif byte == 29:  # callgsubr (global)
                    if stack:
                        subr_index = int(stack.pop()) + gsubr_bias
                        if 0 <= subr_index < len(global_subrs) and not execute(
                            global_subrs[subr_index], depth + 1
                        ):
                            return False
                elif byte in (30, 31):  # vhcurveto / hvcurveto -- alternating tangents
                    horizontal = byte == 31
                    args = list(stack)
                    clear_stack()
                    while len(args) >= 4:
                        if horizontal:  # this segment starts horizontal
                            dx1 = args.pop(0)
                            dy1 = 0.0
                            dx2 = args.pop(0)
                            dy2 = args.pop(0)
                            dy3 = args.pop(0)
                            dx3 = args.pop(0) if len(args) == 1 else 0.0
                        else:  # this segment starts vertical
                            dx1 = 0.0
                            dy1 = args.pop(0)
                            dx2 = args.pop(0)
                            dy2 = args.pop(0)
                            dx3 = args.pop(0)
                            dy3 = args.pop(0) if len(args) == 1 else 0.0
                        curve(dx1, dy1, dx2, dy2, dx3, dy3)
                        horizontal = not horizontal
                else:  # unrecognized operator -- drop its operands and continue
                    clear_stack()
            return True
        except (IndexError, ValueError):
            return False

    if not execute(charstring):
        bbox = (bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y) if bbox_has_points else None
        return contours, bbox
    flush_contour()
    bbox = (bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y) if bbox_has_points else None
    return contours, bbox


def internal_type2_subr_bias(count: int) -> int:
    if count < 1240:
        return 107
    if count < 33900:
        return 1131
    return 32768


def glyph_feature_distance(left: CFFGlyphFeature, right: CFFGlyphFeature) -> float:
    return compiled_feature_distance(
        left.cells,
        left.bitmap,
        left.aspect,
        left.contours,
        right.cells,
        right.bitmap,
        right.aspect,
        right.contours,
    )


SUSPICIOUS_TO_UNICODE = {"\ufffd", "£", "•"}
REPAIRABLE_TO_UNICODE = SUSPICIOUS_TO_UNICODE | {"5", "H"}
LEGITIMATE_MULTI_CHAR_GLYPHS = {"ff", "fi", "fl", "ffi", "ffl", "st"}


def is_repairable_to_unicode_label(label: str) -> bool:
    if len(label) == 1:
        return label in REPAIRABLE_TO_UNICODE
    if label in LEGITIMATE_MULTI_CHAR_GLYPHS:
        return False
    if any(ch in SUSPICIOUS_TO_UNICODE for ch in label):
        return True
    if len(label) > 3:
        return True
    return any(not (ch.isalnum() or ch.isspace()) for ch in label)


def internal_repair_candidate(
    glyph_id: int,
    label: str,
    features: dict[int, CFFGlyphFeature],
    labels: dict[int, str],
    distance_lookup: dict[int, float] | None = None,
) -> str | None:
    feature = features.get(glyph_id, EMPTY_FEATURE)
    if not feature.cells:
        return None
    candidates: list[tuple[float, str]] = []
    same_label = inf
    for other_id, other_label in labels.items():
        if other_id == glyph_id or len(other_label) != 1:
            continue
        if not (other_label.isalnum() or other_label in ".-+"):
            continue
        other_feature = features.get(other_id, EMPTY_FEATURE)
        if not other_feature.cells:
            continue
        distance = (
            distance_lookup[other_id]
            if distance_lookup is not None
            else glyph_feature_distance(feature, other_feature)
        )
        if other_label == label:
            same_label = min(same_label, distance)
        else:
            candidates.append((distance, other_label))
    if not candidates:
        return None
    best_distance, best_label = min(candidates, key=lambda item: item[0])
    if (label in SUSPICIOUS_TO_UNICODE or len(label) > 1) and best_distance < 2.3:
        return best_label
    if label == "5" and best_label == "S" and best_distance < 1.9:
        return best_label
    if label == "H" and best_label == "M" and best_distance < 1.8:
        return best_label
    if same_label < inf and best_distance + 0.35 < same_label and best_distance < 2.0:
        return best_label
    return None


@lru_cache(maxsize=64)
def cff_font_for_data(font_data: bytes) -> CFFFont:
    return CFFFont(font_data)


class CFFUnicodeRepairIndex:
    """Resolve suspicious ToUnicode entries lazily against one CFF program."""

    __slots__ = (
        "internal_candidate_gids",
        "internal_code_to_gid",
        "internal_features",
        "internal_font",
        "internal_gid_to_codes",
        "internal_labels",
        "internal_lock",
        "internal_repairable_gids",
        "internal_repairs",
        "internal_resolved_gids",
    )

    def __init__(
        self,
        font: CFFFont,
        mapping_items: tuple[tuple[bytes, int, str], ...],
    ) -> None:
        glyph_count = len(font.charstrings)
        labels: dict[int, str] = {}
        gid_to_codes: dict[int, list[bytes]] = {}
        code_to_gid: dict[bytes, int] = {}
        if glyph_count >= 2:
            for code_bytes, cid, value in mapping_items:
                gid = font.glyph_id_for_cid(cid)
                if gid >= glyph_count:
                    continue
                labels[gid] = value
                gid_to_codes.setdefault(gid, []).append(code_bytes)
                code_to_gid[code_bytes] = gid

        self.internal_font = font
        self.internal_labels = labels
        self.internal_gid_to_codes = {gid: tuple(codes) for gid, codes in gid_to_codes.items()}
        self.internal_code_to_gid = code_to_gid
        self.internal_repairable_gids = frozenset(
            gid for gid, label in labels.items() if is_repairable_to_unicode_label(label)
        )
        self.internal_candidate_gids = tuple(
            gid
            for gid, label in labels.items()
            if len(label) == 1 and (label.isalnum() or label in ".-+")
        )
        self.internal_features: dict[int, CFFGlyphFeature] = {}
        self.internal_repairs: dict[bytes, str] = {}
        self.internal_resolved_gids: set[int] = set()
        self.internal_lock = RLock()

    def repairs_for_codes(self, codes: Iterable[bytes]) -> dict[bytes, str]:
        """Return repairs for ``codes``, resolving each target glyph at most once."""
        requested_codes = tuple(dict.fromkeys(codes))
        if not requested_codes or not self.internal_repairable_gids:
            return {}
        with self.internal_lock:
            target_gids = tuple(
                dict.fromkeys(
                    gid
                    for code in requested_codes
                    if (gid := self.internal_code_to_gid.get(code)) in self.internal_repairable_gids
                    and gid not in self.internal_resolved_gids
                )
            )
            if target_gids:
                self.internal_resolve_gids(target_gids)
            return {
                code: replacement
                for code in requested_codes
                if (replacement := self.internal_repairs.get(code)) is not None
            }

    def all_repairs(self) -> dict[bytes, str]:
        return self.repairs_for_codes(self.internal_code_to_gid)

    def internal_resolve_gids(self, requested_gids: tuple[int, ...]) -> None:
        feature_gids = (*self.internal_candidate_gids, *requested_gids)
        for gid in feature_gids:
            if gid not in self.internal_features:
                self.internal_features[gid] = self.internal_font.glyph_feature(gid)

        candidate_gids = tuple(
            gid for gid in self.internal_candidate_gids if self.internal_features[gid].cells
        )
        target_gids = tuple(gid for gid in requested_gids if self.internal_features[gid].cells)
        distance_lookups: dict[int, dict[int, float]] = {}
        if (
            target_gids
            and candidate_gids
            and (len(self.internal_repairable_gids) * len(candidate_gids) >= 512)
        ):
            target_features = [self.internal_features[gid] for gid in target_gids]
            candidate_features = [self.internal_features[gid] for gid in candidate_gids]
            distance_matrix = compiled_feature_distance_matrix(
                [feature.cells for feature in target_features],
                [feature.bitmap for feature in target_features],
                [feature.aspect for feature in target_features],
                [feature.contours for feature in target_features],
                [feature.cells for feature in candidate_features],
                [feature.bitmap for feature in candidate_features],
                [feature.aspect for feature in candidate_features],
                [feature.contours for feature in candidate_features],
            )
            distance_lookups = {
                target_gid: {
                    candidate_gid: float(distance_matrix[target_index, candidate_index])
                    for candidate_index, candidate_gid in enumerate(candidate_gids)
                }
                for target_index, target_gid in enumerate(target_gids)
            }

        for glyph_id in target_gids:
            label = self.internal_labels[glyph_id]
            replacement = internal_repair_candidate(
                glyph_id,
                label,
                self.internal_features,
                self.internal_labels,
                distance_lookups.get(glyph_id),
            )
            if replacement is not None and replacement != label:
                for code_bytes in self.internal_gid_to_codes.get(glyph_id, ()):
                    self.internal_repairs[code_bytes] = replacement
        self.internal_resolved_gids.update(requested_gids)


@lru_cache(maxsize=64)
def cff_unicode_repair_index_for_data(
    font_data: bytes,
    mapping_items: tuple[tuple[bytes, int, str], ...],
) -> CFFUnicodeRepairIndex:
    return CFFUnicodeRepairIndex(cff_font_for_data(font_data), mapping_items)


def internal_type2_glyph_geometry(
    charstring: bytes,
    *,
    local_subrs: tuple[bytes, ...],
    global_subrs: tuple[bytes, ...],
    collect_contours: bool,
) -> tuple[list[list[tuple[float, float]]], tuple[float, float, float, float] | None]:
    """Dispatch through the domain adapter so existing instrumentation remains valid."""
    import sys

    adapter = sys.modules.get("core_pdf.impl.engine.spec.s_09_fonts.cff")
    hooked = getattr(adapter, "internal_type2_glyph_geometry", None) if adapter else None
    if hooked is not None and hooked is not internal_type2_glyph_geometry:
        return hooked(
            charstring,
            local_subrs=local_subrs,
            global_subrs=global_subrs,
            collect_contours=collect_contours,
        )
    return internal_type2_glyph_geometry_impl(
        charstring,
        local_subrs=local_subrs,
        global_subrs=global_subrs,
        collect_contours=collect_contours,
    )


__all__ = (
    "STANDARD_GLYPH_SIDS",
    "CFFFont",
    "CFFGlyphFeature",
    "CFFUnicodeRepairIndex",
    "REPAIRABLE_TO_UNICODE",
    "cff_font_for_data",
    "cff_unicode_repair_index_for_data",
    "glyph_feature_distance",
    "is_repairable_to_unicode_label",
)
