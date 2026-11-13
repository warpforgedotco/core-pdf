# SPDX-License-Identifier: AGPL-3.0-only

"""Read the tables of a TrueType (sfnt) font program.

Each table decodes the first time it is read. The readers keep fontTools' results on
damaged fonts, which core's output was built on: when a table fails to decode, the read
raises, but the fields decoded before the failure stay and later reads return them. A
font whose ``maxp`` has trailing bytes, for example, fails its first ``maxp`` read and
then reports its glyph count. Table data that runs past the end of the font, or a tag
the directory lacks, fails every time.

Glyph names follow the font's ``post`` table when it names every glyph, and are
otherwise ``.notdef`` and ``glyph00001``, ``glyph00002`` ... by glyph ID (the names
fontTools would build from the cmap only to be mapped back to IDs).
"""

from __future__ import annotations

import struct
from array import array
from collections.abc import Callable, Iterable
from sys import byteorder
from types import SimpleNamespace
from typing import Any

from core_pdf.impl.adobe_fonts._vendor.font_data.glyph_names import (
    MACINTOSH_GLYPH_NAMES,
    NEW_FONT_GLYPH_NAMES,
)

SFNT_VERSIONS = frozenset((b"\x00\x01\x00\x00", b"OTTO", b"true"))
DIRECTORY = struct.Struct(">4sHHHH")
DIRECTORY_ENTRY = struct.Struct(">4sLLL")
HEAD = struct.Struct(">llIIHHQQhhhhHHhhh")
MAXP_0_5 = struct.Struct(">iH")
MAXP_1_0_SIZE = 26
# hhea and vhea: version, 15 fields core does not read, then the long metric count.
METRICS_HEADER = struct.Struct(">L15hH")
POST_HEADER = struct.Struct(">llhhLLLLL")
GLYPH_HEADER = struct.Struct(">hhhhh")
BIG_ENDIAN = byteorder == "big"

Point = tuple[float, float]
Recording = list[tuple[str, tuple[Any, ...]]]


class Table(SimpleNamespace):
    """A decoded table: whatever fields its reader set, even if it then failed."""

    def take_glyph_order(self) -> list[str] | None:
        """The post table's glyph names, which can be read once."""
        try:
            order: list[str] | None = self.__dict__.pop("glyphOrder")
            return order
        except KeyError:
            raise ValueError("'post' glyph names already read") from None


def big_endian(code: str, data: bytes) -> array[int]:
    values = array(code)
    values.frombytes(data)
    if not BIG_ENDIAN:
        values.byteswap()
    return values


def fixed(value: int) -> float:
    return value / 65536


class SfntFont:
    """The table directory of one sfnt font program, and its tables on demand."""

    __slots__ = ("data", "entries", "tables", "glyph_order", "reverse_glyph_map", "placeholder")

    def __init__(self, data: bytes) -> None:
        self.data = data
        version = data[:4]
        if version == b"ttcf":
            raise ValueError("a font collection, not a font")
        if version in (b"wOFF", b"wOF2"):
            raise ValueError("WOFF font programs are not supported")
        if len(data) < DIRECTORY.size:
            raise ValueError("not a TrueType or OpenType font (not enough data)")
        version, table_count, *_ = DIRECTORY.unpack_from(data)
        if version not in SFNT_VERSIONS:
            raise ValueError("not a TrueType or OpenType font (bad sfntVersion)")
        entries: dict[str, tuple[int, int]] = {}
        for index in range(table_count):
            start = DIRECTORY.size + index * DIRECTORY_ENTRY.size
            tag, _checksum, offset, length = DIRECTORY_ENTRY.unpack(
                data[start : start + DIRECTORY_ENTRY.size]
            )
            entries[tag.decode("latin-1")] = (offset, length)
        self.entries = entries
        self.tables: dict[str, Table] = {}
        self.glyph_order: list[str] = []
        self.reverse_glyph_map: dict[str, int] | None = None
        self.placeholder = False

    def __contains__(self, tag: str) -> bool:
        return tag in self.tables or tag in self.entries or tag == "GlyphOrder"

    def keys(self) -> set[str]:
        return {"GlyphOrder", *self.entries, *self.tables}

    def table_data(self, tag: str) -> bytes:
        offset, length = self.entries[tag]
        data = self.data[offset : offset + length]
        if len(data) != length:
            raise ValueError(f"'{tag}' table runs past the end of the font")
        return data

    def table(self, tag: str) -> Table:
        table = self.tables.get(tag)
        if table is None:
            if tag not in self.entries:
                raise KeyError(f"'{tag}' table not found")
            data = self.table_data(tag)
            table = self.tables[tag] = Table()
            TABLE_READERS.get(tag, read_raw)(table, data, self)
        return table

    def optional_table(self, tag: str) -> Table | None:
        try:
            return self.table(tag)
        except KeyError:
            return None

    # Glyph names.

    def resolve_glyph_order(self) -> None:
        """Name the glyphs: the post table's names when it names them all, otherwise
        placeholders. Raises ValueError when the font gives no glyph count."""
        try:
            self.glyph_order = self.named_glyph_order()
            return
        except Exception:
            pass
        try:
            glyph_count = int(self.table("maxp").numGlyphs)
        except Exception as exc:
            raise ValueError("invalid TrueType glyph order") from exc
        if glyph_count <= 0:
            raise ValueError("invalid TrueType glyph order")
        self.glyph_order = placeholder_glyph_order(glyph_count)
        self.placeholder = True

    def named_glyph_order(self) -> list[str]:
        if "CFF " in self:
            raise ValueError("glyph names of a CFF outline table")
        if "post" in self:
            names = self.table("post").take_glyph_order()
            if names is not None and len(names) >= self.table("maxp").numGlyphs:
                return names
        return self.placeholder_glyph_order()

    def placeholder_glyph_order(self) -> list[str]:
        order = placeholder_glyph_order(int(self.table("maxp").numGlyphs))
        order[0] = ".notdef"
        self.placeholder = True
        if "cmap" in self:
            # Read where fontTools reads it, so a damaged cmap fails here the same way.
            for subtable in self.table("cmap").subtables:
                if subtable.is_unicode():
                    subtable.mapping()
        return order

    def glyph_name(self, glyph_id: int) -> str:
        try:
            return self.glyph_order[glyph_id]
        except IndexError:
            return "glyph%.5d" % glyph_id

    def glyph_names(self, glyph_ids: Iterable[int]) -> list[str]:
        order = self.glyph_order
        count = len(order)
        return [order[gid] if gid < count else "glyph%.5d" % gid for gid in glyph_ids]

    def glyph_id_map(self) -> dict[str, int]:
        reverse = self.reverse_glyph_map
        if reverse is None:
            reverse = self.reverse_glyph_map = {
                name: glyph_id for glyph_id, name in enumerate(self.glyph_order)
            }
        return reverse

    def glyph_id_for_name(self, name: str) -> int | None:
        return self.glyph_id_map().get(name)


PLACEHOLDER_NAMES = [".notdef"]


def placeholder_glyph_order(glyph_count: int) -> list[str]:
    global PLACEHOLDER_NAMES
    names = PLACEHOLDER_NAMES
    if glyph_count > len(names):
        names = PLACEHOLDER_NAMES = [
            *names,
            *("glyph%.5d" % gid for gid in range(len(names), glyph_count)),
        ]
    return names[:glyph_count] if glyph_count > 0 else []


def placeholder_glyph_id(name: str) -> int | None:
    if name == ".notdef":
        return 0
    if not name.startswith("glyph"):
        return None
    try:
        return int(name[5:])
    except ValueError:
        return None


# Table readers: each sets fields in the order fontTools does, so a failure leaves the same
# fields behind.


def read_raw(table: Table, data: bytes, font: SfntFont) -> None:
    table.data = data


def read_head(table: Table, data: bytes, font: SfntFont) -> None:
    (
        table.tableVersion,
        table.fontRevision,
        table.checkSumAdjustment,
        table.magicNumber,
        table.flags,
        table.unitsPerEm,
        table.created,
        table.modified,
        table.xMin,
        table.yMin,
        table.xMax,
        table.yMax,
        table.macStyle,
        table.lowestRecPPEM,
        table.fontDirectionHint,
        table.indexToLocFormat,
        table.glyphDataFormat,
    ) = HEAD.unpack(data[: HEAD.size])
    rest = data[HEAD.size :]
    if rest and rest != b"\0\0":
        raise ValueError("extra bytes at the end of 'head' table")


def read_maxp(table: Table, data: bytes, font: SfntFont) -> None:
    table.tableVersion, table.numGlyphs = MAXP_0_5.unpack(data[: MAXP_0_5.size])
    rest = data[MAXP_0_5.size :]
    if table.tableVersion != 0x00005000:
        if len(rest) < MAXP_1_0_SIZE:
            raise ValueError("'maxp' table too short")
        rest = rest[MAXP_1_0_SIZE:]
    if rest:
        raise ValueError("extra bytes at the end of 'maxp' table")


def read_metrics_header(table: Table, data: bytes, font: SfntFont) -> None:
    fields = METRICS_HEADER.unpack(data)
    table.tableVersion = fields[0]
    table.numberOfMetrics = fields[-1]


def metrics_reader(header_tag: str) -> Callable[[Table, bytes, SfntFont], None]:
    def read_metrics(table: Table, data: bytes, font: SfntFont) -> None:
        glyph_count = font.table("maxp").numGlyphs
        header = font.optional_table(header_tag)
        metric_count = int(header.numberOfMetrics) if header is not None else glyph_count
        metric_count = min(metric_count, glyph_count)
        bearing_count = glyph_count - metric_count
        size = 4 * metric_count + 2 * bearing_count
        if len(data) < size:
            raise ValueError(f"not enough metrics table data: expected {size} bytes")
        metrics = struct.unpack(">" + "Hh" * metric_count, data[: 4 * metric_count])
        bearings = big_endian("h", data[4 * metric_count : size])
        table.metrics = by_name = {}
        order = font.glyph_order
        for index in range(metric_count):
            by_name[order[index]] = metrics[index * 2 : index * 2 + 2]
        last_advance = metrics[-2]
        for index in range(bearing_count):
            by_name[order[index + metric_count]] = (last_advance, bearings[index])

    return read_metrics


def read_loca(table: Table, data: bytes, font: SfntFont) -> None:
    long_format = font.table("head").indexToLocFormat
    locations = big_endian("I" if long_format else "H", data)
    if not long_format:
        locations = array("I", (2 * location for location in locations))
    table.locations = locations


def read_post(table: Table, data: bytes, font: SfntFont) -> None:
    fields = POST_HEADER.unpack(data[: POST_HEADER.size])
    table.formatType = fixed(fields[0])
    data = data[POST_HEADER.size :]
    match table.formatType:
        case 1.0:
            table.glyphOrder = list(MACINTOSH_GLYPH_NAMES[: font.table("maxp").numGlyphs])
        case 2.0:
            read_post_names(table, data, font)
        case 3.0:
            table.glyphOrder = None
        case 4.0:
            read_post_codes(table, data, font)
        case _:
            raise ValueError(f"'post' table format {table.formatType} not supported")


def read_post_names(table: Table, data: bytes, font: SfntFont) -> None:
    glyph_count = font.table("maxp").numGlyphs
    (count,) = struct.unpack(">H", data[:2])
    count = min(int(count), glyph_count)
    indices = big_endian("H", data[2 : 2 + 2 * count])
    strings = data[2 + 2 * count :]
    extra = pascal_strings(strings, max(indices) - 257)
    table.glyphOrder = order = [""] * int(glyph_count)
    for glyph_id in range(count):
        index = indices[glyph_id]
        if index > 257:
            try:
                name = extra[index - 258]
            except IndexError:
                name = ""
        else:
            name = MACINTOSH_GLYPH_NAMES[index]
        order[glyph_id] = name
    dedupe_post_names(order, glyph_count)


def read_post_codes(table: Table, data: bytes, font: SfntFont) -> None:
    glyph_count = font.table("maxp").numGlyphs
    codes = big_endian("H", data)
    table.glyphOrder = order = [""] * int(glyph_count)
    for index in range(min(len(codes), glyph_count)):
        code = codes[index]
        if code == 0xFFFF:
            order[index] = ""
        else:
            order[index] = NEW_FONT_GLYPH_NAMES.get(code) or "uni%04X" % code
    dedupe_post_names(order, glyph_count)


def pascal_strings(data: bytes, count: int) -> list[str]:
    strings = []
    index = 0
    length_of_data = len(data)
    for _ in range(count):
        length = 0 if length_of_data <= index else data[index]
        index += 1
        if length_of_data <= index + length - 1:
            name = ""
        else:
            name = data[index : index + length].decode("latin-1")
        strings.append(name)
        index += length
    return strings


def dedupe_post_names(order: list[str], glyph_count: int) -> None:
    """Name unnamed glyphs glyphNNNNN and suffix repeated names .1, .2 ..."""
    seen: dict[str, int] = {}
    names = set(order)
    for index in range(glyph_count):
        name = post_name = order[index]
        if name == "":
            name = "glyph%.5d" % index
        if name in seen:
            suffix = seen[name]
            while f"{name}.{suffix}" in names:
                suffix += 1
            seen[name] = suffix + 1
            name = f"{name}.{suffix}"
        seen[name] = 1
        if name != post_name:
            order[index] = name


# cmap.


class CmapSubtable:
    """One cmap subtable, decoded when its mapping is first read. A failed decode fails
    again on the next read, except where fontTools had already dropped the data (format
    4), after which the subtable has no mapping."""

    __slots__ = ("platform_id", "encoding_id", "format", "data", "font", "cached", "nGroups")

    def __init__(self, platform_id: int, encoding_id: int, format_: int) -> None:
        self.platform_id = platform_id
        self.encoding_id = encoding_id
        self.format = format_
        self.data: bytes | None = None
        self.font: SfntFont | None = None
        self.cached: dict[int, str] | None = None

    def is_unicode(self) -> bool:
        return self.platform_id == 0 or (self.platform_id == 3 and self.encoding_id in (0, 1, 10))

    def read_header(self, data: bytes, font: SfntFont) -> None:
        if self.format in (12, 13):
            _format, _reserved, length, _language, groups = struct.unpack(">HHLLL", data[:16])
            if not len(data) == 16 + groups * 12 == length:
                raise ValueError("corrupt cmap subtable")
            self.nGroups = groups
            self.data = data[16:]
        elif self.format == 14:
            struct.unpack(">HLL", data[:10])
            self.data = data[10:]
        elif self.format in CMAP_DECODERS:
            _format, length, _language = struct.unpack(">HHH", data[:6])
            if len(data) != length:
                raise ValueError("corrupt cmap subtable")
            self.data = data[6:]
        else:
            self.data = data
        self.font = font

    def mapping(self) -> dict[int, str]:
        cached = self.cached
        if cached is not None:
            return cached
        if self.data is None or self.format not in CMAP_DECODERS:
            raise AttributeError("cmap")
        codes, glyph_ids = CMAP_DECODERS[self.format](self)
        self.data = None
        assert self.font is not None
        names = self.font.glyph_names(glyph_ids)
        self.cached = cached = {
            code: name for code, gid, name in zip(codes, glyph_ids, names, strict=True) if gid
        }
        return cached


def decode_format_0(subtable: CmapSubtable) -> tuple[Iterable[int], list[int]]:
    data = subtable.data or b""
    if len(data) + 6 != 262:
        raise ValueError("format 0 cmap subtable not 262 bytes")
    return range(len(data)), list(data)


def decode_format_2(subtable: CmapSubtable) -> tuple[Iterable[int], list[int]]:
    data = subtable.data or b""
    keys = [key // 8 for key in big_endian("H", data[:512])]
    data = data[512:]
    headers = []
    position = 0
    for _ in range(max(keys) + 1):
        first, count, delta, range_offset = struct.unpack(">HHhH", data[position : position + 8])
        position += 8
        start = position + range_offset - 2
        headers.append((first, count, delta, big_endian("H", data[start : start + count * 2])))
    mapping: dict[int, int] = {}
    for first_byte in range(256):
        header_index = keys[first_byte]
        first, count, delta, glyphs = headers[header_index]
        if header_index == 0:
            if first_byte < first or first_byte >= first + count:
                continue
            gid = glyphs[first_byte - first]
            if gid:
                mapping[first_byte] = (gid + delta) % 65536
        elif count:
            base = first_byte * 256 + first
            for index in range(count):
                gid = glyphs[index]
                if gid:
                    mapping[base + index] = (gid + delta) % 65536
    return list(mapping), list(mapping.values())


def decode_format_4(subtable: CmapSubtable) -> tuple[Iterable[int], list[int]]:
    data = subtable.data or b""
    segment_count = struct.unpack(">4H", data[:8])[0] // 2
    values = big_endian("H", data[8:])
    subtable.data = None
    ends = values[:segment_count]
    values = values[segment_count + 1 :]
    starts = values[:segment_count]
    values = values[segment_count:]
    deltas = values[:segment_count]
    values = values[segment_count:]
    range_offsets = values[:segment_count]
    glyph_array = values[segment_count:]
    codes: list[int] = []
    glyph_ids: list[int] = []
    for segment in range(len(starts) - 1):
        start = starts[segment]
        delta = deltas[segment]
        range_offset = range_offsets[segment]
        partial = range_offset // 2 - start + segment - len(range_offsets)
        segment_codes = range(start, ends[segment] + 1)
        codes.extend(segment_codes)
        if range_offset == 0:
            glyph_ids.extend((code + delta) & 0xFFFF for code in segment_codes)
            continue
        for code in segment_codes:
            index = code + partial
            if index >= len(glyph_array):
                raise ValueError("format 4 cmap glyph index out of range")
            gid = glyph_array[index]
            glyph_ids.append((gid + delta) & 0xFFFF if gid else 0)
    return codes, glyph_ids


def decode_format_6(subtable: CmapSubtable) -> tuple[Iterable[int], list[int]]:
    data = subtable.data or b""
    first, count = struct.unpack(">HH", data[:4])
    glyph_ids = big_endian("H", data[4 : 4 + 2 * count]).tolist()
    return range(first, first + len(glyph_ids)), glyph_ids


def decode_format_12_or_13(subtable: CmapSubtable) -> tuple[Iterable[int], list[int]]:
    data = subtable.data or b""
    groups = big_endian("I", data[: subtable.nGroups * 12])
    codes: list[int] = []
    glyph_ids: list[int] = []
    constant = subtable.format == 13
    for group in range(subtable.nGroups):
        start, end, gid = groups[group * 3 : group * 3 + 3]
        count = 1 + end - start
        codes.extend(range(start, end + 1))
        glyph_ids.extend([gid] * count if constant else range(gid, gid + count))
    return codes, glyph_ids


def decode_format_14(subtable: CmapSubtable) -> tuple[Iterable[int], list[int]]:
    # Variation sequences map no single code; fontTools gives this subtable an empty map.
    return (), []


CMAP_DECODERS: dict[int, Callable[[CmapSubtable], tuple[Iterable[int], list[int]]]] = {
    0: decode_format_0,
    2: decode_format_2,
    4: decode_format_4,
    6: decode_format_6,
    12: decode_format_12_or_13,
    13: decode_format_12_or_13,
    14: decode_format_14,
}


def read_cmap(table: Table, data: bytes, font: SfntFont) -> None:
    version, count = struct.unpack(">HH", data[:4])
    table.tableVersion = version
    subtables: list[CmapSubtable] = []
    table.subtables = subtables
    first_at: dict[int, int] = {}
    for index in range(count):
        platform_id, encoding_id, offset = struct.unpack(
            ">HHl", data[4 + index * 8 : 4 + (index + 1) * 8]
        )
        format_, length = struct.unpack(">HH", data[offset : offset + 4])
        if format_ in (8, 10, 12, 13):
            format_, _reserved, length = struct.unpack(">HHL", data[offset : offset + 8])
        elif format_ == 14:
            format_, length = struct.unpack(">HL", data[offset : offset + 6])
        if not length:
            continue
        subtable = CmapSubtable(platform_id, encoding_id, format_)
        subtable.read_header(data[offset : offset + int(length)], font)
        if offset in first_at:
            # Two records sharing one subtable share its mapping, read now.
            subtable.data = None
            subtable.cached = subtables[first_at[offset]].mapping()
        else:
            first_at[offset] = index
        subtables.append(subtable)


def cmap_subtable(cmap: Table, platform_id: int, encoding_id: int) -> CmapSubtable | None:
    for subtable in cmap.subtables:
        if subtable.platform_id == platform_id and subtable.encoding_id == encoding_id:
            return subtable
    return None


BEST_CMAP_ORDER = ((3, 10), (0, 6), (0, 4), (3, 1), (0, 3), (0, 2), (0, 1), (0, 0))


def best_cmap(cmap: Table) -> dict[int, str] | None:
    """The Unicode mapping HarfBuzz would pick: the widest repertoire, Windows first."""
    for platform_id, encoding_id in BEST_CMAP_ORDER:
        subtable = cmap_subtable(cmap, platform_id, encoding_id)
        if subtable is not None:
            return subtable.mapping()
    return None


# glyf.

FLAG_ON_CURVE = 0x01
FLAG_X_SHORT = 0x02
FLAG_Y_SHORT = 0x04
FLAG_REPEAT = 0x08
FLAG_X_SAME = 0x10
FLAG_Y_SAME = 0x20
FLAG_CUBIC = 0x80
KEEP_FLAGS = 0x01 | 0x40 | 0x80

ARG_1_AND_2_ARE_WORDS = 0x0001
ARGS_ARE_XY_VALUES = 0x0002
WE_HAVE_A_SCALE = 0x0008
MORE_COMPONENTS = 0x0020
WE_HAVE_AN_X_AND_Y_SCALE = 0x0040
WE_HAVE_A_TWO_BY_TWO = 0x0080
WE_HAVE_INSTRUCTIONS = 0x0100
SCALED_COMPONENT_OFFSET = 0x0800
UNSCALED_COMPONENT_OFFSET = 0x1000

IDENTITY = (1, 0, 0, 1, 0, 0)


class GlyphComponent:
    __slots__ = ("name", "flags", "x", "y", "first_point", "second_point", "matrix")

    name: str
    flags: int
    x: int
    y: int
    first_point: int
    second_point: int

    def __init__(self) -> None:
        self.matrix: tuple[tuple[float, float], tuple[float, float]] | None = None

    def transform(self) -> tuple[float, float, float, float, float, float]:
        """The component's (xx, xy, yx, yy, dx, dy); raises AttributeError for a component
        placed by matching points, which has no offset."""
        if self.matrix is not None:
            (xx, xy), (yx, yy) = self.matrix
            return (xx, xy, yx, yy, self.x, self.y)
        return (1, 0, 0, 1, self.x, self.y)


def f2dot14(value: int) -> float:
    return value / 16384


class Glyph:
    """One glyf entry. expand() decodes it once: a body that fails to decode raises the
    first time and leaves the header fields for later reads."""

    __slots__ = (
        "data",
        "numberOfContours",
        "xMin",
        "yMin",
        "xMax",
        "yMax",
        "components",
        "end_points",
        "coordinates",
        "flags",
    )

    def __init__(self, data: bytes) -> None:
        if data:
            self.data: bytes | None = data
        else:
            self.data = None
            self.numberOfContours = 0

    def is_composite(self) -> bool:
        data = self.data
        if data is not None:
            return struct.unpack(">h", data[:2])[0] == -1
        return self.numberOfContours == -1

    def has_bounds(self) -> bool:
        return hasattr(self, "xMin")

    def expand(self, glyf: Table) -> None:
        data = self.data
        if data is None:
            return
        (self.numberOfContours, self.xMin, self.yMin, self.xMax, self.yMax) = GLYPH_HEADER.unpack(
            data[: GLYPH_HEADER.size]
        )
        self.data = None
        body = data[GLYPH_HEADER.size :]
        if self.numberOfContours == 0:
            return
        if self.numberOfContours == -1:
            self.decode_components(body, glyf)
        else:
            self.decode_coordinates(body)

    def decode_components(self, data: bytes, glyf: Table) -> None:
        self.components: list[GlyphComponent] = []
        components = self.components
        more = True
        while more:
            component = GlyphComponent()
            flags, glyph_id = struct.unpack(">HH", data[:4])
            # Unlike a cmap's IDs, a component's ID must name a glyph.
            component.name = glyf.font.glyph_order[glyph_id]
            data = data[4:]
            if flags & ARG_1_AND_2_ARE_WORDS:
                if flags & ARGS_ARE_XY_VALUES:
                    component.x, component.y = struct.unpack(">hh", data[:4])
                else:
                    component.first_point, component.second_point = struct.unpack(">HH", data[:4])
                data = data[4:]
            else:
                if flags & ARGS_ARE_XY_VALUES:
                    component.x, component.y = struct.unpack(">bb", data[:2])
                else:
                    component.first_point, component.second_point = struct.unpack(">BB", data[:2])
                data = data[2:]
            if flags & WE_HAVE_A_SCALE:
                (scale,) = struct.unpack(">h", data[:2])
                component.matrix = ((f2dot14(scale), 0), (0, f2dot14(scale)))
                data = data[2:]
            elif flags & WE_HAVE_AN_X_AND_Y_SCALE:
                x_scale, y_scale = struct.unpack(">hh", data[:4])
                component.matrix = ((f2dot14(x_scale), 0), (0, f2dot14(y_scale)))
                data = data[4:]
            elif flags & WE_HAVE_A_TWO_BY_TWO:
                xx, xy, yx, yy = struct.unpack(">hhhh", data[:8])
                component.matrix = ((f2dot14(xx), f2dot14(xy)), (f2dot14(yx), f2dot14(yy)))
                data = data[8:]
            component.flags = flags
            more = bool(flags & MORE_COMPONENTS)
            components.append(component)
        if any(component.flags & WE_HAVE_INSTRUCTIONS for component in components):
            struct.unpack(">h", data[:2])

    def decode_coordinates(self, data: bytes) -> None:
        contours = self.numberOfContours
        self.end_points = end_points = big_endian("H", data[: 2 * contours]).tolist()
        position = 2 * contours
        (instruction_length,) = struct.unpack(">h", data[position : position + 2])
        position += 2 + instruction_length
        count = end_points[-1] + 1
        flags = bytearray(count)
        x_format = [">"]
        y_format = [">"]
        index = 0
        while True:
            flag = data[position]
            position += 1
            repeat = 1
            if flag & FLAG_REPEAT:
                repeat = data[position] + 1
                position += 1
            for _ in range(repeat):
                if flag & FLAG_X_SHORT:
                    x_format.append("B")
                elif not flag & FLAG_X_SAME:
                    x_format.append("h")
                if flag & FLAG_Y_SHORT:
                    y_format.append("B")
                elif not flag & FLAG_Y_SAME:
                    y_format.append("h")
                flags[index] = flag
                index += 1
            if index >= count:
                break
        if index != count:
            raise ValueError("bad glyph flags")
        x_size = struct.calcsize("".join(x_format))
        y_size = struct.calcsize("".join(y_format))
        xs = struct.unpack("".join(x_format), data[position : position + x_size])
        ys = struct.unpack("".join(y_format), data[position + x_size : position + x_size + y_size])
        coordinates: list[float] = [0.0] * (2 * count)
        x_index = y_index = 0
        x = y = 0.0
        for point in range(count):
            flag = flags[point]
            if flag & FLAG_X_SHORT:
                dx = xs[x_index] if flag & FLAG_X_SAME else -xs[x_index]
                x_index += 1
            elif flag & FLAG_X_SAME:
                dx = 0
            else:
                dx = xs[x_index]
                x_index += 1
            if flag & FLAG_Y_SHORT:
                dy = ys[y_index] if flag & FLAG_Y_SAME else -ys[y_index]
                y_index += 1
            elif flag & FLAG_Y_SAME:
                dy = 0
            else:
                dy = ys[y_index]
                y_index += 1
            x += dx
            y += dy
            coordinates[2 * point] = x
            coordinates[2 * point + 1] = y
        if x_index != len(xs) or y_index != len(ys):
            raise ValueError("glyph coordinate count mismatch")
        self.coordinates = coordinates
        for point in range(len(flags)):
            flags[point] &= KEEP_FLAGS
        self.flags = flags


def read_glyf(table: Table, data: bytes, font: SfntFont) -> None:
    table.font = font
    locations = font.table("loca").locations
    position = int(locations[0])
    table.glyphs = glyphs = {}
    order = font.glyph_order
    for index in range(len(locations) - 1):
        try:
            name = order[index]
        except IndexError:
            name = "ttxautoglyph%s" % index
        end = int(locations[index + 1])
        glyph_data = data[position:end]
        if len(glyph_data) != end - position:
            raise ValueError("not enough 'glyf' table data")
        glyphs[name] = Glyph(glyph_data)
        position = end


def glyf_glyph(glyf: Table, name: str) -> Glyph:
    glyph: Glyph = glyf.glyphs[name]
    glyph.expand(glyf)
    return glyph


TABLE_READERS: dict[str, Callable[[Table, bytes, SfntFont], None]] = {
    "head": read_head,
    "maxp": read_maxp,
    "hhea": read_metrics_header,
    "vhea": read_metrics_header,
    "hmtx": metrics_reader("hhea"),
    "vmtx": metrics_reader("vhea"),
    "loca": read_loca,
    "post": read_post,
    "cmap": read_cmap,
    "glyf": read_glyf,
}


# Drawing: the outline of a glyph as pen operations, components drawn in place.


class GlyphOutlines:
    """Draws glyf outlines the way fontTools' glyph set and decomposing recording pen do:
    a top-level glyph moved by its left side bearing minus xMin, its components looked
    up by name and transformed, a component without metrics skipped."""

    __slots__ = ("font", "glyf", "metrics", "vertical", "depth")

    def __init__(self, font: SfntFont) -> None:
        self.font = font
        self.glyf = font.table("glyf")
        self.metrics = font.table("hmtx").metrics
        vertical = font.optional_table("vmtx")
        self.vertical = getattr(vertical, "metrics", None) if vertical is not None else None
        self.depth = 0

    def check_metrics(self, name: str) -> int | float:
        lsb: int | float = self.metrics[name][1]
        if self.vertical is not None:
            self.vertical[name]  # noqa: B018 - a glyph without vertical metrics fails
        return lsb

    def record(self, name: str) -> Recording:
        recording: Recording = []
        lsb = self.check_metrics(name)
        self.draw(name, lsb, recording, None)
        return recording

    def draw(
        self, name: str, lsb: int | float, recording: Recording, transform: tuple[float, ...] | None
    ) -> None:
        glyph = glyf_glyph(self.glyf, name)
        offset = lsb - glyph.xMin if glyph.has_bounds() else 0
        depth = self.depth
        self.depth += 1
        try:
            if depth:
                offset = 0
            if glyph.is_composite():
                for component in glyph.components:
                    combined: tuple[float, ...] = component.transform()
                    if transform is not None:
                        combined = compose(transform, combined)
                    self.add_component(component.name, combined, recording)
                return
            emit = recording.append if transform is None else transformed(recording, transform)
            draw_simple(glyph, offset, emit)
        finally:
            self.depth -= 1

    def add_component(self, name: str, transform: tuple[float, ...], recording: Recording) -> None:
        try:
            lsb = self.check_metrics(name)
        except KeyError:
            return
        self.draw(name, lsb, recording, None if tuple(transform) == IDENTITY else transform)


def compose(outer: tuple[float, ...], inner: tuple[float, ...]) -> tuple[float, ...]:
    xx1, xy1, yx1, yy1, dx1, dy1 = inner
    xx2, xy2, yx2, yy2, dx2, dy2 = outer
    return (
        xx1 * xx2 + xy1 * yx2,
        xx1 * xy2 + xy1 * yy2,
        yx1 * xx2 + yy1 * yx2,
        yx1 * xy2 + yy1 * yy2,
        xx2 * dx1 + yx2 * dy1 + dx2,
        xy2 * dx1 + yy2 * dy1 + dy2,
    )


def transform_point(transform: tuple[float, ...], point: Point) -> Point:
    x, y = point
    xx, xy, yx, yy, dx, dy = transform
    return (xx * x + yx * y + dx, xy * x + yy * y + dy)


def transformed(
    recording: Recording, transform: tuple[float, ...]
) -> Callable[[tuple[str, tuple[Any, ...]]], None]:
    def emit(operation: tuple[str, tuple[Any, ...]]) -> None:
        operator, points = operation
        recording.append(
            (
                operator,
                tuple(
                    point if point is None else transform_point(transform, point)
                    for point in points
                ),
            )
        )

    return emit


def as_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def draw_simple(
    glyph: Glyph, offset: int | float, emit: Callable[[tuple[str, tuple[Any, ...]]], None]
) -> None:
    if glyph.numberOfContours <= 0:
        return
    coordinates = glyph.coordinates
    if offset:
        coordinates = [
            value + offset if index % 2 == 0 else value for index, value in enumerate(coordinates)
        ]
    points = [
        (as_number(coordinates[2 * index]), as_number(coordinates[2 * index + 1]))
        for index in range(len(coordinates) // 2)
    ]
    flags = glyph.flags
    start = 0
    for end in glyph.end_points:
        end += 1
        draw_contour(points[start:end], flags[start:end], emit)
        start = end


def midpoint(a: Point, b: Point) -> Point:
    x = (a[0] + b[0]) * 0.5
    y = (a[1] + b[1]) * 0.5
    return (as_number(x), as_number(y))


def draw_contour(
    contour: list[Any], flags: bytearray, emit: Callable[[tuple[str, tuple[Any, ...]]], None]
) -> None:
    on_curve = [FLAG_ON_CURVE & flag for flag in flags]
    cubic_flags = [FLAG_CUBIC & flag for flag in flags]
    if 1 not in on_curve:
        if any(cubic_flags) and not all(cubic_flags):
            raise ValueError("mixed cubic and quadratic contour")
        if all(cubic_flags):
            count = len(contour)
            if count % 2:
                raise ValueError("odd number of cubic off-curve points")
            emit(("moveTo", (midpoint(contour[-1], contour[0]),)))
            for index in range(0, count, 2):
                after = contour[index + 2 if index + 2 < count else 0]
                emit(
                    (
                        "curveTo",
                        (contour[index], contour[index + 1], midpoint(contour[index + 1], after)),
                    )
                )
        else:
            emit(("qCurveTo", (*contour, None)))
        emit(("closePath", ()))
        return
    first = on_curve.index(1) + 1
    contour = contour[first:] + contour[:first]
    on_curve = on_curve[first:] + on_curve[:first]
    cubic_flags = cubic_flags[first:] + cubic_flags[:first]
    emit(("moveTo", (contour[-1],)))
    while contour:
        next_on = on_curve.index(1) + 1
        if next_on == 1:
            if len(contour) > 1:
                emit(("lineTo", (contour[0],)))
        else:
            segment_flags = cubic_flags[: next_on - 1]
            if any(segment_flags):
                if not all(segment_flags):
                    raise ValueError("mixed cubic and quadratic segment")
                if next_on < 3 or (next_on - 1) % 2:
                    raise ValueError("bad cubic segment")
                for index in range(0, next_on - 3, 2):
                    emit(
                        (
                            "curveTo",
                            (
                                contour[index],
                                contour[index + 1],
                                midpoint(contour[index + 1], contour[index + 2]),
                            ),
                        )
                    )
                emit(("curveTo", tuple(contour[next_on - 3 : next_on])))
            else:
                emit(("qCurveTo", tuple(contour[:next_on])))
        contour = contour[next_on:]
        on_curve = on_curve[next_on:]
        cubic_flags = cubic_flags[next_on:]
    emit(("closePath", ()))


__all__ = (
    "Glyph",
    "GlyphOutlines",
    "SfntFont",
    "Table",
    "best_cmap",
    "cmap_subtable",
    "glyf_glyph",
    "placeholder_glyph_id",
    "placeholder_glyph_order",
)
