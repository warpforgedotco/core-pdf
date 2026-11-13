# SPDX-License-Identifier: AGPL-3.0-only

"""The glyphs of an OpenType font's ``CFF `` or ``CFF2`` table, by name.

A ``CFF `` table is read by core's CFF parser; its glyphs are named by its charset
(``cid00042`` for a CID-keyed font). A ``CFF2`` table names no glyphs, so its glyphs
take the font's names: the ``post`` table's when it names them all, otherwise names
built from the cmap (an AGLFN name, ``uniXXXX`` or ``uXXXXX``, ``.alt1`` for repeats)
and ``glyphNNNNN`` for the rest. A CFF2 table is read at its default instance: each
``blend`` keeps its default values.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

from core_pdf.impl.adobe_fonts._vendor.font_data.cff_tables import CFF_STANDARD_STRINGS
from core_pdf.impl.adobe_fonts._vendor.font_data.glyph_names import NEW_FONT_GLYPH_NAMES
from core_pdf.impl.adobe_fonts.cff.font import CFFFont as BaseCFFFont
from core_pdf.impl.fonts.outline_pens import Recording, Type2Outline
from core_pdf.impl.fonts.program_cff import CFFFont
from core_pdf.impl.fonts.sfnt import SfntFont, placeholder_glyph_order


class CffGlyphs:
    """Charstrings by glyph name, and what executing one needs."""

    __slots__ = ("names", "by_name", "charstrings", "global_subrs", "cff2", "subrs_for", "regions")

    def __init__(
        self,
        names: list[str],
        charstrings: Sequence[bytes],
        global_subrs: Sequence[bytes],
        subrs_for: Sequence[Sequence[bytes]],
        *,
        cff2: bool = False,
        regions: Sequence[int] = (),
        default_vsindex: Sequence[int] = (),
    ) -> None:
        self.names = names
        # A repeated name finds the last glyph with it, as a dict built in order does.
        self.by_name = {name: index for index, name in enumerate(names)}
        self.charstrings = charstrings
        self.global_subrs = global_subrs
        self.subrs_for = subrs_for
        self.cff2 = cff2
        self.regions = (tuple(regions), tuple(default_vsindex))

    def record(self, name: str) -> Recording:
        glyph_id = self.by_name[name]
        regions, default_vsindex = self.regions
        outline = Type2Outline(
            self.subrs_for[glyph_id],
            self.global_subrs,
            region_count=regions.__getitem__ if self.cff2 else None,
            default_vsindex=default_vsindex[glyph_id] if self.cff2 else 0,
            cff2=self.cff2,
        )
        return outline.record(self.charstrings[glyph_id])


# CFF.


def cff_glyphs(table: bytes) -> CffGlyphs:
    cff = CFFFont(table)
    count = len(cff.charstrings)
    names = [".notdef"] * count
    if cff.is_cid_keyed:
        for cid, glyph_id in cff.cid_to_gid.items():
            if 0 < glyph_id < count:
                names[glyph_id] = "cid%05d" % cid
    else:
        strings = {sid: name for name, sid in cff.custom_string_sids.items()}
        for sid, glyph_id in cff.cid_to_gid.items():
            if 0 < glyph_id < count:
                names[glyph_id] = (
                    CFF_STANDARD_STRINGS[sid] if sid < len(CFF_STANDARD_STRINGS) else strings[sid]
                )
    subrs = [cff.local_subrs_for_glyph(glyph_id) for glyph_id in range(count)]
    return CffGlyphs(names, cff.charstrings, cff.global_subrs, subrs)


# CFF2.

CFF2_TOP_CHARSTRINGS = 17
CFF2_TOP_VSTORE = 24
CFF2_TOP_FDARRAY = (12, 36)
CFF2_TOP_FDSELECT = (12, 37)
CFF2_PRIVATE = 18
CFF2_SUBRS = 19
CFF2_VSINDEX = 22
CFF2_BLEND = 23


def cff2_index(data: bytes, position: int) -> tuple[list[bytes], int]:
    (count,) = struct.unpack(">L", data[position : position + 4])
    position += 4
    if count == 0:
        return [], position
    offset_size = data[position]
    position += 1
    if not 1 <= offset_size <= 4:
        raise ValueError("invalid CFF2 INDEX")
    offsets = [
        int.from_bytes(data[position + index * offset_size : position + (index + 1) * offset_size])
        for index in range(count + 1)
    ]
    base = position + (count + 1) * offset_size - 1
    items = [data[base + offsets[index] : base + offsets[index + 1]] for index in range(count)]
    end = base + offsets[-1]
    if end > len(data) or any(a > b for a, b in zip(offsets, offsets[1:], strict=False)):
        raise ValueError("invalid CFF2 INDEX")
    return items, end


def cff2_dict(item: bytes, regions: Sequence[int] = ()) -> dict[int | tuple[int, int], list[float]]:
    """A CFF2 DICT; a blend keeps its default values."""
    result: dict[int | tuple[int, int], list[float]] = {}
    stack: list[float] = []
    vsindex = 0
    position = 0
    while position < len(item):
        b0 = item[position]
        if b0 <= 21 or b0 in (22, 23, 24, 25, 26, 27, 31):
            position += 1
            operator: int | tuple[int, int]
            if b0 == 12:
                operator = (12, item[position])
                position += 1
            else:
                operator = b0
            if operator == CFF2_BLEND:
                count = int(stack.pop())
                deltas = count * regions[vsindex]
                if deltas:
                    del stack[-deltas:]
                continue
            if operator == CFF2_VSINDEX:
                vsindex = int(stack[-1])
            result[operator] = stack
            stack = []
        else:
            value, position = BaseCFFFont.parse_number(item, position, dict_number=True)
            stack.append(value)
    return result


def variation_regions(data: bytes, offset: int) -> list[int]:
    """The region count of each item variation data of the table's VariationStore."""
    store = offset + 2  # after the store's uint16 length
    _format, _regions, count = struct.unpack(">HLH", data[store : store + 8])
    data_offsets = struct.unpack(f">{count}L", data[store + 8 : store + 8 + 4 * count])
    counts = []
    for data_offset in data_offsets:
        start = store + data_offset
        _items, _words, region_count = struct.unpack(">HHH", data[start : start + 6])
        counts.append(region_count)
    return counts


def fd_select(data: bytes, offset: int, glyph_count: int) -> list[int]:
    format_ = data[offset]
    if format_ == 0:
        return list(data[offset + 1 : offset + 1 + glyph_count])
    if format_ in (3, 4):
        count_format, range_format, size = (">H", ">HB", 3) if format_ == 3 else (">L", ">LH", 6)
        width = struct.calcsize(count_format)
        (count,) = struct.unpack(count_format, data[offset + 1 : offset + 1 + width])
        position = offset + 1 + width
        ranges = [
            struct.unpack(
                range_format, data[position + index * size : position + (index + 1) * size]
            )
            for index in range(count)
        ]
        (sentinel,) = struct.unpack(
            count_format, data[position + count * size : position + count * size + width]
        )
        result = [0] * glyph_count
        bounds = [*(first for first, _ in ranges), sentinel]
        for (first, fd), end in zip(ranges, bounds[1:], strict=True):
            for glyph_id in range(first, min(end, glyph_count)):
                result[glyph_id] = fd
        return result
    raise ValueError(f"unsupported CFF2 FDSelect format {format_}")


def cff2_glyphs(table: bytes, names: list[str]) -> CffGlyphs:
    header_size = table[2]
    (top_length,) = struct.unpack(">H", table[3:5])
    top = cff2_dict(table[header_size : header_size + top_length])
    global_subrs, _ = cff2_index(table, header_size + top_length)
    charstrings, _ = cff2_index(table, int(top[CFF2_TOP_CHARSTRINGS][0]))
    regions = (
        variation_regions(table, int(top[CFF2_TOP_VSTORE][0])) if CFF2_TOP_VSTORE in top else []
    )
    font_dicts, _ = cff2_index(table, int(top[CFF2_TOP_FDARRAY][0]))
    private_subrs: list[list[bytes]] = []
    private_vsindex: list[int] = []
    for font_dict in font_dicts:
        size, offset = (int(value) for value in cff2_dict(font_dict)[CFF2_PRIVATE])
        private = cff2_dict(table[offset : offset + size], regions)
        subrs: list[bytes] = []
        if CFF2_SUBRS in private:
            subrs, _ = cff2_index(table, offset + int(private[CFF2_SUBRS][0]))
        private_subrs.append(subrs)
        private_vsindex.append(int(private[CFF2_VSINDEX][0]) if CFF2_VSINDEX in private else 0)
    count = len(charstrings)
    selection = (
        fd_select(table, int(top[CFF2_TOP_FDSELECT][0]), count)
        if CFF2_TOP_FDSELECT in top
        else [0] * count
    )
    if len(names) > count:
        raise ValueError("CFF2 table has fewer glyphs than the font names")
    return CffGlyphs(
        names,
        charstrings,
        global_subrs,
        [private_subrs[fd] for fd in selection],
        cff2=True,
        regions=regions,
        default_vsindex=[private_vsindex[fd] for fd in selection],
    )


def cmap_glyph_order(font: SfntFont) -> list[str]:
    """Glyph names for a font whose post table does not name its glyphs, as fontTools
    builds them: from the lowest Unicode code point the cmap maps to each glyph."""
    order = placeholder_glyph_order(int(font.table("maxp").numGlyphs))
    order[0] = ".notdef"
    font.glyph_order = order
    lowest: dict[str, int] = {}
    if "cmap" in font:
        for subtable in font.table("cmap").subtables:
            if subtable.is_unicode():
                for code, name in subtable.mapping().items():
                    lowest[name] = min(lowest[name], code) if name in lowest else code
        # The names the cmap was read with are only placeholders.
        del font.tables["cmap"]
    uses: dict[str, int] = {}
    for index, placeholder in enumerate(order):
        if placeholder in lowest:
            code = lowest[placeholder]
            name = NEW_FONT_GLYPH_NAMES.get(code) or (
                "uni%04X" % code if code <= 0xFFFF else "u%X" % code
            )
            uses[name] = used = uses.get(name, 0) + 1
            order[index] = name if used == 1 else "%s.alt%d" % (name, used - 1)
    return order


def opentype_glyph_order(font: SfntFont) -> list[str]:
    """The names fontTools gives a CFF2 font's glyphs."""
    if "post" in font:
        names = font.table("post").take_glyph_order()
        if names is not None and len(names) >= font.table("maxp").numGlyphs:
            return names
    return cmap_glyph_order(font)


__all__ = ("CffGlyphs", "cff2_glyphs", "cff_glyphs", "opentype_glyph_order")
