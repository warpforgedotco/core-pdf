# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import struct


def tt_tables(data: bytes) -> dict[str, tuple[int, int]]:
    if len(data) < 12:
        raise ValueError("invalid TrueType table directory")
    try:
        num_tables = struct.unpack(">H", data[4:6])[0]
        tables: dict[str, tuple[int, int]] = {}
        if num_tables < 0 or num_tables > 64:
            raise ValueError("invalid TrueType table directory")
        for i in range(num_tables):
            off = 12 + i * 16
            if off + 16 > len(data):
                raise ValueError("invalid TrueType table directory")
            tag = data[off : off + 4].decode("ascii")
            tbl_off = struct.unpack(">I", data[off + 8 : off + 12])[0]
            length = struct.unpack(">I", data[off + 12 : off + 16])[0]
            if tbl_off + length > len(data):
                raise ValueError("invalid TrueType table directory")
            tables[tag] = (tbl_off, length)
        return tables
    except struct.error, UnicodeDecodeError, IndexError:
        raise ValueError("invalid TrueType table directory")


def tt_loca(data: bytes, tables: dict[str, tuple[int, int]], n_glyphs: int) -> list[int] | None:
    try:
        head_off = tables["head"][0]
        loca_off = tables["loca"][0]
        if head_off + 52 > len(data):
            raise ValueError("invalid TrueType loca table")
        idx_fmt = struct.unpack(">h", data[head_off + 50 : head_off + 52])[0]
        if idx_fmt == 0:
            if loca_off + (n_glyphs + 1) * 2 > len(data):
                raise ValueError("invalid TrueType loca table")
            return [
                struct.unpack(">H", data[loca_off + i * 2 : loca_off + i * 2 + 2])[0] * 2
                for i in range(n_glyphs + 1)
            ]
        if loca_off + (n_glyphs + 1) * 4 > len(data):
            raise ValueError("invalid TrueType loca table")
        return [
            struct.unpack(">I", data[loca_off + i * 4 : loca_off + i * 4 + 4])[0]
            for i in range(n_glyphs + 1)
        ]
    except struct.error, IndexError, KeyError:
        raise ValueError("invalid TrueType loca table")


def tt_glyph_bbox(
    data: bytes, glyf_off: int, loca: list[int], gid: int
) -> tuple[int, int, int, int] | None:
    try:
        start = loca[gid]
        end = loca[gid + 1]
        if start >= end or glyf_off + start + 10 > len(data):
            return None
        xmin, ymin, xmax, ymax = struct.unpack(
            ">hhhh", data[glyf_off + start + 2 : glyf_off + start + 10]
        )
        return (xmin, ymin, xmax, ymax)
    except struct.error, IndexError:
        return None


def tt_cmap(data: bytes, tables: dict[str, tuple[int, int]]) -> dict[int, int]:
    """Parse TrueType cmap (format 4 and 6) -> Unicode codepoint -> GID."""
    cmap_off, cmap_len = tables.get("cmap", (0, 0))
    if not cmap_off:
        return {}
    try:
        cmap = data[cmap_off : cmap_off + cmap_len]
        n_sub = struct.unpack(">H", cmap[2:4])[0]
        cp_to_gid: dict[int, int] = {}
        for i in range(n_sub):
            rec = 4 + i * 8
            if rec + 8 > len(cmap):
                raise ValueError("invalid TrueType cmap")
            sub_off = struct.unpack(">I", cmap[rec + 4 : rec + 8])[0]
            if sub_off + 2 > len(cmap):
                raise ValueError("invalid TrueType cmap")
            fmt = struct.unpack(">H", cmap[sub_off : sub_off + 2])[0]
            if fmt == 4:
                if sub_off + 14 > len(cmap):
                    raise ValueError("invalid TrueType cmap")
                seg_count = struct.unpack(">H", cmap[sub_off + 6 : sub_off + 8])[0] // 2
                base = sub_off + 14
                if base + seg_count * 2 + 2 + seg_count * 2 + seg_count * 2 + seg_count * 2 > len(
                    cmap
                ):
                    raise ValueError("invalid TrueType cmap")
                ends = [
                    struct.unpack(">H", cmap[base + j * 2 : base + j * 2 + 2])[0]
                    for j in range(seg_count)
                ]
                base2 = sub_off + 14 + seg_count * 2 + 2
                starts = [
                    struct.unpack(">H", cmap[base2 + j * 2 : base2 + j * 2 + 2])[0]
                    for j in range(seg_count)
                ]
                base3 = base2 + seg_count * 2
                deltas = [
                    struct.unpack(">h", cmap[base3 + j * 2 : base3 + j * 2 + 2])[0]
                    for j in range(seg_count)
                ]
                base4 = base3 + seg_count * 2
                range_offs = [
                    struct.unpack(">H", cmap[base4 + j * 2 : base4 + j * 2 + 2])[0]
                    for j in range(seg_count)
                ]
                glyph_arr_base = base4 + seg_count * 2
                for j in range(seg_count):
                    if starts[j] == 0xFFFF:
                        break
                    for c in range(starts[j], ends[j] + 1):
                        if range_offs[j] == 0:
                            gid = (c + deltas[j]) & 0xFFFF
                        else:
                            idx = range_offs[j] // 2 + (c - starts[j]) + j - seg_count
                            off = glyph_arr_base + idx * 2
                            if off + 2 > len(cmap):
                                raise ValueError("invalid TrueType cmap")
                            gid = struct.unpack(">H", cmap[off : off + 2])[0]
                            if gid and deltas[j]:
                                gid = (gid + deltas[j]) & 0xFFFF
                        if gid:
                            cp_to_gid[c] = gid
            elif fmt == 6:
                if sub_off + 10 > len(cmap):
                    raise ValueError("invalid TrueType cmap")
                first_code = struct.unpack(">H", cmap[sub_off + 6 : sub_off + 8])[0]
                entry_count = struct.unpack(">H", cmap[sub_off + 8 : sub_off + 10])[0]
                if sub_off + 10 + entry_count * 2 > len(cmap):
                    raise ValueError("invalid TrueType cmap")
                for j in range(entry_count):
                    gid = struct.unpack(">H", cmap[sub_off + 10 + j * 2 : sub_off + 12 + j * 2])[0]
                    if gid:
                        cp_to_gid[first_code + j] = gid
        return cp_to_gid
    except struct.error, IndexError, KeyError:
        raise ValueError("invalid TrueType cmap")


def tt_gid_composite_info(
    data: bytes, glyf_off: int, loca: list[int], gid: int
) -> tuple[tuple[int, int, int, int] | None, bool]:
    """Return (body_bbox, has_dot) if gid is a composite glyph with a dot-above component."""
    try:
        start = loca[gid]
        end = loca[gid + 1]
        if start >= end or glyf_off + start + 2 > len(data):
            return (None, False)
        if struct.unpack(">h", data[glyf_off + start : glyf_off + start + 2])[0] >= 0:
            return (None, False)
        FLAGS_MORE = 0x0020
        pos = glyf_off + start + 10
        comp_gids: list[int] = []
        while pos + 4 <= len(data):
            flags, cgid = struct.unpack(">HH", data[pos : pos + 4])
            comp_gids.append(cgid)
            pos += 4 + (4 if flags & 0x0001 else 2)
            if flags & 0x0008:
                pos += 2
            elif flags & 0x0040:
                pos += 4
            elif flags & 0x0080:
                pos += 8
            if not (flags & FLAGS_MORE):
                break
        body_bbox: tuple[int, int, int, int] | None = None
        has_dot = False
        for cgid in comp_gids:
            bbox = tt_glyph_bbox(data, glyf_off, loca, cgid)
            if bbox is None:
                continue
            xmin, ymin, xmax, ymax = bbox
            w, h = xmax - xmin, ymax - ymin
            if h > 0 and h < 600 and 0.4 < w / h < 2.5 and ymin > 900:
                has_dot = True
            else:
                body_bbox = bbox
        return (body_bbox, has_dot)
    except struct.error, IndexError, KeyError:
        return (None, False)
