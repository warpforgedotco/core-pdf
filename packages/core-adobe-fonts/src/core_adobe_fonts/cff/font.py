# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from math import isfinite
from typing import NamedTuple

from core_adobe_fonts._vendor.font_data.cff_tables import (
    CFF_EXPERT_STRINGS,
    CFF_EXPERT_SUBSET_STRINGS,
    CFF_ISO_ADOBE_STRINGS,
    CFF_STANDARD_STRINGS,
)
from core_adobe_fonts._vendor.font_data.encoding_names import STANDARD_ENCODING_GLYPH_NAMES


class CffFontMatrix(NamedTuple):
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def multiply(self, right: CffFontMatrix) -> CffFontMatrix:
        return CffFontMatrix(
            self.a * right.a + self.b * right.c,
            self.a * right.b + self.b * right.d,
            self.c * right.a + self.d * right.c,
            self.c * right.b + self.d * right.d,
            self.e * right.a + self.f * right.c + right.e,
            self.e * right.b + self.f * right.d + right.f,
        )


STANDARD_GLYPH_SIDS = {name: sid for sid, name in enumerate(CFF_STANDARD_STRINGS)}


CFF_STANDARD_STRING_COUNT = len(CFF_STANDARD_STRINGS)


DEFAULT_CFF_FONT_MATRIX = CffFontMatrix(0.001, 0.0, 0.0, 0.001, 0.0, 0.0)


CFF_EXPERT_ENCODING_CODES = tuple(
    code
    for code in (*range(32, 127), *range(161, 256))
    if code
    not in {
        35,
        64,
        70,
        71,
        72,
        74,
        75,
        80,
        81,
        85,
        92,
        164,
        165,
        171,
        173,
        174,
        176,
        177,
        180,
        181,
        185,
        186,
        187,
        198,
        199,
    }
)


def cff_offset(values: list[float] | None, context: str) -> int:
    if not values or len(values) != 1:
        raise ValueError(f"invalid CFF {context}")
    value = values[0]
    if not isfinite(value) or value < 0 or int(value) != value:
        raise ValueError(f"invalid CFF {context}")
    return int(value)


def cff_font_matrix(
    font_dict: dict[int | tuple[int, int], list[float]],
) -> CffFontMatrix | None:
    values = font_dict.get((12, 7))
    if values is None:
        return None
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError("invalid CFF FontMatrix")
    numbers: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("invalid CFF FontMatrix")
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError("invalid CFF FontMatrix") from exc
        if not isfinite(number):
            raise ValueError("invalid CFF FontMatrix")
        numbers.append(number)
    return CffFontMatrix(*numbers)


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
        "font_dicts",
    )

    def __init__(self, data: bytes | memoryview | None) -> None:
        if data is None:
            raise ValueError("missing CFF font program")
        self.data = data
        pos = self.read_header()
        ignored_names, pos = self.read_index(pos)
        top_index, pos = self.read_index(pos)
        custom_strings, pos = self.read_index(pos)
        self.custom_string_sids = {
            value.decode("latin-1"): CFF_STANDARD_STRING_COUNT + index
            for index, value in enumerate(custom_strings)
        }
        global_subrs, pos = self.read_index(pos)
        self.global_subrs = tuple(global_subrs)
        if not top_index:
            raise ValueError("invalid CFF top dict")
        self.top_dict = self.parse_dict(top_index[0])
        self.is_cid_keyed = (12, 30) in self.top_dict
        self.charstrings, ignored_pos = self.read_index(self.dict_offset(17))
        self.cid_to_gid = self.read_charset(self.dict_offset(15, default=0), len(self.charstrings))
        self.fd_select = self.read_fd_select()
        self.font_dicts = self.read_font_dicts()
        self.local_subrs = self.read_local_subrs()

    def read_header(self) -> int:
        if len(self.data) < 4 or self.data[0] != 1:
            raise ValueError("invalid CFF font program")
        size = self.data[2]
        if not 4 <= size <= len(self.data) or not 1 <= self.data[3] <= 4:
            raise ValueError("invalid CFF header")
        return size

    def dict_offset(self, operator: int, *, default: int | None = None) -> int:
        values = self.top_dict.get(operator)
        if values is None and default is not None:
            return default
        return cff_offset(values, "dictionary offset")

    def read_index(self, pos: int) -> tuple[list[bytes], int]:
        data = memoryview(self.data)
        if pos < 0 or pos + 2 > len(data):
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
    def parse_number(item: bytes, pos: int, *, dict_number: bool = False) -> tuple[float, int]:
        if not 0 <= pos < len(item):
            raise ValueError("invalid CFF number offset")
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
            return CFFFont.parse_real_number(item, pos + 1)
        if b0 == 255 and not dict_number:
            if pos + 5 > len(item):
                raise ValueError("invalid Type 2 number")
            return (
                int.from_bytes(item[pos + 1 : pos + 5], "big", signed=True) / 65536.0,
                pos + 5,
            )
        raise ValueError("invalid CFF number")

    @staticmethod
    def parse_real_number(item: bytes, pos: int) -> tuple[float, int]:
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
                elif nibble == 13:
                    raise ValueError("invalid CFF real number")
                elif nibble == 14:
                    parts.append("-")
        raise ValueError("invalid CFF real number")

    def parse_dict(self, item: bytes) -> dict[int | tuple[int, int], list[float]]:
        result, trailing = self.read_dict_entries(item)
        if trailing:
            raise ValueError("unterminated CFF dictionary operands")
        return result

    def read_dict_entries(
        self, item: bytes
    ) -> tuple[dict[int | tuple[int, int], list[float]], list[float]]:
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
                value, pos = self.parse_number(item, pos, dict_number=True)
                stack.append(value)
        return result, stack

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

    def read_local_subrs(self) -> tuple[tuple[bytes, ...], ...]:
        if self.is_cid_keyed:
            return tuple(tuple(self.read_private_subrs(font_dict)) for font_dict in self.font_dicts)
        return (tuple(self.read_private_subrs(self.top_dict)),)

    def read_charset(self, pos: int, glyph_count: int) -> dict[int, int]:
        if glyph_count < 1:
            raise ValueError("CFF charset has no .notdef glyph")
        if pos in {0, 1, 2}:
            if self.is_cid_keyed:
                raise ValueError("CID-keyed CFF font uses a predefined charset")
            names = (CFF_ISO_ADOBE_STRINGS, CFF_EXPERT_STRINGS, CFF_EXPERT_SUBSET_STRINGS)[pos]
            if glyph_count > len(names):
                raise ValueError("CFF predefined charset is too short")
            return {STANDARD_GLYPH_SIDS[name]: gid for gid, name in enumerate(names[:glyph_count])}
        if glyph_count == 1:
            return {0: 0}
        if not 0 <= pos < len(self.data):
            raise ValueError("invalid CFF charset offset")
        fmt = self.data[pos]
        pos += 1
        if fmt not in {0, 1, 2}:
            raise ValueError("invalid CFF charset format")
        mapping = {0: 0}
        gid = 1
        while gid < glyph_count:
            size = 2 if fmt == 0 else 3 if fmt == 1 else 4
            if pos + size > len(self.data):
                raise ValueError("truncated CFF charset")
            first = int.from_bytes(self.data[pos : pos + 2], "big")
            count = 1 if fmt == 0 else 1 + int.from_bytes(self.data[pos + 2 : pos + size], "big")
            if gid + count > glyph_count:
                raise ValueError("CFF charset range exceeds glyph count")
            for offset in range(count):
                if first + offset in mapping:
                    raise ValueError("duplicate CFF charset entry")
                mapping[first + offset] = gid + offset
            gid += count
            pos += size
        return mapping

    def read_encoding_codes(self, pos: int) -> dict[int, int]:
        if not 0 < pos < len(self.data):
            raise ValueError("invalid CFF encoding offset")
        raw_format = self.data[pos]
        fmt = raw_format & 127
        if fmt not in {0, 1} or pos + 1 >= len(self.data):
            raise ValueError("invalid CFF encoding")
        count = self.data[pos + 1]
        pos += 2
        codes: dict[int, int] = {}
        gid = 1
        for _ in range(count):
            size = 1 if fmt == 0 else 2
            if pos + size > len(self.data):
                raise ValueError("truncated CFF encoding")
            first = self.data[pos]
            length = 1 if fmt == 0 else self.data[pos + 1] + 1
            if first + length > 256 or gid + length > len(self.charstrings):
                raise ValueError("CFF encoding range exceeds font")
            for offset in range(length):
                if first + offset in codes:
                    raise ValueError("duplicate CFF encoding code")
                codes[first + offset] = gid + offset
            pos += size
            gid += length
        if raw_format & 128:
            if pos >= len(self.data):
                raise ValueError("truncated CFF encoding supplements")
            count = self.data[pos]
            pos += 1
            for _ in range(count):
                if pos + 3 > len(self.data):
                    raise ValueError("truncated CFF encoding supplement")
                code = self.data[pos]
                sid = int.from_bytes(self.data[pos + 1 : pos + 3], "big")
                if sid not in self.cid_to_gid:
                    raise ValueError("CFF encoding supplement references missing glyph")
                codes[code] = self.cid_to_gid[sid]
                pos += 3
        return codes

    def builtin_encoding(self) -> dict[int, str]:
        if self.is_cid_keyed:
            return {}
        offset = self.dict_offset(16, default=0)
        if offset == 0:
            return {
                code: name
                for code, name in enumerate(STANDARD_ENCODING_GLYPH_NAMES)
                if name != ".notdef"
            }
        if offset == 1:
            return {
                code: name
                for code, name in zip(
                    CFF_EXPERT_ENCODING_CODES, CFF_EXPERT_STRINGS[1:], strict=True
                )
                if STANDARD_GLYPH_SIDS[name] in self.cid_to_gid
            }
        reverse = {gid: sid for sid, gid in self.cid_to_gid.items()}
        names = dict(enumerate(CFF_STANDARD_STRINGS)) | {
            sid: name for name, sid in self.custom_string_sids.items()
        }
        return {code: names[reverse[gid]] for code, gid in self.read_encoding_codes(offset).items()}

    def read_fd_select(self) -> tuple[int, ...]:
        count = len(self.charstrings)
        if not self.is_cid_keyed:
            return (0,) * count
        values = self.top_dict.get((12, 37))
        pos = cff_offset(values, "FDSelect offset")
        if not 0 <= pos < len(self.data):
            raise ValueError("invalid CFF FDSelect offset")
        fmt = self.data[pos]
        pos += 1
        if fmt == 0 and pos + count <= len(self.data):
            return tuple(self.data[pos : pos + count])
        if fmt != 3 or pos + 2 > len(self.data):
            raise ValueError("invalid CFF FDSelect")
        ranges = int.from_bytes(self.data[pos : pos + 2], "big")
        pos += 2
        if not ranges or pos + ranges * 3 + 2 > len(self.data):
            raise ValueError("truncated CFF FDSelect")
        entries = [
            (
                int.from_bytes(self.data[pos + i * 3 : pos + i * 3 + 2], "big"),
                self.data[pos + i * 3 + 2],
            )
            for i in range(ranges)
        ]
        sentinel = int.from_bytes(self.data[pos + ranges * 3 : pos + ranges * 3 + 2], "big")
        if entries[0][0] != 0 or sentinel != count:
            raise ValueError("invalid CFF FDSelect bounds")
        selection: list[int] = []
        for index, (first, fd) in enumerate(entries):
            last = entries[index + 1][0] if index + 1 < len(entries) else sentinel
            if last <= first:
                raise ValueError("invalid CFF FDSelect range")
            selection.extend([fd] * (last - first))
        return tuple(selection)

    def read_font_dicts(self) -> tuple[dict[int | tuple[int, int], list[float]], ...]:
        if not self.is_cid_keyed:
            return ()
        values = self.top_dict.get((12, 36))
        items, _ = self.read_index(cff_offset(values, "FDArray offset"))
        return tuple(self.parse_dict(item) for item in items)

    def read_private_subrs(
        self, font_dict: dict[int | tuple[int, int], list[float]]
    ) -> list[bytes]:
        private = font_dict.get(18)
        if private is None:
            return []
        if len(private) != 2 or any(
            not isfinite(value) or int(value) != value for value in private
        ):
            raise ValueError("invalid CFF Private dictionary")
        size, offset = map(int, private)
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise ValueError("invalid CFF Private bounds")
        private_dict = self.parse_dict(bytes(self.data[offset : offset + size]))
        subrs = private_dict.get(19)
        if subrs is None:
            return []
        items, _ = self.read_index(offset + cff_offset(subrs, "Subrs offset"))
        return items

    def local_subrs_for_glyph(self, glyph_id: int) -> tuple[bytes, ...]:
        if not 0 <= glyph_id < len(self.charstrings):
            raise ValueError("invalid CFF glyph id")
        fd = self.fd_select[glyph_id]
        if not 0 <= fd < len(self.local_subrs):
            raise ValueError("invalid CFF font dictionary index")
        return self.local_subrs[fd]

    def font_matrix(self, glyph_id: int) -> CffFontMatrix:
        if not self.has_glyph_id(glyph_id) or glyph_id >= len(self.fd_select):
            raise ValueError("invalid CFF glyph id")
        top = cff_font_matrix(self.top_dict)
        fd = self.fd_select[glyph_id]
        if self.font_dicts and not 0 <= fd < len(self.font_dicts):
            raise ValueError("invalid CFF font dictionary index")
        child = cff_font_matrix(self.font_dicts[fd]) if self.font_dicts else None
        if top is None:
            return child or DEFAULT_CFF_FONT_MATRIX
        return top if child is None else child.multiply(top)


__all__ = (
    "CffFontMatrix",
    "STANDARD_GLYPH_SIDS",
    "CFF_STANDARD_STRING_COUNT",
    "DEFAULT_CFF_FONT_MATRIX",
    "CFF_EXPERT_ENCODING_CODES",
    "cff_font_matrix",
    "CFFFont",
)
