from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

from core_adobe_fonts.cff import font
from core_adobe_fonts.cff.font import (
    CFF_EXPERT_ENCODING_CODES,
    CFF_EXPERT_STRINGS,
    CFF_EXPERT_SUBSET_STRINGS,
    CFF_ISO_ADOBE_STRINGS,
    CFF_STANDARD_STRING_COUNT,
    DEFAULT_CFF_FONT_MATRIX,
    STANDARD_GLYPH_SIDS,
    CffFontMatrix,
)
from core_adobe_fonts.encodings import STANDARD_ENCODING_GLYPH_NAMES
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.fonts_cff_repair import (
    EMPTY_FEATURE,
    CFFGlyphFeature,
    feature_from_contours,
)
from core_pdf.impl.fonts_program_base import GlyphNaming, GlyphProgram
from core_pdf.impl.fonts_raster_kernel import (
    Contours,
    transform_contours,
)
from core_pdf.impl.geometry import points_bbox, union_bbox
from core_pdf.impl.types import Rectangle
from core_pdf_cythonized import type2_glyph_geometry
from core_pdf_spec.s_08_graphics.matrix import Matrix

MALFORMED_CFF_TABLE = (IndexError, OverflowError, TypeError, ValueError)


def with_recovery[T](strict: Callable[..., T], repair: Callable[..., T], /, *args: Any) -> T:
    try:
        return strict(*args)
    except MALFORMED_CFF_TABLE:
        return repair(*args)


assert len(STANDARD_GLYPH_SIDS) == CFF_STANDARD_STRING_COUNT


def cff_font_matrix(
    font_dict: dict[int | tuple[int, int], list[float]],
) -> CffFontMatrix | None:
    try:
        return font.cff_font_matrix(font_dict)
    except TypeError, ValueError:
        return None


class CFFFont(font.CFFFont, GlyphProgram):
    __slots__ = ("source_size",)

    def __init__(self, data: bytes | memoryview | None, *, source_size: int | None = None) -> None:
        super().__init__(data)
        self.source_size = len(self.data) if source_size is None else source_size

    def read_header(self) -> int:
        if len(self.data) < 4 or self.data[0] != 1:
            raise ValueError("invalid CFF font program")
        return self.data[2]

    def dict_offset(self, operator: int, *, default: int | None = None) -> int:
        value = self.top_dict.get(operator, [default])[0]
        if not isinstance(value, (int, float)):
            if default is not None:
                return default
            raise ValueError("invalid CFF CharStrings offset")
        return int(value)

    def parse_dict(self, item: bytes) -> dict[int | tuple[int, int], list[float]]:
        entries, ignored_trailing = self.read_dict_entries(item)
        return entries

    def read_charset(self, pos: int, glyph_count: int) -> dict[int, int]:
        return with_recovery(super().read_charset, self.recover_read_charset, pos, glyph_count)

    def recover_read_charset(self, pos: int, glyph_count: int) -> dict[int, int]:
        if glyph_count <= 0:
            return {}
        if self.is_cid_keyed and pos in {0, 1, 2}:
            raise ValueError("CID-keyed CFF font uses a predefined charset")
        if glyph_count == 1:
            return {0: 0}
        glyph_names: tuple[str, ...] | None
        match pos:
            case 0:
                glyph_names = CFF_ISO_ADOBE_STRINGS
            case 1:
                glyph_names = CFF_EXPERT_STRINGS
            case 2:
                glyph_names = CFF_EXPERT_SUBSET_STRINGS
            case _:
                glyph_names = None
        if glyph_names is not None:
            return {
                STANDARD_GLYPH_SIDS[name]: gid for gid, name in enumerate(glyph_names[:glyph_count])
            }
        data = self.data
        if pos >= len(data):
            return {gid: gid for gid in range(glyph_count)}
        fmt = data[pos]
        pos += 1
        cid_to_gid = {0: 0}
        gid = 1
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
        return cid_to_gid

    def read_encoding_codes(self, pos: int) -> dict[int, int]:
        return with_recovery(super().read_encoding_codes, self.recover_read_encoding_codes, pos)

    def recover_read_encoding_codes(self, pos: int) -> dict[int, int]:
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
                supplement_gid = self.cid_to_gid.get(sid)
                if supplement_gid is not None and supplement_gid < glyph_count:
                    codes[code] = supplement_gid
        return codes

    def builtin_encoding(self) -> dict[int, str]:
        if self.is_cid_keyed:
            return {}
        operand = self.top_dict.get(16, [0])[0]
        if not isinstance(operand, (int, float)):
            return {}
        offset = int(operand)
        if offset == 0:
            return {}
        if offset == 1:
            sid_to_gid = self.cid_to_gid
            return {
                code: name
                for code, name in zip(
                    CFF_EXPERT_ENCODING_CODES,
                    CFF_EXPERT_STRINGS[1:],
                    strict=True,
                )
                if STANDARD_GLYPH_SIDS[name] in sid_to_gid
            }
        sid_to_name = {sid: name for name, sid in STANDARD_GLYPH_SIDS.items()}
        sid_to_name.update({sid: name for name, sid in self.custom_string_sids.items()})
        gid_to_name = {
            gid: sid_to_name[sid] for sid, gid in self.cid_to_gid.items() if sid in sid_to_name
        }
        encoding: dict[int, str] = {}
        for code, gid in self.read_encoding_codes(offset).items():
            name = gid_to_name.get(gid)
            if name is not None and name != ".notdef":
                encoding[code] = name
        return encoding

    def builtin_encoding_is_authoritative(self) -> bool:
        if self.is_cid_keyed:
            return False
        operand = self.top_dict.get(16, [0])[0]
        if not isinstance(operand, (int, float)):
            return False
        return int(operand) != 0

    def read_fd_select(self) -> tuple[int, ...]:
        if not self.is_cid_keyed and (12, 37) in self.top_dict:
            return self.recover_read_fd_select()
        return with_recovery(super().read_fd_select, self.recover_read_fd_select)

    def recover_read_fd_select(self) -> tuple[int, ...]:
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
        return tuple(fd_select)

    def read_font_dicts(
        self,
    ) -> tuple[dict[int | tuple[int, int], list[float]], ...]:
        return with_recovery(super().read_font_dicts, self.recover_read_font_dicts)

    def recover_read_font_dicts(
        self,
    ) -> tuple[dict[int | tuple[int, int], list[float]], ...]:
        if not self.is_cid_keyed:
            return ()
        fdarray_off = self.top_dict.get((12, 36), [None])[0]
        if (
            not isinstance(fdarray_off, (int, float))
            or not isfinite(fdarray_off)
            or fdarray_off < 0
            or fdarray_off != int(fdarray_off)
        ):
            return ()
        try:
            raw_font_dicts, ignored_pos = self.read_index(int(fdarray_off))
        except ValueError:
            return ()

        font_dicts: list[dict[int | tuple[int, int], list[float]]] = []
        for raw_font_dict in raw_font_dicts:
            try:
                font_dicts.append(self.parse_dict(raw_font_dict))
            except IndexError, ValueError:
                font_dicts.append({})
        return tuple(font_dicts)

    def read_private_subrs(
        self, font_dict: dict[int | tuple[int, int], list[float]]
    ) -> list[bytes]:
        return with_recovery(super().read_private_subrs, self.recover_read_private_subrs, font_dict)

    def recover_read_private_subrs(
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
        private_dict = self.parse_dict(bytes(self.data[private_off : private_off + private_size]))
        subrs_off = private_dict.get(19, [None])[0]
        if not isinstance(subrs_off, (int, float)):
            return []
        try:
            subrs, ignored_pos = self.read_index(private_off + int(subrs_off))
        except ValueError:
            return []
        return subrs

    def local_subrs_for_glyph(self, glyph_id: int) -> tuple[bytes, ...]:
        return with_recovery(
            super().local_subrs_for_glyph, self.recover_local_subrs_for_glyph, glyph_id
        )

    def recover_local_subrs_for_glyph(self, glyph_id: int) -> tuple[bytes, ...]:
        fd_index = self.fd_select[glyph_id] if 0 <= glyph_id < len(self.fd_select) else 0
        if 0 <= fd_index < len(self.local_subrs):
            return self.local_subrs[fd_index]
        return ()

    def font_matrix(self, glyph_id: int) -> CffFontMatrix:
        return with_recovery(super().font_matrix, self.recover_font_matrix, glyph_id)

    def recover_font_matrix(self, glyph_id: int) -> CffFontMatrix:
        fd_index = self.fd_select[glyph_id] if 0 <= glyph_id < len(self.fd_select) else 0
        top_matrix = cff_font_matrix(self.top_dict)
        font_dict = self.font_dicts[fd_index] if 0 <= fd_index < len(self.font_dicts) else None
        font_dict_matrix = cff_font_matrix(font_dict) if font_dict is not None else None
        if top_matrix is None:
            return font_dict_matrix or DEFAULT_CFF_FONT_MATRIX
        if font_dict_matrix is None:
            return top_matrix
        return font_dict_matrix.multiply(top_matrix)

    def seac_contours(
        self,
        base_code: int,
        accent_code: int,
        accent_dx: float,
        accent_dy: float,
    ) -> tuple[tuple[tuple[float, float], ...], ...]:
        if self.is_cid_keyed:
            return ()
        contours: list[tuple[tuple[float, float], ...]] = []
        for code, offset_x, offset_y in (
            (base_code, 0.0, 0.0),
            (accent_code, accent_dx, accent_dy),
        ):
            if not 0 <= code < len(STANDARD_ENCODING_GLYPH_NAMES):
                continue
            sid = STANDARD_GLYPH_SIDS.get(STANDARD_ENCODING_GLYPH_NAMES[code])
            glyph_id = self.cid_to_gid.get(sid) if sid is not None else None
            if glyph_id is None or not 0 <= glyph_id < len(self.charstrings):
                continue
            component_contours, ignored_bbox = type2_glyph_geometry_impl(
                self.charstrings[glyph_id],
                local_subrs=self.local_subrs_for_glyph(glyph_id),
                global_subrs=self.global_subrs,
            )
            contours.extend(
                tuple((x + offset_x, y + offset_y) for x, y in contour)
                for contour in component_contours
            )
        return tuple(contours)

    def glyph_geometry_for_gid(
        self, glyph_id: int, *, bounds_only: bool = False
    ) -> tuple[
        tuple[tuple[tuple[float, float], ...], ...],
        tuple[float, float, float, float] | None,
    ]:
        try:
            charstring = self.charstrings[glyph_id]
        except IndexError:
            return ((), None)
        matrix = Matrix(*self.font_matrix(glyph_id))
        contours, raw_bbox = type2_glyph_geometry_impl(
            charstring,
            local_subrs=self.local_subrs_for_glyph(glyph_id),
            global_subrs=self.global_subrs,
            seac_resolver=self.seac_contours,
            flatten=not bounds_only or matrix[1] != 0.0 or matrix[2] != 0.0,
            retain_contours=not bounds_only or matrix != DEFAULT_CFF_FONT_MATRIX,
        )
        if matrix == DEFAULT_CFF_FONT_MATRIX:
            return (tuple(tuple(contour) for contour in contours), raw_bbox)
        normalized = transform_contours(contours, matrix)
        return (normalized, contours_bbox(normalized))

    def glyph_feature(self, glyph_id: int) -> CFFGlyphFeature:
        geometry = self.glyph_geometry_for_gid(glyph_id)
        contours = geometry[0]
        if not contours:
            return EMPTY_FEATURE
        return feature_from_contours(contours)

    def glyph_bbox_for_gid(self, glyph_id: int) -> Rectangle | None:
        geometry = self.glyph_geometry_for_gid(glyph_id, bounds_only=True)
        return geometry[1]

    def normalized_glyph_contours(self, glyph_id: int) -> Contours:
        return self.glyph_geometry_for_gid(glyph_id)[0]

    def glyph_id_for_code(self, code: int, naming: GlyphNaming) -> int | None:
        if naming.is_cid_font:
            return self.glyph_id_for_cid(code)
        return self.glyph_id_for_name(naming.glyph_name(code))

    def font_builtin_encoding(self) -> tuple[dict[int, str], bool] | None:
        try:
            return (self.builtin_encoding(), self.builtin_encoding_is_authoritative())
        except PdfParseError, ValueError:
            return {}, False


def contours_bbox(
    contours: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[float, float, float, float] | None:
    return points_bbox(point for contour in contours for point in contour)


def type2_glyph_geometry_impl(
    charstring: bytes,
    *,
    local_subrs: tuple[bytes, ...],
    global_subrs: tuple[bytes, ...],
    seac_resolver: (
        Callable[
            [int, int, float, float],
            tuple[tuple[tuple[float, float], ...], ...],
        ]
        | None
    ) = None,
    flatten: bool = True,
    retain_contours: bool = True,
) -> tuple[list[list[tuple[float, float]]], tuple[float, float, float, float] | None]:
    contours, bbox, seac, _valid = type2_glyph_geometry(
        charstring, local_subrs, global_subrs, flatten, retain_contours
    )
    if seac is None or seac_resolver is None:
        return contours, bbox

    base_code, accent_code, dx, dy = seac
    for component in seac_resolver(base_code, accent_code, dx, dy):
        if not component:
            continue
        if retain_contours:
            contours.append(list(component))
        bbox = union_bbox(bbox, points_bbox(component))
    return contours, bbox
