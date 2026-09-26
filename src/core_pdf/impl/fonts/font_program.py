from __future__ import annotations

import logging
import re
import struct
import threading
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from io import BytesIO
from math import inf, isfinite
from typing import Any, ClassVar, TypeAlias

import numpy

from core_adobe_fonts.cff.font import (
    CFF_EXPERT_ENCODING_CODES,
    CFF_STANDARD_STRING_COUNT,
    DEFAULT_CFF_FONT_MATRIX,
    STANDARD_GLYPH_SIDS,
    CffFontMatrix,
)
from core_adobe_fonts.cff.font import CFFFont as PdfCFFFont
from core_adobe_fonts.cff.font import (
    cff_font_matrix as pdf_cff_font_matrix,
)
from core_adobe_fonts.type1.program import (
    binary_entries,
    decode_charstring,
    decode_eexec_payload,
)
from core_adobe_fonts.type1.program import parse_type1_font_program_encoding as parse_encoding
from core_pdf._vendor.fontTools.cffLib import (
    cffExpertSubsetStrings,
    cffIExpertStrings,
    cffISOAdobeStrings,
)
from core_pdf._vendor.fontTools.encodings.StandardEncoding import StandardEncoding
from core_pdf._vendor.fontTools.misc.psCharStrings import T1CharString
from core_pdf._vendor.fontTools.pens.boundsPen import BoundsPen
from core_pdf._vendor.fontTools.pens.recordingPen import (
    DecomposingRecordingPen,
    RecordingPen,
)
from core_pdf._vendor.fontTools.pens.transformPen import TransformPen
from core_pdf._vendor.fontTools.ttLib import TTFont
from core_pdf.impl.fonts.raster_kernel import (
    Point,
    rasterize_contours,
    scale_contours,
    transform_contours,
)
from core_pdf.impl.geometry import points_bbox, transform_bbox
from core_pdf.impl.types import FrozenFields, ReplaceFields, ReprFields, frozen_setattr
from core_pdf_cythonized import decrypt_type1, truetype_contours, type2_glyph_geometry
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.s_09_fonts.font_program_truetype import (
    is_unicode_scalar,
    symbol_character_code,
)

MALFORMED_CFF_TABLE = (IndexError, OverflowError, TypeError, ValueError)


def with_recovery[T](strict: Callable[..., T], repair: Callable[..., T], /, *args: Any) -> T:
    """Read a CFF table strictly; rebuild it from the raw bytes if that fails."""
    try:
        return strict(*args)
    except MALFORMED_CFF_TABLE:
        return repair(*args)


class CFFGlyphFeature(FrozenFields, ReplaceFields, ReprFields):
    cells: tuple[tuple[int, int], ...]
    aspect: float
    contours: int
    bitmap: tuple[int, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("cells", "aspect", "contours", "bitmap")
    __match_args__ = ("cells", "aspect", "contours", "bitmap")

    def __init__(
        self,
        cells: tuple[tuple[int, int], ...],
        aspect: float,
        contours: int,
        bitmap: tuple[int, ...] = (),
    ) -> None:
        frozen_setattr(self, "cells", cells)
        frozen_setattr(self, "aspect", aspect)
        frozen_setattr(self, "contours", contours)
        frozen_setattr(self, "bitmap", bitmap)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.cells == other.cells
            and self.aspect == other.aspect
            and self.contours == other.contours
            and self.bitmap == other.bitmap
        )

    def __hash__(self) -> int:
        return hash((self.cells, self.aspect, self.contours, self.bitmap))


EMPTY_FEATURE = CFFGlyphFeature((), 0.0, 0, ())
assert len(STANDARD_GLYPH_SIDS) == CFF_STANDARD_STRING_COUNT


def cff_font_matrix(
    font_dict: dict[int | tuple[int, int], list[float]],
) -> CffFontMatrix | None:
    try:
        return pdf_cff_font_matrix(font_dict)
    except TypeError, ValueError:
        return None


class CFFFont(PdfCFFFont):
    __slots__ = ()

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
        glyph_names: list[str] | None
        match pos:
            case 0:
                glyph_names = cffISOAdobeStrings
            case 1:
                glyph_names = cffIExpertStrings
            case 2:
                glyph_names = cffExpertSubsetStrings
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
                    cffIExpertStrings[1:],
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
            if not 0 <= code < len(StandardEncoding):
                continue
            sid = STANDARD_GLYPH_SIDS.get(StandardEncoding[code])
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

    def glyph_bitmap_for_gid(
        self, glyph_id: int, *, width: int = 24, height: int = 32
    ) -> tuple[int, ...]:
        geometry = self.glyph_geometry_for_gid(glyph_id)
        contours = geometry[0]
        if not contours:
            return ()
        return rasterize_contours(contours, width=width, height=height)

    def glyph_bbox_for_gid(self, glyph_id: int) -> tuple[float, float, float, float] | None:
        geometry = self.glyph_geometry_for_gid(glyph_id, bounds_only=True)
        return geometry[1]

    def normalized_glyph_contours(
        self, glyph_id: int
    ) -> tuple[tuple[tuple[float, float], ...], ...]:
        return self.glyph_geometry_for_gid(glyph_id)[0]


def contours_bbox(
    contours: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[float, float, float, float] | None:
    return points_bbox(point for contour in contours for point in contour)


def feature_from_contours(
    contours: tuple[tuple[tuple[float, float], ...], ...] | list[list[tuple[float, float]]],
) -> CFFGlyphFeature:
    if not contours:
        return EMPTY_FEATURE

    points = [point for contour in contours for point in contour]
    if not points:
        return EMPTY_FEATURE
    xs, ys = zip(*points, strict=True)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    cells: set[tuple[int, int]] = set()
    add_cell = cells.add
    for px, py in points:
        cell_x = round((px - min_x) / width * 17)
        cell_y = round((py - min_y) / height * 23)
        add_cell((cell_x, cell_y))
    bitmap = rasterize_contours(contours, width=18, height=24)
    return CFFGlyphFeature(tuple(sorted(cells)), round(width / height, 2), len(contours), bitmap)


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
    """Interpret a Type 2 charstring into contours and a bounding box.

    The interpretation and the pen that used to drive it both live in the
    compiled kernel now; what stays here is the one thing the kernel cannot
    finish, because it needs the font's charset rather than the charstring: a
    composite glyph's components. The kernel reports the request and the
    components are run back through this same function.
    """
    # `valid` is deliberately ignored: a malformed charstring still yields
    # whatever it completed, exactly as the raising interpreter plus its
    # swallowing caller did. The conformance tests are what read it.
    contours, bbox, seac, _valid = type2_glyph_geometry(
        charstring, local_subrs, global_subrs, flatten, retain_contours
    )
    if seac is None or seac_resolver is None:
        return contours, bbox

    base_code, accent_code, dx, dy = seac
    min_x, min_y, max_x, max_y = bbox if bbox is not None else (inf, inf, -inf, -inf)
    has_points = bbox is not None
    for component in seac_resolver(base_code, accent_code, dx, dy):
        if not component:
            continue
        if retain_contours:
            contours.append(list(component))
        min_x = min(min_x, *(point[0] for point in component))
        min_y = min(min_y, *(point[1] for point in component))
        max_x = max(max_x, *(point[0] for point in component))
        max_y = max(max_y, *(point[1] for point in component))
        has_points = True
    return contours, (min_x, min_y, max_x, max_y) if has_points else None


def glyph_feature_distance(left: CFFGlyphFeature, right: CFFGlyphFeature) -> float:
    return feature_distance(
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
LEGITIMATE_MULTI_CHAR_GLYPHS = frozenset({"ff", "fi", "fl", "ffi", "ffl", "st"})


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


def repair_candidate(
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


class CFFUnicodeRepairIndex:
    __slots__ = (
        "resolve_candidate_gids",
        "code_to_gid_map",
        "make_font",
        "label_names",
        "repairable_gids",
        "feature_cache",
        "candidate_arrays_cache",
    )

    def __init__(
        self,
        font: CFFFont,
        mapping_items: tuple[tuple[bytes, int, str], ...],
    ) -> None:
        glyph_count = len(font.charstrings)
        labels: dict[int, str] = {}
        code_to_gid: dict[bytes, int] = {}
        if glyph_count >= 2:
            for code_bytes, cid, value in mapping_items:
                gid = font.glyph_id_for_cid(cid)
                if gid >= glyph_count:
                    continue
                labels[gid] = value
                code_to_gid[code_bytes] = gid

        self.make_font = font
        # A decoder asks again for every string it decodes, and each request
        # compares against the same candidate glyphs. Features depend only on
        # the glyph's outline, so they are computed once per glyph, and the
        # candidates' arrays once per index.
        self.feature_cache: dict[int, CFFGlyphFeature] = {}
        self.candidate_arrays_cache: FeatureArrays | None = None
        self.label_names = labels
        self.code_to_gid_map = code_to_gid
        self.repairable_gids = frozenset(
            gid for gid, label in labels.items() if is_repairable_to_unicode_label(label)
        )
        self.resolve_candidate_gids = tuple(
            gid
            for gid, label in labels.items()
            if len(label) == 1 and (label.isalnum() or label in ".-+")
        )

    def repairs_for_codes(self, codes: Iterable[bytes]) -> dict[bytes, str]:
        requested_codes = tuple(dict.fromkeys(codes))
        if not requested_codes or not self.repairable_gids:
            return {}
        target_gids = tuple(
            dict.fromkeys(
                gid
                for code in requested_codes
                if (gid := self.code_to_gid_map.get(code)) in self.repairable_gids
            )
        )
        if not target_gids:
            return {}
        repairs = self.repairs_for_gids(target_gids)
        return {
            code: replacement
            for code in requested_codes
            if (gid := self.code_to_gid_map.get(code)) is not None
            and (replacement := repairs.get(gid)) is not None
        }

    def repairs_for_gids(self, requested_gids: tuple[int, ...]) -> dict[int, str]:
        feature_cache = self.feature_cache
        glyph_feature = self.make_font.glyph_feature
        features: dict[int, CFFGlyphFeature] = {}
        for gid in dict.fromkeys((*self.resolve_candidate_gids, *requested_gids)):
            feature = feature_cache.get(gid)
            if feature is None:
                feature = feature_cache[gid] = glyph_feature(gid)
            features[gid] = feature

        candidate_gids = tuple(gid for gid in self.resolve_candidate_gids if features[gid].cells)
        target_gids = tuple(gid for gid in requested_gids if features[gid].cells)
        distance_lookups: dict[int, dict[int, float]] = {}
        if (
            target_gids
            and candidate_gids
            and (len(self.repairable_gids) * len(candidate_gids) >= 512)
        ):
            target_features = [features[gid] for gid in target_gids]
            candidate_features = [features[gid] for gid in candidate_gids]
            candidate_arrays = self.candidate_arrays_cache
            if candidate_arrays is None:
                candidate_arrays = self.candidate_arrays_cache = feature_arrays(
                    [feature.cells for feature in candidate_features],
                    [feature.bitmap for feature in candidate_features],
                    [feature.aspect for feature in candidate_features],
                    [feature.contours for feature in candidate_features],
                )
            distance_matrix = feature_distance_matrix(
                [feature.cells for feature in target_features],
                [feature.bitmap for feature in target_features],
                [feature.aspect for feature in target_features],
                [feature.contours for feature in target_features],
                [feature.cells for feature in candidate_features],
                [feature.bitmap for feature in candidate_features],
                [feature.aspect for feature in candidate_features],
                [feature.contours for feature in candidate_features],
                right_arrays=candidate_arrays,
            )
            distance_lookups = {
                target_gid: {
                    candidate_gid: float(distance_matrix[target_index, candidate_index])
                    for candidate_index, candidate_gid in enumerate(candidate_gids)
                }
                for target_index, target_gid in enumerate(target_gids)
            }

        repairs: dict[int, str] = {}
        for glyph_id in target_gids:
            label = self.label_names[glyph_id]
            replacement = repair_candidate(
                glyph_id,
                label,
                features,
                self.label_names,
                distance_lookups.get(glyph_id),
            )
            if replacement is not None and replacement != label:
                repairs[glyph_id] = replacement
        return repairs


FEATURE_GRID_WIDTH = 18
FEATURE_GRID_HEIGHT = 24
FeatureArrays: TypeAlias = tuple[
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
]


def cell_distance_map(cells: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    if not cells:
        return ()
    limit = FEATURE_GRID_WIDTH + FEATURE_GRID_HEIGHT
    distances = numpy.full((FEATURE_GRID_HEIGHT, FEATURE_GRID_WIDTH), limit, dtype=numpy.int64)
    for x, y in cells:
        if 0 <= x < FEATURE_GRID_WIDTH and 0 <= y < FEATURE_GRID_HEIGHT:
            distances[y, x] = 0
    # The two-pass 4-neighbour chamfer is the exact L1 distance transform, which is
    # separable: one forward and one backward running minimum per axis. `limit` exceeds
    # the largest achievable distance on this grid, so the all-unseeded case still
    # returns `limit` everywhere exactly as the sequential scan did.
    for axis, extent in ((1, FEATURE_GRID_WIDTH), (0, FEATURE_GRID_HEIGHT)):
        offsets = numpy.arange(extent, dtype=numpy.int64)
        if axis == 0:
            offsets = offsets[:, None]
        forward = numpy.minimum.accumulate(distances - offsets, axis=axis) + offsets
        reversed_slice: tuple[Any, ...] = (
            (slice(None, None, -1),)
            if axis == 0
            else (
                slice(None),
                slice(None, None, -1),
            )
        )
        backward = (
            numpy.minimum.accumulate((distances + offsets)[reversed_slice], axis=axis)[
                reversed_slice
            ]
            - offsets
        )
        distances = numpy.minimum(forward, backward)
    return tuple(distances.reshape(-1).tolist())


def average_nearest_distance(
    cells: tuple[tuple[int, int], ...], distance_map: tuple[int, ...]
) -> float:
    total = 0.0
    count = 0
    for x, y in cells:
        if 0 <= x < FEATURE_GRID_WIDTH and 0 <= y < FEATURE_GRID_HEIGHT:
            total += distance_map[y * FEATURE_GRID_WIDTH + x]
            count += 1
    return total / count if count else inf


def bitmap_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    intersection = 0
    union = 0
    for left_row, right_row in zip(left, right, strict=True):
        intersection += (left_row & right_row).bit_count()
        union += (left_row | right_row).bit_count()
    if union == 0:
        return 0.0
    return 1.0 - intersection / union


def feature_distance(
    left_cells: tuple[tuple[int, int], ...],
    left_bitmap: tuple[int, ...],
    left_aspect: float,
    left_contours: int,
    right_cells: tuple[tuple[int, int], ...],
    right_bitmap: tuple[int, ...],
    right_aspect: float,
    right_contours: int,
) -> float:
    if not left_cells or not right_cells:
        return inf
    left_map = cell_distance_map(left_cells)
    right_map = cell_distance_map(right_cells)
    return (
        average_nearest_distance(left_cells, right_map)
        + average_nearest_distance(right_cells, left_map)
        + bitmap_distance(left_bitmap, right_bitmap) * 0.75
        + abs(left_aspect - right_aspect) * 2.0
        + abs(left_contours - right_contours) * 0.2
    )


def feature_arrays(
    cells: Sequence[tuple[tuple[int, int], ...]],
    bitmaps: Sequence[tuple[int, ...]],
    aspects: Sequence[float],
    contours: Sequence[int],
) -> FeatureArrays:
    count = len(cells)
    masks = numpy.zeros((count, FEATURE_GRID_HEIGHT, FEATURE_GRID_WIDTH), dtype=numpy.float64)
    distance_maps = numpy.zeros_like(masks)
    valid_counts = numpy.zeros(count, dtype=numpy.float64)
    for index, feature_cells in enumerate(cells):
        valid_cells = tuple(
            (x, y)
            for x, y in feature_cells
            if 0 <= x < FEATURE_GRID_WIDTH and 0 <= y < FEATURE_GRID_HEIGHT
        )
        if not valid_cells:
            continue
        valid_counts[index] = len(valid_cells)
        for x, y in valid_cells:
            masks[index, y, x] += 1.0
        distance_maps[index] = numpy.asarray(
            cell_distance_map(feature_cells), dtype=numpy.float64
        ).reshape(FEATURE_GRID_HEIGHT, FEATURE_GRID_WIDTH)

    bitmap_width = max((len(bitmap) for bitmap in bitmaps), default=0)
    bitmap_rows = numpy.zeros((count, bitmap_width), dtype=numpy.uint64)
    for index, bitmap in enumerate(bitmaps):
        if bitmap:
            bitmap_rows[index, : len(bitmap)] = bitmap
    return (
        masks,
        distance_maps,
        valid_counts,
        numpy.asarray(aspects, dtype=numpy.float64),
        numpy.asarray(contours, dtype=numpy.float64),
        bitmap_rows,
    )


def feature_distance_matrix(
    left_cells: Sequence[tuple[tuple[int, int], ...]],
    left_bitmaps: Sequence[tuple[int, ...]],
    left_aspects: Sequence[float],
    left_contours: Sequence[int],
    right_cells: Sequence[tuple[tuple[int, int], ...]],
    right_bitmaps: Sequence[tuple[int, ...]],
    right_aspects: Sequence[float],
    right_contours: Sequence[int],
    *,
    right_arrays: FeatureArrays | None = None,
) -> numpy.ndarray[Any, Any]:
    (
        left_masks,
        left_maps,
        left_counts,
        left_aspects_array,
        left_contours_array,
        left_bitmap_rows,
    ) = feature_arrays(left_cells, left_bitmaps, left_aspects, left_contours)
    if right_arrays is None:
        right_arrays = feature_arrays(right_cells, right_bitmaps, right_aspects, right_contours)
    (
        right_masks,
        right_maps,
        right_counts,
        right_aspects_array,
        right_contours_array,
        right_bitmap_rows,
    ) = right_arrays

    distance = numpy.full((len(left_cells), len(right_cells)), numpy.inf, dtype=numpy.float64)
    valid = (left_counts[:, None] > 0) & (right_counts[None, :] > 0)
    if not numpy.any(valid):
        return distance

    left_to_right = numpy.einsum("lxy,rxy->lr", left_masks, right_maps)
    left_to_right = numpy.divide(
        left_to_right,
        left_counts[:, None],
        out=numpy.zeros_like(left_to_right),
        where=left_counts[:, None] > 0,
    )
    right_to_left = numpy.einsum("rxy,lxy->rl", right_masks, left_maps).T
    right_to_left = numpy.divide(
        right_to_left,
        right_counts[None, :],
        out=numpy.zeros_like(right_to_left),
        where=right_counts[None, :] > 0,
    )

    if left_bitmap_rows.shape[1] == 0 and right_bitmap_rows.shape[1] == 0:
        bitmap_distance = numpy.zeros_like(distance)
    else:
        bitmap_width = max(left_bitmap_rows.shape[1], right_bitmap_rows.shape[1])
        if left_bitmap_rows.shape[1] != bitmap_width:
            left_bitmap_rows = numpy.pad(
                left_bitmap_rows,
                ((0, 0), (0, bitmap_width - left_bitmap_rows.shape[1])),
            )
        if right_bitmap_rows.shape[1] != bitmap_width:
            right_bitmap_rows = numpy.pad(
                right_bitmap_rows,
                ((0, 0), (0, bitmap_width - right_bitmap_rows.shape[1])),
            )
        intersection = numpy.bitwise_count(
            left_bitmap_rows[:, None, :] & right_bitmap_rows[None, :, :]
        ).sum(axis=2)
        union = numpy.bitwise_count(
            left_bitmap_rows[:, None, :] | right_bitmap_rows[None, :, :]
        ).sum(axis=2)
        same_bitmap_shape = numpy.equal(
            numpy.asarray([len(bitmap) for bitmap in left_bitmaps])[:, None],
            numpy.asarray([len(bitmap) for bitmap in right_bitmaps])[None, :],
        )
        bitmap_ratio = numpy.divide(
            intersection,
            union,
            out=numpy.zeros_like(intersection, dtype=numpy.float64),
            where=union != 0,
        )
        bitmap_distance = numpy.where(
            same_bitmap_shape & (union != 0),
            1.0 - bitmap_ratio,
            0.0,
        )

    combined = (
        left_to_right
        + right_to_left
        + bitmap_distance * 0.75
        + numpy.abs(left_aspects_array[:, None] - right_aspects_array[None, :]) * 2.0
        + numpy.abs(left_contours_array[:, None] - right_contours_array[None, :]) * 0.2
    )
    distance[valid] = combined[valid]
    return distance


def parse_truetype_program(data: bytes) -> TTFont:
    font = TTFont(BytesIO(data), lazy=True)
    if not {"maxp", "glyf", "loca", "head"} <= set(font.keys()):
        raise ValueError("invalid TrueType glyph tables")
    return font


def glyph_set_of(font: Any) -> Any:
    """font.getGlyphSet(), built once per font and thread.

    Each call built a fresh glyph set -- reading fvar, hmtx and the glyf
    table's mapping -- to draw one glyph: 57 ms of a 240 ms profiled page
    render on PyMuPDF test_3357, 166 glyphs. The set is equivalent every
    time, but drawing tracks composite depth and variation location on it,
    so one is kept per thread. It lives on the font, whose lifetime it
    shares; a thread id reused after its thread ended takes over that set.
    """
    sets = font.__dict__.get("_core_pdf_glyph_sets")
    if sets is None:
        sets = font.__dict__["_core_pdf_glyph_sets"] = {}
    thread = threading.get_ident()
    glyph_set = sets.get(thread)
    if glyph_set is None:
        glyph_set = sets[thread] = font.getGlyphSet()
    return glyph_set


def fonttools_bbox(
    font: Any,
    glyph_id: int,
    scale: float,
) -> tuple[float, float, float, float] | None:
    glyph_name = font.getGlyphName(glyph_id)
    glyph_set = glyph_set_of(font)
    bounds_pen = BoundsPen(glyph_set)
    glyph_set[glyph_name].draw(TransformPen(bounds_pen, (scale, 0.0, 0.0, scale, 0.0, 0.0)))
    if bounds_pen.bounds is None:
        return None
    x_min, y_min, x_max, y_max = bounds_pen.bounds
    return float(x_min), float(y_min), float(x_max), float(y_max)


def glyph_bbox(glyf: Any, glyph_name: str) -> tuple[float, float, float, float] | None:
    glyph = glyf[glyph_name]
    if glyph.numberOfContours == 0:
        return None
    if not all(hasattr(glyph, attr) for attr in ("xMin", "yMin", "xMax", "yMax")):
        glyph.recalcBounds(glyf)
    return (
        float(glyph.xMin),
        float(glyph.yMin),
        float(glyph.xMax),
        float(glyph.yMax),
    )


GLYPH_HEADER = struct.Struct(">hhhhh")


def raw_glyph_locations(font: TTFont) -> tuple[Any, bytes]:
    try:
        locations = font["loca"]
        reader = font.reader
        glyph_data = bytes(reader["glyf"]) if reader is not None else b""
    except Exception:
        return (), b""
    return locations, glyph_data


def glyph_header_bbox(
    locations: Any, glyph_data: bytes, gid: int
) -> tuple[float, float, float, float] | None:
    if gid < 0 or gid + 1 >= len(locations):
        return None
    start = locations[gid]
    end = locations[gid + 1]
    if end - start < GLYPH_HEADER.size or end > len(glyph_data):
        return None
    contours, x_min, y_min, x_max, y_max = GLYPH_HEADER.unpack_from(glyph_data, start)
    if contours == 0:
        return None
    return (float(x_min), float(y_min), float(x_max), float(y_max))


def fonttools_contours(font: Any, glyph_id: int) -> tuple[tuple[Point, ...], ...]:
    glyph_name = font.getGlyphName(glyph_id)
    glyph_set = glyph_set_of(font)
    pen = DecomposingRecordingPen(glyph_set, skipMissingComponents=True)
    glyph_set[glyph_name].draw(pen)
    return tuple(tuple(contour) for contour in recording_to_contours(pen.value))


class BitmapFromContours:
    """`glyph_bitmap_for_gid` for programs that produce their own contours."""

    __slots__ = ()

    @abstractmethod
    def normalized_glyph_contours(self, glyph_id: int) -> tuple[tuple[Point, ...], ...]: ...

    def glyph_bitmap_for_gid(
        self, glyph_id: int, *, width: int = 24, height: int = 32
    ) -> tuple[int, ...]:
        return rasterize_contours(
            self.normalized_glyph_contours(glyph_id), width=width, height=height
        )


class BitmapFromOutlines:
    """`glyph_bitmap_for_gid` for programs backed by a `FontToolsOutlineAccess`."""

    __slots__ = ()

    outlines: FontToolsOutlineAccess

    def glyph_bitmap_for_gid(
        self, glyph_id: int, *, width: int = 24, height: int = 32
    ) -> tuple[int, ...]:
        return self.outlines.glyph_bitmap_for_gid(glyph_id, width=width, height=height)


TrueTypeTables = tuple[bytes, numpy.ndarray[Any, Any], numpy.ndarray[Any, Any], int]


def truetype_tables(font: TTFont) -> TrueTypeTables | None:
    """What truetype_contours reads, when it can stand in for fontTools' drawing.

    That is a font fontTools draws from glyf -- one without a CFF table,
    which getGlyphSet would prefer, and not a variable font -- whose glyph
    set fontTools can build without decompiling glyf here: every glyph's loca
    slice lies within the table, as the decompile's length check demands,
    and hmtx, and vmtx if there is one, hold every glyph. Its glyph names
    must be unique, since fontTools finds glyphs by name. None otherwise,
    and fontTools draws every glyph.
    """
    try:
        keys = set(font.keys())
        if (
            "CFF " in keys
            or "CFF2" in keys
            or "fvar" in keys
            or not {"glyf", "loca", "hmtx"} <= keys
        ):
            # A glyph set reads fvar's axes, and fails when it has none; a
            # variable font is left to fontTools altogether.
            return None
        reader = font.reader
        if reader is None:
            return None
        glyf = bytes(reader["glyf"])
        loca = numpy.asarray(font["loca"].locations, dtype=numpy.int64)
        order = font.getGlyphOrder()
        if len(set(order)) != len(order):
            return None
        metrics = font["hmtx"].metrics
        lsb = numpy.asarray([int(metrics[name][1]) for name in order], dtype=numpy.int64)
        if "vmtx" in keys:
            # The glyph set reads vmtx too, and each glyph drawn its entry.
            vertical = font["vmtx"].metrics
            if not all(name in vertical for name in order):
                return None
    except Exception:
        return None
    if len(loca):
        starts = loca[:-1]
        ends = loca[1:]
        # data[pos:next] must be next - pos bytes long.
        if bool(((ends < starts) | ((ends > len(glyf)) & (ends != starts))).any()):
            return None
    return glyf, loca, lsb, len(order)


class FontToolsOutlineAccess(BitmapFromContours):
    __slots__ = ("font", "glyph_count", "reverse_glyph_map", "scale", "truetype", "truetype_read")

    def __init__(self, font: TTFont) -> None:
        self.font = font
        self.glyph_count = len(font.getGlyphOrder())
        self.reverse_glyph_map = font.getReverseGlyphMap()
        units_per_em = float(getattr(font["head"], "unitsPerEm", 1000) or 1000)
        self.scale = 1000.0 / units_per_em if units_per_em else 1.0
        # Read on the first glyph drawn; None if truetype_contours cannot be used.
        self.truetype: TrueTypeTables | None = None
        self.truetype_read = False

    def glyph_id_for_name(self, glyph_name: str) -> int | None:
        return self.reverse_glyph_map.get(glyph_name)

    def has_glyph_id(self, glyph_id: int) -> bool:
        return 0 <= glyph_id < self.glyph_count

    def normalized_glyph_contours(self, glyph_id: int) -> tuple[tuple[Point, ...], ...]:
        if not self.truetype_read:
            self.truetype = truetype_tables(self.font)
            self.truetype_read = True
        tables = self.truetype
        if tables is not None:
            glyf, loca, lsb, glyph_count = tables
            drawn = truetype_contours(glyf, loca, lsb, glyph_count, glyph_id, self.scale)
            if drawn is not None:
                return drawn
        try:
            contours = fonttools_contours(self.font, glyph_id)
            return contours if self.scale == 1.0 else scale_contours(contours, self.scale)
        except Exception:
            return ()

    def glyph_bbox_for_gid(self, glyph_id: int) -> tuple[float, float, float, float] | None:
        try:
            return fonttools_bbox(self.font, glyph_id, self.scale)
        except Exception:
            return None


class RecoverableFontTableWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            message.endswith(
                (
                    "extra bytes in post.stringData array",
                    " timestamp seems very low; regarding as unix timestamp",
                )
            )
        )


FONT_TABLE_WARNING_FILTER = RecoverableFontTableWarningFilter()
for logger_name in (
    "fontTools.ttLib.tables._p_o_s_t",
    "fontTools.ttLib.tables._h_e_a_d",
    "core_pdf._vendor.fontTools.ttLib.tables._p_o_s_t",
    "core_pdf._vendor.fontTools.ttLib.tables._h_e_a_d",
):
    logging.getLogger(logger_name).addFilter(FONT_TABLE_WARNING_FILTER)


class TrueTypeFontProgram(BitmapFromOutlines):
    __slots__ = (
        "data",
        "font",
        "units_per_em",
        "cid_to_gid",
        "cmap",
        "unicode_cmap",
        "glyph_to_unicode",
        "outlines",
        "glyph_locations",
        "glyph_table_data",
        "composite_bbox_cache",
    )

    def __init__(
        self,
        data: bytes,
        cid_to_gid: bytes | None = None,
        *,
        use_cmap: bool = False,
    ) -> None:
        self.data = data
        self.composite_bbox_cache: dict[
            int, tuple[tuple[float, float, float, float] | None, bool]
        ] = {}
        self.font = tt_font_from_data(data)
        if not {"maxp", "glyf", "loca", "head"} <= set(self.font.keys()):
            raise ValueError("invalid TrueType glyph tables")
        ensure_glyph_order(self.font)
        self.units_per_em = float(getattr(self.font["head"], "unitsPerEm", 1000) or 1000)
        self.glyph_locations, self.glyph_table_data = raw_glyph_locations(self.font)
        self.outlines = FontToolsOutlineAccess(self.font)
        self.cid_to_gid = cid_to_gid
        self.unicode_cmap = best_unicode_gid_cmap(self.font)
        self.glyph_to_unicode = invert_unicode_cmap(self.unicode_cmap)
        if use_cmap:
            self.cmap = self.unicode_cmap or code_gid_cmap(self.font)
        else:
            self.cmap = {}

    def variant(self, cid_to_gid: bytes | None, *, use_cmap: bool) -> TrueTypeFontProgram:
        variant = object.__new__(TrueTypeFontProgram)
        for name in TrueTypeFontProgram.__slots__:
            setattr(variant, name, getattr(self, name))
        variant.cid_to_gid = cid_to_gid
        if use_cmap:
            variant.cmap = self.unicode_cmap or code_gid_cmap(self.font)
        else:
            variant.cmap = {}
        return variant

    def glyph_id_for_code(self, code: int) -> int:
        if self.cid_to_gid is not None:
            pos = code * 2
            if pos >= 0 and pos + 2 <= len(self.cid_to_gid):
                return struct.unpack(">H", self.cid_to_gid[pos : pos + 2])[0]
            return 0
        if self.cmap:
            return self.cmap.get(code, code)
        return code

    def glyph_id_for_unicode(self, codepoint: int) -> int:
        if self.cmap:
            return self.cmap.get(codepoint, 0)
        return 0

    def has_glyph_id(self, gid: int) -> bool:
        return self.outlines.has_glyph_id(gid)

    def unicode_for_gid(self, gid: int) -> str:
        return self.glyph_to_unicode.get(gid, "")

    def glyph_bbox(self, code: int) -> tuple[float, float, float, float] | None:
        return self.glyph_bbox_for_gid(self.glyph_id_for_code(code))

    def glyph_bbox_for_gid(self, gid: int) -> tuple[float, float, float, float] | None:
        bbox = glyph_header_bbox(self.glyph_locations, self.glyph_table_data, gid)
        if bbox is None:
            return None
        scale = 1000.0 / self.units_per_em if self.units_per_em else 1.0
        if scale == 1.0:
            return bbox
        x0, y0, x1, y1 = bbox
        return (x0 * scale, y0 * scale, x1 * scale, y1 * scale)

    def normalized_glyph_contours(self, gid: int) -> tuple[tuple[Point, ...], ...]:
        return self.outlines.normalized_glyph_contours(gid)

    def glyph_contours_for_gid(self, gid: int) -> tuple[tuple[Point, ...], ...]:
        try:
            return fonttools_contours(self.font, gid)
        except Exception:
            return ()

    def composite_body_bbox(
        self, gid: int
    ) -> tuple[tuple[float, float, float, float] | None, bool]:
        cache = self.composite_bbox_cache
        try:
            return cache[gid]
        except KeyError:
            result = cache[gid] = self.composite_body_bbox_uncached(gid)
            return result

    def composite_body_bbox_uncached(
        self, gid: int
    ) -> tuple[tuple[float, float, float, float] | None, bool]:
        try:
            glyph_name = self.font.getGlyphName(gid)
            glyf = self.font["glyf"]
            glyph = glyf[glyph_name]
            if not glyph.isComposite():
                return (None, False)
            body_bbox: tuple[float, float, float, float] | None = None
            has_dot = False
            for component in glyph.components:
                component_name, transform = component.getComponentInfo()
                bbox = glyph_bbox(glyf, component_name)
                if bbox is None:
                    continue
                xx, xy, yx, yy, dx, dy = transform
                xmin, ymin, xmax, ymax = transform_bbox(bbox, (xx, yx, xy, yy, dx, dy))
                w, h = xmax - xmin, ymax - ymin
                if h > 0 and h < 600 and 0.4 < w / h < 2.5 and ymin > 900:
                    has_dot = True
                else:
                    body_bbox = (xmin, ymin, xmax, ymax)
            return (body_bbox, has_dot)
        except Exception:
            return (None, False)


PROGRAM_CACHE_LIMIT = 64
program_cache = threading.local()


def cached_truetype_program(
    data: bytes, cid_to_gid: bytes | None = None, *, use_cmap: bool = False
) -> TrueTypeFontProgram:
    programs: dict[object, TrueTypeFontProgram] | None = getattr(program_cache, "programs", None)
    if programs is None:
        programs = program_cache.programs = {}
    key: object = (data, cid_to_gid, use_cmap)
    program = programs.get(key)
    if program is None:
        base = programs.get(data)
        if base is None:
            if len(programs) >= PROGRAM_CACHE_LIMIT:
                programs.clear()
            base = programs[data] = TrueTypeFontProgram(data)
        program = (
            base
            if cid_to_gid is None and not use_cmap
            else base.variant(cid_to_gid, use_cmap=use_cmap)
        )
        programs[key] = program
    return program


def tt_font_from_data(data: bytes) -> TTFont:
    try:
        return parse_truetype_program(data)
    except Exception as exc:
        raise ValueError("invalid TrueType font program") from exc


def ensure_glyph_order(font: TTFont) -> None:
    try:
        font.getGlyphOrder()
        return
    except Exception:
        pass
    try:
        glyph_count = int(font["maxp"].numGlyphs)
    except Exception as exc:
        raise ValueError("invalid TrueType glyph order") from exc
    if glyph_count <= 0:
        raise ValueError("invalid TrueType glyph order")
    font.setGlyphOrder([".notdef", *(f"glyph{gid:05d}" for gid in range(1, glyph_count))])


def best_unicode_gid_cmap(font: TTFont) -> dict[int, int]:
    symbol_fallback = False
    try:
        cmap_table = font["cmap"]
        name_cmap = cmap_table.getBestCmap()
        if name_cmap is None:
            symbol_cmap = cmap_table.getcmap(3, 0)
            name_cmap = symbol_cmap.cmap if symbol_cmap is not None else {}
            symbol_fallback = bool(name_cmap)
        reverse_glyph_map = font.getReverseGlyphMap()
    except Exception:
        return {}
    mapping: dict[int, int] = {}
    for codepoint, glyph_name in name_cmap.items():
        if not is_unicode_scalar(codepoint):
            continue
        try:
            gid = reverse_glyph_map[glyph_name]
        except KeyError:
            if not glyph_name.startswith("glyph"):
                continue
            try:
                gid = int(glyph_name[5:])
            except ValueError:
                continue
        if gid > 0:
            mapping[codepoint] = gid
            if symbol_fallback and 0xF000 <= codepoint <= 0xF2FF:
                mapping.setdefault(symbol_character_code(codepoint), gid)
    return mapping


def code_gid_cmap(font: TTFont) -> dict[int, int]:
    try:
        cmap_table = font["cmap"]
        reverse_glyph_map = font.getReverseGlyphMap()
    except Exception:
        return {}

    def gid_for(glyph_name: str) -> int:
        try:
            return int(reverse_glyph_map[glyph_name])
        except KeyError:
            if glyph_name.startswith("glyph"):
                try:
                    return int(glyph_name[5:])
                except ValueError:
                    return 0
            return 0

    for platform, encoding in ((3, 0), (1, 0)):
        try:
            subtable = cmap_table.getcmap(platform, encoding)
        except Exception:
            continue
        if subtable is None:
            continue
        mapping: dict[int, int] = {}
        for code, glyph_name in subtable.cmap.items():
            gid = gid_for(glyph_name)
            if gid <= 0:
                continue
            mapping.setdefault(code, gid)
            if platform == 3 and 0xF000 <= code <= 0xF0FF:
                mapping.setdefault(code - 0xF000, gid)
        if mapping:
            return mapping
    return {}


def invert_unicode_cmap(cmap: dict[int, int]) -> dict[int, str]:
    by_gid: dict[int, str] = {}
    for codepoint, gid in cmap.items():
        if gid <= 0 or not is_unicode_scalar(codepoint):
            continue
        char = chr(codepoint)
        previous = by_gid.get(gid)
        if previous is None or prefer_unicode_text(char, previous):
            by_gid[gid] = char
    return by_gid


def prefer_unicode_text(candidate: str, current: str) -> bool:
    candidate_score = unicode_text_score(candidate)
    current_score = unicode_text_score(current)
    if candidate_score != current_score:
        return candidate_score > current_score
    return ord(candidate) < ord(current)


def unicode_text_score(char: str) -> int:
    code = ord(char)
    if char.isalnum():
        return 5
    if char.isprintable() and not char.isspace() and code < 0xE000:
        return 4
    if char.isspace():
        return 3
    if 0xE000 <= code <= 0xF8FF:
        return 1
    if code < 32:
        return 0
    return 2


def recording_to_contours(
    recording: list[tuple[str, tuple[Any, ...]]],
) -> list[list[Point]]:
    contours: list[list[Point]] = []
    contour: list[Point] = []
    current: Point | None = None
    start: Point | None = None
    for operator, operands in recording:
        match operator:
            case "moveTo":
                if contour:
                    contours.append(close_contour(contour))
                start = make_point(operands[0])
                current = start
                contour = [start]
            case "lineTo" if current is not None:
                current = make_point(operands[0])
                contour.append(current)
            case "qCurveTo" if current is not None:
                current = append_quadratic(contour, current, start, operands)
            case "curveTo" if current is not None:
                current = append_cubic(contour, current, operands)
            case "closePath" | "endPath":
                if contour:
                    contours.append(close_contour(contour))
                    contour = []
                    current = None
                    start = None
    if contour:
        contours.append(close_contour(contour))
    return [contour for contour in contours if len(contour) >= 3]


def make_point(value: Any) -> Point:
    x, y = value
    return (float(x), float(y))


def append_quadratic(
    contour: list[Point],
    current: Point,
    start: Point | None,
    operands: tuple[Any, ...],
) -> Point:
    points = list(operands)
    if not points:
        return current
    if points[-1] is None:
        if start is None:
            return current
        controls = [make_point(point) for point in points[:-1]]
        end = start
    else:
        controls = [make_point(point) for point in points[:-1]]
        end = make_point(points[-1])
    if not controls:
        contour.append(end)
        return end
    segment_start = current
    for index, control in enumerate(controls):
        segment_end = (
            end
            if index == len(controls) - 1
            else (
                (control[0] + controls[index + 1][0]) * 0.5,
                (control[1] + controls[index + 1][1]) * 0.5,
            )
        )
        contour.extend(flatten_quadratic(segment_start, control, segment_end))
        segment_start = segment_end
    return end


def append_cubic(contour: list[Point], current: Point, operands: tuple[Any, ...]) -> Point:
    if len(operands) % 3:
        return current
    segment_start = current
    for index in range(0, len(operands), 3):
        c1 = make_point(operands[index])
        c2 = make_point(operands[index + 1])
        end = make_point(operands[index + 2])
        contour.extend(flatten_cubic(segment_start, c1, c2, end))
        segment_start = end
    return segment_start


def close_contour(contour: list[Point]) -> list[Point]:
    if contour and contour[0] != contour[-1]:
        return [*contour, contour[0]]
    return contour


def flatten_quadratic(p0: Point, p1: Point, p2: Point, segments: int = 6) -> list[Point]:
    out: list[Point] = []
    for i in range(1, segments + 1):
        t = i / segments
        mt = 1.0 - t
        out.append(
            (
                mt * mt * p0[0] + 2.0 * mt * t * p1[0] + t * t * p2[0],
                mt * mt * p0[1] + 2.0 * mt * t * p1[1] + t * t * p2[1],
            )
        )
    return out


def flatten_cubic(p0: Point, p1: Point, p2: Point, p3: Point, segments: int = 8) -> list[Point]:
    out: list[Point] = []
    for i in range(1, segments + 1):
        t = i / segments
        mt = 1.0 - t
        out.append(
            (
                mt**3 * p0[0] + 3.0 * mt * mt * t * p1[0] + 3.0 * mt * t * t * p2[0] + t**3 * p3[0],
                mt**3 * p0[1] + 3.0 * mt * mt * t * p1[1] + 3.0 * mt * t * t * p2[1] + t**3 * p3[1],
            )
        )
    return out


def parse_opentype_program(data: bytes) -> TTFont:
    font = TTFont(BytesIO(data), lazy=True, recalcBBoxes=False, recalcTimestamp=False)
    if not ({"CFF ", "CFF2"} & set(font.keys())):
        raise ValueError("OpenType font has no CFF outline table")
    return font


class OpenTypeFontProgram(BitmapFromOutlines):
    __slots__ = (
        "font",
        "outlines",
    )

    def __init__(self, data: bytes) -> None:
        try:
            self.font = parse_opentype_program(data)
            self.outlines = FontToolsOutlineAccess(self.font)
            if "CFF2" in self.font and "fvar" in self.font:
                del self.font["fvar"]
        except Exception as exc:
            raise ValueError("invalid OpenType CFF font program") from exc

    def glyph_id_for_name(self, glyph_name: str) -> int | None:
        return self.outlines.glyph_id_for_name(glyph_name)

    def has_glyph_id(self, glyph_id: int) -> bool:
        return self.outlines.has_glyph_id(glyph_id)

    def normalized_glyph_contours(self, glyph_id: int) -> tuple[tuple[Point, ...], ...]:
        return self.outlines.normalized_glyph_contours(glyph_id)

    def glyph_bbox_for_gid(self, glyph_id: int) -> tuple[float, float, float, float] | None:
        return self.outlines.glyph_bbox_for_gid(glyph_id)


LEN_IV_RE = re.compile(rb"/lenIV\s+(-?\d+)\s+def\b")
FONT_MATRIX_RE = re.compile(
    rb"/FontMatrix\s*\[\s*([-+.\dEe]+)\s+([-+.\dEe]+)\s+"
    rb"([-+.\dEe]+)\s+([-+.\dEe]+)\s+([-+.\dEe]+)\s+([-+.\dEe]+)\s*\]"
)
SUBR_RE = re.compile(rb"\bdup\s+(\d+)\s+(\d+)\s+(?:RD|-\|)[ \t\r\n]")
CHARSTRING_RE = re.compile(rb"/([^\s/]+)\s+(\d+)\s+(?:RD|-\|)[ \t\r\n]")
HEX_BYTES = frozenset(b"0123456789abcdefABCDEF \t\r\n")
MAX_SUBROUTINES = 4096


class Type1FontProgramBase:
    __slots__ = (
        "charstrings",
        "font_matrix",
        "glyph_names",
        "glyph_name_to_id",
        "subrs",
    )

    def __init__(self, data: bytes, *, length1: int | None = None) -> None:
        private = self.decode_private(data, length1)
        len_iv_match = LEN_IV_RE.search(private)
        len_iv = int(len_iv_match.group(1)) if len_iv_match is not None else 4
        if len_iv < -1 or len_iv > 32:
            raise ValueError("invalid Type 1 lenIV")

        subr_data = {
            int(index): payload for index, payload in self.binary_entries(private, SUBR_RE)
        }
        subr_count = max(subr_data, default=-1) + 1
        if subr_count > MAX_SUBROUTINES:
            raise ValueError("Type 1 subroutine index exceeds decoder limit")
        empty = T1CharString(b"\x0b", subrs=[])
        subrs = [empty for _ in range(subr_count)]
        for index, encrypted in subr_data.items():
            subrs[index] = self.prepare_charstring(encrypted, len_iv, subrs)
        for subr in subrs:
            subr.subrs = subrs
        self.subrs = subrs

        charstrings = {
            name.decode("latin-1"): payload
            for name, payload in self.binary_entries(private, CHARSTRING_RE)
        }
        self.charstrings = {
            name: self.prepare_charstring(encrypted, len_iv, subrs)
            for name, encrypted in charstrings.items()
        }
        if not self.charstrings:
            raise ValueError("Type 1 CharStrings are missing")
        self.glyph_names = tuple(self.charstrings)
        self.glyph_name_to_id = {name: gid for gid, name in enumerate(self.glyph_names)}

        matrix_match = FONT_MATRIX_RE.search(data)
        self.font_matrix = (
            tuple(float(value) for value in matrix_match.groups())
            if matrix_match is not None
            else (0.001, 0.0, 0.0, 0.001, 0.0, 0.0)
        )

    def glyph_id_for_name(self, glyph_name: str) -> int | None:
        glyph_id = self.glyph_name_to_id.get(glyph_name)
        if glyph_id is not None:
            return glyph_id
        return self.glyph_name_to_id.get(".notdef")

    def has_glyph_id(self, glyph_id: int) -> bool:
        return 0 <= glyph_id < len(self.glyph_names)

    def glyph_bbox_for_gid(self, glyph_id: int) -> tuple[float, float, float, float] | None:
        if not self.has_glyph_id(glyph_id):
            return None
        glyph_name = self.glyph_names[glyph_id]
        charstring = self.charstrings.get(glyph_name) or self.charstrings.get(".notdef")
        if charstring is None:
            return None
        bounds_pen = BoundsPen(self.charstrings)
        a, b, c, d, e, f = self.font_matrix
        normalized_pen = TransformPen(
            bounds_pen,
            (a * 1000.0, b * 1000.0, c * 1000.0, d * 1000.0, e * 1000.0, f * 1000.0),
        )
        charstring.draw(normalized_pen)
        bounds = bounds_pen.bounds
        if bounds is None:
            return None
        x_min, y_min, x_max, y_max = bounds
        return (float(x_min), float(y_min), float(x_max), float(y_max))

    @staticmethod
    def binary_entries(data: bytes, pattern: re.Pattern[bytes]) -> Iterator[tuple[bytes, bytes]]:
        return binary_entries(data, pattern)

    @staticmethod
    def prepare_charstring(
        encrypted: bytes, len_iv: int, subrs: list[T1CharString]
    ) -> T1CharString:
        return T1CharString(decode_charstring(encrypted, len_iv), subrs=subrs)

    @staticmethod
    def decode_private(data: bytes, length1: int | None) -> bytes:
        return decode_eexec_payload(data, length1)


def eexec_payload(data: bytes, length1: int | None) -> bytes:
    if length1 is not None and 0 < length1 < len(data):
        encrypted = data[length1:]
    else:
        marker = data.find(b"currentfile eexec")
        if marker < 0:
            raise ValueError("Type 1 eexec section is missing")
        encrypted = data[marker + len(b"currentfile eexec") :].lstrip()
    sample = encrypted[: min(len(encrypted), 512)]
    if sample and all(byte in HEX_BYTES for byte in sample):
        compact = bytes(byte for byte in encrypted if byte not in b" \t\r\n")
        if len(compact) % 2:
            compact = compact[:-1]
        try:
            encrypted = bytes.fromhex(compact.decode("ascii"))
        except ValueError as exc:
            raise ValueError("invalid hexadecimal Type 1 eexec section") from exc
    decrypted = decrypt_type1(encrypted, 55665)
    if len(decrypted) < 4:
        raise ValueError("truncated Type 1 eexec section")
    return decrypted[4:]


class Type1FontProgram(Type1FontProgramBase, BitmapFromContours):
    __slots__ = ()

    @staticmethod
    def binary_entries(data: bytes, pattern: re.Pattern[bytes]) -> Iterator[tuple[bytes, bytes]]:
        for match in pattern.finditer(data):
            length = int(match.group(2))
            start = match.end()
            if length >= 0 and start + length <= len(data):
                yield match.group(1), data[start : start + length]

    @staticmethod
    def prepare_charstring(
        encrypted: bytes, len_iv: int, subrs: list[T1CharString]
    ) -> T1CharString:
        decoded = decrypt_type1(encrypted, 4330)
        return T1CharString(decoded[len_iv:] if len_iv >= 0 else decoded, subrs=subrs)

    @staticmethod
    def decode_private(data: bytes, length1: int | None) -> bytes:
        return eexec_payload(data, length1)

    def glyph_bbox_for_gid(self, glyph_id: int) -> tuple[float, float, float, float] | None:
        try:
            return super().glyph_bbox_for_gid(glyph_id)
        except Exception:
            return None

    def normalized_glyph_contours(self, glyph_id: int) -> tuple[tuple[Point, ...], ...]:
        if not self.has_glyph_id(glyph_id):
            return ()
        return self.glyph_contours(self.glyph_names[glyph_id])

    def glyph_contours(self, glyph_name: str) -> tuple[tuple[Point, ...], ...]:
        charstring = self.charstrings.get(glyph_name) or self.charstrings.get(".notdef")
        if charstring is None:
            return ()
        try:
            pen = RecordingPen()
            charstring.draw(pen)
            contours = recording_to_contours(pen.value)
            return transform_contours(contours, self.font_matrix)
        except Exception:
            return ()


def parse_type1_font_program_encoding(font_program: bytes | memoryview) -> dict[int, str]:
    return parse_encoding(font_program, skip_out_of_range=True)


__all__ = (
    "CFFFont",
    "CFFGlyphFeature",
    "CFFUnicodeRepairIndex",
    "OpenTypeFontProgram",
    "Point",
    "STANDARD_GLYPH_SIDS",
    "TrueTypeFontProgram",
    "cached_truetype_program",
    "glyph_feature_distance",
    "is_repairable_to_unicode_label",
    "rasterize_contours",
)
