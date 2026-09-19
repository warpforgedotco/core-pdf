"""CFF font-program parsing and glyph geometry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import inf, isfinite

from core_pdf._vendor.fontTools.cffLib import (
    cffExpertSubsetStrings,
    cffIExpertStrings,
    cffISOAdobeStrings,
)
from core_pdf._vendor.fontTools.encodings.StandardEncoding import StandardEncoding
from core_pdf.impl._impl.fonts.feature_distance_kernel import (
    feature_distance as compiled_feature_distance,
)
from core_pdf.impl._impl.fonts.feature_distance_kernel import (
    feature_distance_matrix as compiled_feature_distance_matrix,
)
from core_pdf.impl._impl.fonts.feature_distance_kernel import internal_feature_arrays
from core_pdf.impl._impl.fonts.raster_kernel import rasterize_contours, transform_contours
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.s_09_fonts.font_program import (
    CFF_EXPERT_ENCODING_CODES,
    CFF_STANDARD_STRING_COUNT,
    DEFAULT_CFF_FONT_MATRIX,
    STANDARD_GLYPH_SIDS,
    cubic_extrema_times,
    cubic_point,
    execute_type2_charstring,
)
from core_pdf_spec.s_09_fonts.font_program import CFFFont as PdfCFFFont
from core_pdf_spec.s_09_fonts.font_program import (
    cff_font_matrix as pdf_cff_font_matrix,
)


@dataclass(frozen=True)
class CFFGlyphFeature:
    cells: tuple[tuple[int, int], ...]
    aspect: float
    contours: int
    bitmap: tuple[int, ...] = ()


EMPTY_FEATURE = CFFGlyphFeature((), 0.0, 0, ())
# Type 2 charstrings may call subroutines, which may call further subroutines. The spec
# allows 10 levels; deeper than that means a malformed or maliciously recursive font.
assert len(STANDARD_GLYPH_SIDS) == CFF_STANDARD_STRING_COUNT

internal_TYPE2_RANDOM_INITIAL_STATE = 0x1234ABCD
internal_CUBIC_FLATNESS = 0.25
internal_CUBIC_MAX_DEPTH = 12


def internal_cff_font_matrix(
    font_dict: dict[int | tuple[int, int], list[float]],
) -> Matrix | None:
    try:
        return pdf_cff_font_matrix(font_dict)
    except (TypeError, ValueError):
        return None


class CFFFont(PdfCFFFont):
    __slots__ = ()

    def __init__(self, data: bytes | memoryview | None) -> None:
        if data is not None:
            super().__init__(data)
            return
        self.data = b""
        self.top_dict = {}
        self.charstrings = []
        self.cid_to_gid = {}
        self.custom_string_sids = {}
        self.is_cid_keyed = False
        self.global_subrs = ()
        self.local_subrs = ()
        self.fd_select = ()
        self.font_dicts = ()

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
        try:
            return super().read_charset(pos, glyph_count)
        except (IndexError, TypeError, ValueError):
            return self.internal_recover_read_charset(pos, glyph_count)

    def internal_recover_read_charset(self, pos: int, glyph_count: int) -> dict[int, int]:
        if glyph_count <= 0:
            return {}
        if self.is_cid_keyed and pos in {0, 1, 2}:
            # Section 18 explicitly forbids predefined charsets for CIDFonts:
            # their charset values are CIDs, not the SIDs in these tables.
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
            # Predefined charsets are name/SID sequences in GID order, not
            # identity SID-to-GID maps. Malformed fonts that declare more
            # glyphs than the selected charset simply leave the excess GIDs
            # unreachable by name.
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
        try:
            return super().read_encoding_codes(pos)
        except (IndexError, TypeError, ValueError):
            return self.internal_recover_read_encoding_codes(pos)

    def internal_recover_read_encoding_codes(self, pos: int) -> dict[int, int]:
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
        supplies none. StandardEncoding may be left to the caller's standard
        fallback, while predefined ExpertEncoding is exposed explicitly.
        """
        if self.is_cid_keyed:
            # A CIDFont specifies no encoding (CFF specification, section 12).
            return {}
        operand = self.top_dict.get(16, [0])[0]
        if not isinstance(operand, (int, float)):
            return {}
        offset = int(operand)
        if offset == 0:
            # The caller already applies StandardEncoding as its implicit base.
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
        """Return whether the CFF encoding completely governs its code space.

        StandardEncoding is already represented by the decoder's named base.
        ExpertEncoding and custom encodings are authoritative even when a
        subset happens to expose no encoded glyphs: all unspecified codes are
        unencoded rather than inherited from StandardEncoding.
        """
        if self.is_cid_keyed:
            return False
        operand = self.top_dict.get(16, [0])[0]
        if not isinstance(operand, (int, float)):
            return False
        return int(operand) != 0

    def read_fd_select(self) -> tuple[int, ...]:
        if not self.is_cid_keyed and (12, 37) in self.top_dict:
            return self.internal_recover_read_fd_select()
        try:
            return super().read_fd_select()
        except (IndexError, TypeError, ValueError):
            return self.internal_recover_read_fd_select()

    def internal_recover_read_fd_select(self) -> tuple[int, ...]:
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
        try:
            return super().read_font_dicts()
        except (IndexError, OverflowError, TypeError, ValueError):
            return self.internal_recover_read_font_dicts()

    def internal_recover_read_font_dicts(
        self,
    ) -> tuple[dict[int | tuple[int, int], list[float]], ...]:
        """Read the CID font dictionaries while preserving their FD indices."""
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
            except (IndexError, ValueError):
                # An invalid entry must retain its position because FDSelect
                # addresses this INDEX by ordinal.
                font_dicts.append({})
        return tuple(font_dicts)

    def read_private_subrs(
        self, font_dict: dict[int | tuple[int, int], list[float]]
    ) -> list[bytes]:
        try:
            return super().read_private_subrs(font_dict)
        except (IndexError, TypeError, ValueError):
            return self.internal_recover_read_private_subrs(font_dict)

    def internal_recover_read_private_subrs(
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
        try:
            return super().local_subrs_for_glyph(glyph_id)
        except (IndexError, TypeError, ValueError):
            return self.internal_recover_local_subrs_for_glyph(glyph_id)

    def internal_recover_local_subrs_for_glyph(self, glyph_id: int) -> tuple[bytes, ...]:
        fd_index = self.fd_select[glyph_id] if 0 <= glyph_id < len(self.fd_select) else 0
        if 0 <= fd_index < len(self.local_subrs):
            return self.local_subrs[fd_index]
        return ()

    def font_matrix(self, glyph_id: int) -> Matrix:
        try:
            return super().font_matrix(glyph_id)
        except (IndexError, TypeError, ValueError):
            return self.internal_recover_font_matrix(glyph_id)

    def internal_recover_font_matrix(self, glyph_id: int) -> Matrix:
        """Return the effective font matrix for a glyph."""
        fd_index = self.fd_select[glyph_id] if 0 <= glyph_id < len(self.fd_select) else 0
        top_matrix = internal_cff_font_matrix(self.top_dict)
        font_dict = self.font_dicts[fd_index] if 0 <= fd_index < len(self.font_dicts) else None
        font_dict_matrix = internal_cff_font_matrix(font_dict) if font_dict is not None else None
        if top_matrix is None:
            return font_dict_matrix or DEFAULT_CFF_FONT_MATRIX
        if font_dict_matrix is None:
            return top_matrix
        return font_dict_matrix.multiply(top_matrix)

    def internal_seac_contours(
        self,
        base_code: int,
        accent_code: int,
        accent_dx: float,
        accent_dy: float,
    ) -> tuple[tuple[tuple[float, float], ...], ...]:
        """Resolve deprecated endchar components in raw charstring coordinates."""
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
            component_contours, ignored_bbox = internal_type2_glyph_geometry_impl(
                self.charstrings[glyph_id],
                local_subrs=self.local_subrs_for_glyph(glyph_id),
                global_subrs=self.global_subrs,
            )
            contours.extend(
                tuple((x + offset_x, y + offset_y) for x, y in contour)
                for contour in component_contours
            )
        return tuple(contours)

    def internal_glyph_geometry_for_gid(
        self, glyph_id: int, *, bounds_only: bool = False
    ) -> tuple[
        tuple[tuple[tuple[float, float], ...], ...],
        tuple[float, float, float, float] | None,
    ]:
        """Return the glyph's outline points and their bounds.

        With ``bounds_only`` the default font matrix needs no stored outline.
        Other axis-aligned matrices retain endpoints and coordinate extrema
        for transformation; a skewed or rotated matrix needs the flattened outline.
        """
        try:
            charstring = self.charstrings[glyph_id]
        except IndexError:
            return ((), None)
        matrix = self.font_matrix(glyph_id)
        contours, raw_bbox = internal_type2_glyph_geometry_impl(
            charstring,
            local_subrs=self.local_subrs_for_glyph(glyph_id),
            global_subrs=self.global_subrs,
            seac_resolver=self.internal_seac_contours,
            flatten=not bounds_only or matrix[1] != 0.0 or matrix[2] != 0.0,
            retain_contours=not bounds_only or matrix != DEFAULT_CFF_FONT_MATRIX,
        )
        if matrix == DEFAULT_CFF_FONT_MATRIX:
            # The interpreter tracked the bounds of exactly these points.
            return (tuple(tuple(contour) for contour in contours), raw_bbox)
        normalized = transform_contours(contours, matrix)
        return (normalized, internal_contours_bbox(normalized))

    def glyph_feature(self, glyph_id: int) -> CFFGlyphFeature:
        geometry = self.internal_glyph_geometry_for_gid(glyph_id)
        contours = geometry[0]
        if not contours:
            return EMPTY_FEATURE
        return internal_feature_from_contours(contours)

    def glyph_bitmap_for_gid(
        self, glyph_id: int, width: int = 24, height: int = 32
    ) -> tuple[int, ...]:
        geometry = self.internal_glyph_geometry_for_gid(glyph_id)
        contours = geometry[0]
        if not contours:
            return ()
        return rasterize_contours(contours, width=width, height=height)

    def glyph_bbox_for_gid(self, glyph_id: int) -> tuple[float, float, float, float] | None:
        geometry = self.internal_glyph_geometry_for_gid(glyph_id, bounds_only=True)
        return geometry[1]

    def normalized_glyph_contours(
        self, glyph_id: int
    ) -> tuple[tuple[tuple[float, float], ...], ...]:
        """Return the Type 2 outline normalized into PDF's 1000-unit glyph space."""
        return self.internal_glyph_geometry_for_gid(glyph_id)[0]


def internal_contours_bbox(
    contours: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[float, float, float, float] | None:
    points = tuple(point for contour in contours for point in contour)
    if not points:
        return None
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


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
    cells: set[tuple[int, int]] = set()
    add_cell = cells.add
    for px, py in points:
        cell_x = round((px - min_x) / width * 17)
        cell_y = round((py - min_y) / height * 23)
        add_cell((cell_x, cell_y))
    bitmap = rasterize_contours(contours, width=18, height=24)
    return CFFGlyphFeature(tuple(sorted(cells)), round(width / height, 2), len(contours), bitmap)


def internal_cubic_is_flat(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> bool:
    dx = p3[0] - p0[0]
    dy = p3[1] - p0[1]
    chord_squared = dx * dx + dy * dy
    tolerance_squared = internal_CUBIC_FLATNESS * internal_CUBIC_FLATNESS
    if chord_squared <= 1e-18:
        return (
            max(
                (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2,
                (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2,
            )
            <= tolerance_squared
        )
    cross1 = dx * (p1[1] - p0[1]) - dy * (p1[0] - p0[0])
    cross2 = dx * (p2[1] - p0[1]) - dy * (p2[0] - p0[0])
    return max(cross1 * cross1, cross2 * cross2) <= tolerance_squared * chord_squared


def internal_cubic_sample_times(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> tuple[float, ...]:
    """Adaptively flatten a cubic while retaining its exact coordinate extrema."""
    times = {
        1.0,
        *cubic_extrema_times(p0[0], p1[0], p2[0], p3[0]),
        *cubic_extrema_times(p0[1], p1[1], p2[1], p3[1]),
    }

    def subdivide(
        start: tuple[float, float],
        control1: tuple[float, float],
        control2: tuple[float, float],
        end: tuple[float, float],
        start_t: float,
        end_t: float,
        depth: int,
    ) -> None:
        if depth >= internal_CUBIC_MAX_DEPTH or internal_cubic_is_flat(
            start, control1, control2, end
        ):
            times.add(end_t)
            return
        point01 = ((start[0] + control1[0]) / 2.0, (start[1] + control1[1]) / 2.0)
        point12 = (
            (control1[0] + control2[0]) / 2.0,
            (control1[1] + control2[1]) / 2.0,
        )
        point23 = ((control2[0] + end[0]) / 2.0, (control2[1] + end[1]) / 2.0)
        point012 = ((point01[0] + point12[0]) / 2.0, (point01[1] + point12[1]) / 2.0)
        point123 = ((point12[0] + point23[0]) / 2.0, (point12[1] + point23[1]) / 2.0)
        midpoint = (
            (point012[0] + point123[0]) / 2.0,
            (point012[1] + point123[1]) / 2.0,
        )
        middle_t = (start_t + end_t) / 2.0
        subdivide(start, point01, point012, midpoint, start_t, middle_t, depth + 1)
        subdivide(midpoint, point123, point23, end, middle_t, end_t, depth + 1)

    subdivide(p0, p1, p2, p3, 0.0, 1.0, 0)
    return tuple(sorted(times))


def internal_type2_glyph_geometry_impl(  # noqa: C901 - direct dispatch mirrors Type 2's spec table
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
    """Execute a charstring into contours and their bounds.

    Flattening samples every curve adaptively for rasterization. Without it a
    curve contributes only its endpoints and coordinate extrema, the points that
    determine its bounds.

    Bounds-only consumers can omit contour storage when they do not need to
    transform the points. Contour completion and malformed-program recovery
    still determine which bounds are committed.
    """
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
    random_state = internal_TYPE2_RANDOM_INITIAL_STATE

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

    def append_completed_contour(points: tuple[tuple[float, float], ...]) -> None:
        nonlocal bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, bbox_has_points
        if not points:
            return
        if retain_contours:
            contours.append(list(points))
        bbox_min_x = min(bbox_min_x, *(point[0] for point in points))
        bbox_min_y = min(bbox_min_y, *(point[1] for point in points))
        bbox_max_x = max(bbox_max_x, *(point[0] for point in points))
        bbox_max_y = max(bbox_max_y, *(point[1] for point in points))
        bbox_has_points = True

    def record_point(px: float, py: float) -> None:
        nonlocal current_min_x, current_min_y, current_max_x, current_max_y
        nonlocal current_has_points
        if retain_contours:
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
        point0 = (x, y)
        point1 = (x + dx1, y + dy1)
        point2 = (point1[0] + dx2, point1[1] + dy2)
        point3 = (point2[0] + dx3, point2[1] + dy3)
        if flatten:
            for t in internal_cubic_sample_times(point0, point1, point2, point3):
                record_point(*cubic_point(point0, point1, point2, point3, t))
        else:
            for t in cubic_extrema_times(point0[0], point1[0], point2[0], point3[0]):
                record_point(*cubic_point(point0, point1, point2, point3, t))
            for t in cubic_extrema_times(point0[1], point1[1], point2[1], point3[1]):
                record_point(*cubic_point(point0, point1, point2, point3, t))
            record_point(*point3)
        x, y = point3

    def has_current_point() -> bool:
        return current_has_points

    def seac(base_code: int, accent_code: int, dx: float, dy: float) -> None:
        if seac_resolver is not None:
            for component in seac_resolver(base_code, accent_code, dx, dy):
                append_completed_contour(component)

    def random_value() -> float:
        nonlocal random_state
        random_state = (1103515245 * random_state + 12345) & 0x7FFFFFFF
        return (random_state + 1) / 0x80000000

    try:
        unfinished = execute_type2_charstring(
            charstring,
            local_subrs=local_subrs,
            global_subrs=global_subrs,
            move=move,
            line=line,
            curve=curve,
            flush_contour=flush_contour,
            has_current_point=has_current_point,
            seac=seac,
            random_value=random_value,
        )
    except (ArithmeticError, IndexError, ValueError):
        unfinished = False
    if unfinished:
        flush_contour()
    bbox = (bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y) if bbox_has_points else None
    return contours, bbox


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


class CFFUnicodeRepairIndex:
    """Match suspicious ToUnicode entries against one CFF program."""

    __slots__ = (
        "internal_candidate_gids",
        "internal_code_to_gid",
        "internal_font",
        "internal_labels",
        "internal_repairable_gids",
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

        self.internal_font = font
        self.internal_labels = labels
        self.internal_code_to_gid = code_to_gid
        self.internal_repairable_gids = frozenset(
            gid for gid, label in labels.items() if is_repairable_to_unicode_label(label)
        )
        self.internal_candidate_gids = tuple(
            gid
            for gid, label in labels.items()
            if len(label) == 1 and (label.isalnum() or label in ".-+")
        )

    def repairs_for_codes(self, codes: Iterable[bytes]) -> dict[bytes, str]:
        """Return repairs for the requested content-stream codes."""
        requested_codes = tuple(dict.fromkeys(codes))
        if not requested_codes or not self.internal_repairable_gids:
            return {}
        target_gids = tuple(
            dict.fromkeys(
                gid
                for code in requested_codes
                if (gid := self.internal_code_to_gid.get(code)) in self.internal_repairable_gids
            )
        )
        if not target_gids:
            return {}
        repairs = self.internal_repairs_for_gids(target_gids)
        return {
            code: replacement
            for code in requested_codes
            if (gid := self.internal_code_to_gid.get(code)) is not None
            and (replacement := repairs.get(gid)) is not None
        }

    def internal_repairs_for_gids(self, requested_gids: tuple[int, ...]) -> dict[int, str]:
        feature_gids = dict.fromkeys((*self.internal_candidate_gids, *requested_gids))
        features = {gid: self.internal_font.glyph_feature(gid) for gid in feature_gids}

        candidate_gids = tuple(gid for gid in self.internal_candidate_gids if features[gid].cells)
        target_gids = tuple(gid for gid in requested_gids if features[gid].cells)
        distance_lookups: dict[int, dict[int, float]] = {}
        if (
            target_gids
            and candidate_gids
            and (len(self.internal_repairable_gids) * len(candidate_gids) >= 512)
        ):
            target_features = [features[gid] for gid in target_gids]
            candidate_features = [features[gid] for gid in candidate_gids]
            candidate_arrays = internal_feature_arrays(
                [feature.cells for feature in candidate_features],
                [feature.bitmap for feature in candidate_features],
                [feature.aspect for feature in candidate_features],
                [feature.contours for feature in candidate_features],
            )
            distance_matrix = compiled_feature_distance_matrix(
                [feature.cells for feature in target_features],
                [feature.bitmap for feature in target_features],
                [feature.aspect for feature in target_features],
                [feature.contours for feature in target_features],
                [feature.cells for feature in candidate_features],
                [feature.bitmap for feature in candidate_features],
                [feature.aspect for feature in candidate_features],
                [feature.contours for feature in candidate_features],
                internal_right_arrays=candidate_arrays,
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
            label = self.internal_labels[glyph_id]
            replacement = internal_repair_candidate(
                glyph_id,
                label,
                features,
                self.internal_labels,
                distance_lookups.get(glyph_id),
            )
            if replacement is not None and replacement != label:
                repairs[glyph_id] = replacement
        return repairs


__all__ = (
    "STANDARD_GLYPH_SIDS",
    "CFFFont",
    "CFFGlyphFeature",
    "CFFUnicodeRepairIndex",
    "glyph_feature_distance",
    "is_repairable_to_unicode_label",
)
