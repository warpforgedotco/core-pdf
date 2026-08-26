"""CFF font-program parsing and glyph geometry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from math import inf, isfinite, sqrt
from threading import RLock

from core_pdf._vendor.fontTools.cffLib import (
    cffExpertSubsetStrings,
    cffIExpertStrings,
    cffISOAdobeStrings,
    cffStandardStrings,
)
from core_pdf._vendor.fontTools.encodings.StandardEncoding import StandardEncoding
from core_pdf.impl.engine.spec.s_09_fonts.feature_distance_kernel import (
    FeatureArrays,
    internal_feature_arrays,
)
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
CFFMatrix = tuple[float, float, float, float, float, float]
STANDARD_GLYPH_SIDS = {name: sid for sid, name in enumerate(cffStandardStrings)}
CFF_STANDARD_STRING_COUNT = len(cffStandardStrings)
# Type 2 charstrings may call subroutines, which may call further subroutines. The spec
# allows 10 levels; deeper than that means a malformed or maliciously recursive font.
TYPE2_MAX_SUBR_DEPTH = 10
assert len(STANDARD_GLYPH_SIDS) == CFF_STANDARD_STRING_COUNT

internal_TYPE2_MAX_STACK = 48
internal_TYPE2_TRANSIENT_SIZE = 32
internal_TYPE2_RANDOM_INITIAL_STATE = 0x1234ABCD
internal_CUBIC_FLATNESS = 0.25
internal_CUBIC_MAX_DEPTH = 12
internal_DEFAULT_CFF_FONT_MATRIX: CFFMatrix = (0.001, 0.0, 0.0, 0.001, 0.0, 0.0)

# Appendix B of the CFF specification assigns the non-.notdef entries of the
# predefined ExpertEncoding to the Expert charset names in this order. Keeping
# only the occupied codes lets the authoritative names continue to come from
# vendored fontTools instead of duplicating another 165-name table here.
internal_CFF_EXPERT_ENCODING_CODES = tuple(
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
assert len(internal_CFF_EXPERT_ENCODING_CODES) == len(cffIExpertStrings) - 1


def internal_cff_font_matrix(
    font_dict: dict[int | tuple[int, int], list[float]],
) -> CFFMatrix | None:
    values = font_dict.get((12, 7))
    if not isinstance(values, list) or len(values) != 6:
        return None
    try:
        a, b, c, d, e, f = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    matrix = (a, b, c, d, e, f)
    return matrix if all(isfinite(value) for value in matrix) else None


def internal_compose_cff_matrices(outer: CFFMatrix, inner: CFFMatrix) -> CFFMatrix:
    """Compose CFF matrices so that ``inner`` is applied before ``outer``."""
    oa, ob, oc, od, oe, of = outer
    ia, ib, ic, id_, ie, if_ = inner
    return (
        oa * ia + oc * ib,
        ob * ia + od * ib,
        oa * ic + oc * id_,
        ob * ic + od * id_,
        oa * ie + oc * if_ + oe,
        ob * ie + od * if_ + of,
    )


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
            self.font_dicts: tuple[dict[int | tuple[int, int], list[float]], ...] = ()
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
        self.font_dicts = self.internal_read_font_dicts()
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
                elif nibble == 13:
                    # 0xd is reserved by the CFF real-number encoding. Treating
                    # it as whitespace silently joins the surrounding digits.
                    raise ValueError("invalid CFF real number")
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
                    internal_CFF_EXPERT_ENCODING_CODES,
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
        for code, gid in self.internal_read_encoding_codes(offset).items():
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

    def internal_read_font_dicts(
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
            raw_font_dicts, ignored_pos = self.internal_read_index(int(fdarray_off))
        except ValueError:
            return ()

        font_dicts: list[dict[int | tuple[int, int], list[float]]] = []
        for raw_font_dict in raw_font_dicts:
            try:
                font_dicts.append(self.internal_parse_dict(raw_font_dict))
            except (IndexError, ValueError):
                # An invalid entry must retain its position because FDSelect
                # addresses this INDEX by ordinal.
                font_dicts.append({})
        return tuple(font_dicts)

    def internal_read_local_subrs(self) -> tuple[tuple[bytes, ...], ...]:
        if self.is_cid_keyed:
            return tuple(
                tuple(self.internal_read_private_subrs(font_dict)) for font_dict in self.font_dicts
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

    def internal_font_matrix(self, glyph_id: int) -> CFFMatrix:
        top_matrix = internal_cff_font_matrix(self.top_dict)
        fd_index = self.fd_select[glyph_id] if 0 <= glyph_id < len(self.fd_select) else 0
        font_dict = self.font_dicts[fd_index] if 0 <= fd_index < len(self.font_dicts) else None
        font_dict_matrix = internal_cff_font_matrix(font_dict) if font_dict is not None else None

        if top_matrix is None:
            return font_dict_matrix or internal_DEFAULT_CFF_FONT_MATRIX
        if font_dict_matrix is None:
            return top_matrix
        return internal_compose_cff_matrices(top_matrix, font_dict_matrix)

    def internal_normalize_contours(
        self,
        glyph_id: int,
        contours: tuple[tuple[tuple[float, float], ...], ...],
    ) -> tuple[tuple[tuple[float, float], ...], ...]:
        matrix = self.internal_font_matrix(glyph_id)
        if matrix == internal_DEFAULT_CFF_FONT_MATRIX:
            return contours
        a, b, c, d, e, f = matrix
        return tuple(
            tuple(
                (
                    (x * a + y * c + e) * 1000.0,
                    (x * b + y * d + f) * 1000.0,
                )
                for x, y in contour
            )
            for contour in contours
        )

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
                collect_contours=True,
            )
            contours.extend(
                tuple((x + offset_x, y + offset_y) for x, y in contour)
                for contour in component_contours
            )
        return tuple(contours)

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
            contours, ignored_bbox = internal_type2_glyph_geometry_impl(
                charstring,
                local_subrs=self.local_subrs_for_glyph(glyph_id),
                global_subrs=self.global_subrs,
                collect_contours=True,
                seac_resolver=self.internal_seac_contours,
            )
            normalized = self.internal_normalize_contours(
                glyph_id, tuple(tuple(contour) for contour in contours)
            )
            geometry = (normalized, internal_contours_bbox(normalized))
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

    def glyph_contours_for_gid(self, glyph_id: int) -> tuple[tuple[tuple[float, float], ...], ...]:
        """Return the Type 2 outline normalized into PDF's 1000-unit glyph space."""
        return self.internal_glyph_geometry_for_gid(glyph_id)[0]

    def normalized_glyph_contours(
        self, glyph_id: int
    ) -> tuple[tuple[tuple[float, float], ...], ...]:
        """Return contours through the shared embedded-program geometry contract."""
        return self.glyph_contours_for_gid(glyph_id)


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
        if cell_x < 0:
            cell_x = 0
        elif cell_x > 17:
            cell_x = 17
        if cell_y < 0:
            cell_y = 0
        elif cell_y > 23:
            cell_y = 23
        add_cell((cell_x, cell_y))
    bitmap = rasterize_contours(contours, width=18, height=24)
    return CFFGlyphFeature(tuple(sorted(cells)), round(width / height, 2), len(contours), bitmap)


def internal_cubic_extrema_times(p0: float, p1: float, p2: float, p3: float) -> tuple[float, ...]:
    """Return the interior extrema parameters of one cubic coordinate."""
    a = -p0 + 3.0 * p1 - 3.0 * p2 + p3
    b = 2.0 * (p0 - 2.0 * p1 + p2)
    c = p1 - p0
    epsilon = 1e-12
    if abs(a) <= epsilon:
        if abs(b) <= epsilon:
            return ()
        root = -c / b
        return (root,) if 0.0 < root < 1.0 else ()
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return ()
    root_delta = sqrt(discriminant)
    roots = ((-b - root_delta) / (2.0 * a), (-b + root_delta) / (2.0 * a))
    return tuple(dict.fromkeys(root for root in roots if 0.0 < root < 1.0))


def internal_cubic_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    mt = 1.0 - t
    return (
        mt**3 * p0[0] + 3.0 * mt * mt * t * p1[0] + 3.0 * mt * t * t * p2[0] + t**3 * p3[0],
        mt**3 * p0[1] + 3.0 * mt * mt * t * p1[1] + 3.0 * mt * t * t * p2[1] + t**3 * p3[1],
    )


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
        *internal_cubic_extrema_times(p0[0], p1[0], p2[0], p3[0]),
        *internal_cubic_extrema_times(p0[1], p1[1], p2[1], p3[1]),
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


def internal_execute_type2_flex(
    operator: int,
    operands: list[float],
    curve: Callable[[float, float, float, float, float, float], None],
) -> None:
    """Execute one of the four escaped Type 2 flex operators."""
    match operator:
        case 34:  # hflex
            dx1, dx2, dy2, dx3, dx4, dx5, dx6 = operands
            curve(dx1, 0.0, dx2, dy2, dx3, 0.0)
            curve(dx4, 0.0, dx5, -dy2, dx6, 0.0)
        case 35:  # flex
            (
                dx1,
                dy1,
                dx2,
                dy2,
                dx3,
                dy3,
                dx4,
                dy4,
                dx5,
                dy5,
                dx6,
                dy6,
                ignored_flex_depth,
            ) = operands
            curve(dx1, dy1, dx2, dy2, dx3, dy3)
            curve(dx4, dy4, dx5, dy5, dx6, dy6)
        case 36:  # hflex1
            dx1, dy1, dx2, dy2, dx3, dx4, dx5, dy5, dx6 = operands
            dy6 = -(dy1 + dy2 + dy5)
            curve(dx1, dy1, dx2, dy2, dx3, 0.0)
            curve(dx4, 0.0, dx5, dy5, dx6, dy6)
        case 37:  # flex1
            dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, dx5, dy5, d6 = operands
            dx = dx1 + dx2 + dx3 + dx4 + dx5
            dy = dy1 + dy2 + dy3 + dy4 + dy5
            if abs(dx) > abs(dy):
                dx6, dy6 = d6, -dy
            else:
                dx6, dy6 = -dx, d6
            curve(dx1, dy1, dx2, dy2, dx3, dy3)
            curve(dx4, dy4, dx5, dy5, dx6, dy6)
        case _:
            raise ValueError("invalid Type 2 flex operator")


def internal_type2_glyph_geometry_impl(  # noqa: C901 - direct dispatch mirrors Type 2's spec table
    charstring: bytes,
    *,
    local_subrs: tuple[bytes, ...],
    global_subrs: tuple[bytes, ...],
    collect_contours: bool,
    seac_resolver: (
        Callable[
            [int, int, float, float],
            tuple[tuple[tuple[float, float], ...], ...],
        ]
        | None
    ) = None,
) -> tuple[list[list[tuple[float, float]]], tuple[float, float, float, float] | None]:
    stack: list[float] = []
    transient = [0.0] * internal_TYPE2_TRANSIENT_SIZE
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
    width_resolved = False
    random_state = internal_TYPE2_RANDOM_INITIAL_STATE
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

    def append_completed_contour(points: tuple[tuple[float, float], ...]) -> None:
        nonlocal bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, bbox_has_points
        if not points:
            return
        if collect_contours:
            contours.append(list(points))
        bbox_min_x = min(bbox_min_x, *(point[0] for point in points))
        bbox_min_y = min(bbox_min_y, *(point[1] for point in points))
        bbox_max_x = max(bbox_max_x, *(point[0] for point in points))
        bbox_max_y = max(bbox_max_y, *(point[1] for point in points))
        bbox_has_points = True

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
        point0 = (x, y)
        point1 = (x + dx1, y + dy1)
        point2 = (point1[0] + dx2, point1[1] + dy2)
        point3 = (point2[0] + dx3, point2[1] + dy3)
        for t in internal_cubic_sample_times(point0, point1, point2, point3):
            record_point(*internal_cubic_point(point0, point1, point2, point3, t))
        x, y = point3

    def clear_stack() -> None:
        del stack[:]

    def push(value: float) -> None:
        if len(stack) >= internal_TYPE2_MAX_STACK or not isfinite(value):
            raise ValueError("invalid Type 2 operand stack")
        stack.append(float(value))

    def require_integer(value: float) -> int:
        integer = int(value)
        if value != integer:
            raise ValueError("Type 2 operator requires an integer")
        return integer

    def pop_integer() -> int:
        return require_integer(stack.pop())

    def execute_escaped_operator(operator: int) -> None:
        nonlocal random_state
        match operator:
            case 0:  # dotsection -- deprecated no-op with a clearing stack contract
                clear_stack()
            case 3:  # and
                second = stack.pop()
                first = stack.pop()
                push(float(first != 0.0 and second != 0.0))
            case 4:  # or
                second = stack.pop()
                first = stack.pop()
                push(float(first != 0.0 or second != 0.0))
            case 5:  # not
                push(float(stack.pop() == 0.0))
            case 9:  # abs
                push(abs(stack.pop()))
            case 10:  # add
                second = stack.pop()
                push(stack.pop() + second)
            case 11:  # sub
                second = stack.pop()
                push(stack.pop() - second)
            case 12:  # div
                second = stack.pop()
                push(stack.pop() / second)
            case 14:  # neg
                push(-stack.pop())
            case 15:  # eq
                second = stack.pop()
                push(float(stack.pop() == second))
            case 18:  # drop
                stack.pop()
            case 20:  # put
                index = pop_integer()
                value = stack.pop()
                if not 0 <= index < len(transient):
                    raise ValueError("invalid Type 2 transient-array index")
                transient[index] = value
            case 21:  # get
                index = pop_integer()
                if not 0 <= index < len(transient):
                    raise ValueError("invalid Type 2 transient-array index")
                push(transient[index])
            case 22:  # ifelse
                value2 = stack.pop()
                value1 = stack.pop()
                choice2 = stack.pop()
                choice1 = stack.pop()
                push(choice1 if value1 <= value2 else choice2)
            case 23:  # random
                random_state = (1103515245 * random_state + 12345) & 0x7FFFFFFF
                push((random_state + 1) / 0x80000000)
            case 24:  # mul
                second = stack.pop()
                push(stack.pop() * second)
            case 26:  # sqrt
                push(sqrt(stack.pop()))
            case 27:  # dup
                push(stack[-1])
            case 28:  # exch
                stack[-1], stack[-2] = stack[-2], stack[-1]
            case 29:  # index
                index = max(pop_integer(), 0)
                if index >= len(stack):
                    raise ValueError("invalid Type 2 stack index")
                push(stack[-index - 1])
            case 30:  # roll
                shift = pop_integer()
                count = pop_integer()
                if count < 0 or count > len(stack):
                    raise ValueError("invalid Type 2 roll count")
                if count:
                    shift %= count
                    if shift:
                        values = stack[-count:]
                        stack[-count:] = values[-shift:] + values[:-shift]
            case 34 | 35 | 36 | 37:  # hflex / flex / hflex1 / flex1
                if not current_has_points:
                    raise ValueError("Type 2 flex operator has no current point")
                internal_execute_type2_flex(operator, stack, curve)
                clear_stack()
            case _:
                raise ValueError("unsupported Type 2 escaped operator")

    def execute(  # noqa: C901 - keeping operator cases together makes the bytecode contract auditable
        program: bytes, depth: int = 0
    ) -> bool:
        """Interpret one Type 2 charstring, appending to the enclosing contour state.

        Returns whether the caller still owns an unflushed contour: ``True`` when the
        program simply ran out (so the caller must call ``flush_contour``), ``False``
        when ``endchar`` already flushed it or the charstring was malformed. A
        subroutine returning ``False`` aborts its caller the same way.

        The branches below are keyed by raw Type 2 operator bytes; each carries the
        operator's spec name. Operands are values above 31, plus 28 (a two-byte
        integer) and 255 (a 16.16 fixed-point number).
        """
        nonlocal stem_count, width_resolved
        if depth > TYPE2_MAX_SUBR_DEPTH:
            return False
        pos = 0
        try:
            while pos < len(program):
                byte = program[pos]
                if byte > 31 or byte in {28, 255}:
                    value, pos = CFFFont.internal_parse_number(program, pos)
                    push(value)
                    continue
                pos += 1
                match byte:
                    case 1 | 3 | 18 | 23:  # hstem, vstem, hstemhm, vstemhm
                        operand_count = len(stack)
                        if not width_resolved and operand_count % 2:
                            operand_count -= 1
                        if operand_count < 2 or operand_count % 2:
                            return False
                        stem_count += operand_count // 2
                        if stem_count > 96:
                            return False
                        width_resolved = True
                        clear_stack()
                    case 4:  # vmoveto
                        if len(stack) == 1:
                            dy = stack[0]
                        elif not width_resolved and len(stack) == 2:
                            dy = stack[1]
                        else:
                            return False
                        width_resolved = True
                        move(0.0, dy)
                        clear_stack()
                    case 5:  # rlineto
                        if not current_has_points or len(stack) < 2 or len(stack) % 2:
                            return False
                        for i in range(0, len(stack) - 1, 2):
                            line(stack[i], stack[i + 1])
                        clear_stack()
                    case 6:  # hlineto -- alternates horizontal/vertical
                        if not current_has_points or not stack:
                            return False
                        horizontal = True
                        for value in stack:
                            line(value, 0.0) if horizontal else line(0.0, value)
                            horizontal = not horizontal
                        clear_stack()
                    case 7:  # vlineto -- alternates vertical/horizontal
                        if not current_has_points or not stack:
                            return False
                        vertical = True
                        for value in stack:
                            line(0.0, value) if vertical else line(value, 0.0)
                            vertical = not vertical
                        clear_stack()
                    case 8:  # rrcurveto
                        if not current_has_points or len(stack) < 6 or len(stack) % 6:
                            return False
                        for i in range(0, len(stack) - 5, 6):
                            curve(*stack[i : i + 6])
                        clear_stack()
                    case 10:  # callsubr (local)
                        if not stack:
                            return False
                        subr_index = pop_integer() + subr_bias
                        if not 0 <= subr_index < len(local_subrs) or not execute(
                            local_subrs[subr_index], depth + 1
                        ):
                            return False
                    case 11:  # return -- leave this subroutine, caller keeps going
                        return True
                    case 12:  # two-byte escaped operator
                        if pos >= len(program):
                            return False
                        escaped_operator = program[pos]
                        pos += 1
                        execute_escaped_operator(escaped_operator)
                    case 14:  # endchar -- glyph complete
                        arguments = list(stack)
                        if not width_resolved:
                            if len(arguments) in {1, 5}:
                                arguments = arguments[1:]
                            elif len(arguments) not in {0, 4}:
                                return False
                            width_resolved = True
                        elif len(arguments) not in {0, 4}:
                            return False
                        clear_stack()
                        flush_contour()
                        if arguments and seac_resolver is not None:
                            base_code = require_integer(arguments[2])
                            accent_code = require_integer(arguments[3])
                            for component in seac_resolver(
                                base_code,
                                accent_code,
                                arguments[0],
                                arguments[1],
                            ):
                                append_completed_contour(component)
                        return False
                    case 19 | 20:  # hintmask, cntrmask -- skip trailing mask bytes
                        operand_count = len(stack)
                        if not width_resolved and operand_count % 2:
                            operand_count -= 1
                        if operand_count % 2:
                            return False
                        stem_count += operand_count // 2
                        if stem_count <= 0 or stem_count > 96:
                            return False
                        mask_bytes = (stem_count + 7) // 8
                        if pos + mask_bytes > len(program):
                            return False
                        width_resolved = True
                        clear_stack()
                        pos += mask_bytes
                    case 21:  # rmoveto
                        if len(stack) == 2:
                            dx, dy = stack
                        elif not width_resolved and len(stack) == 3:
                            dx, dy = stack[1:]
                        else:
                            return False
                        width_resolved = True
                        move(dx, dy)
                        clear_stack()
                    case 22:  # hmoveto
                        if len(stack) == 1:
                            dx = stack[0]
                        elif not width_resolved and len(stack) == 2:
                            dx = stack[1]
                        else:
                            return False
                        width_resolved = True
                        move(dx, 0.0)
                        clear_stack()
                    case 24:  # rcurveline -- curves followed by exactly one line
                        if not current_has_points or len(stack) < 8 or (len(stack) - 2) % 6:
                            return False
                        curve_args = stack[:-2]
                        for i in range(0, len(curve_args) - 5, 6):
                            curve(*curve_args[i : i + 6])
                        line(stack[-2], stack[-1])
                        clear_stack()
                    case 25:  # rlinecurve -- lines followed by exactly one curve
                        if not current_has_points or len(stack) < 8 or (len(stack) - 6) % 2:
                            return False
                        line_args = stack[:-6]
                        for i in range(0, len(line_args) - 1, 2):
                            line(line_args[i], line_args[i + 1])
                        curve(*stack[-6:])
                        clear_stack()
                    case 26:  # vvcurveto
                        if not current_has_points or len(stack) < 4 or len(stack) % 4 not in {0, 1}:
                            return False
                        dx1 = stack.pop(0) if len(stack) % 2 else 0.0
                        for i in range(0, len(stack) - 3, 4):
                            curve(
                                dx1,
                                stack[i],
                                stack[i + 1],
                                stack[i + 2],
                                0.0,
                                stack[i + 3],
                            )
                            dx1 = 0.0
                        clear_stack()
                    case 27:  # hhcurveto
                        if not current_has_points or len(stack) < 4 or len(stack) % 4 not in {0, 1}:
                            return False
                        dy1 = stack.pop(0) if len(stack) % 2 else 0.0
                        for i in range(0, len(stack) - 3, 4):
                            curve(
                                stack[i],
                                dy1,
                                stack[i + 1],
                                stack[i + 2],
                                stack[i + 3],
                                0.0,
                            )
                            dy1 = 0.0
                        clear_stack()
                    case 29:  # callgsubr (global)
                        if not stack:
                            return False
                        subr_index = pop_integer() + gsubr_bias
                        if not 0 <= subr_index < len(global_subrs) or not execute(
                            global_subrs[subr_index], depth + 1
                        ):
                            return False
                    case 30 | 31:  # vhcurveto / hvcurveto -- alternating tangents
                        if not current_has_points or len(stack) < 4 or len(stack) % 4 not in {0, 1}:
                            return False
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
                    case _:
                        return False
            return True
        except (ArithmeticError, IndexError, ValueError):
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
        "internal_candidate_arrays",
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
        self.internal_candidate_arrays: FeatureArrays | None = None
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
            if self.internal_candidate_arrays is None:
                self.internal_candidate_arrays = internal_feature_arrays(
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
                internal_right_arrays=self.internal_candidate_arrays,
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


__all__ = (
    "STANDARD_GLYPH_SIDS",
    "CFFFont",
    "CFFGlyphFeature",
    "CFFUnicodeRepairIndex",
    "cff_font_for_data",
    "cff_unicode_repair_index_for_data",
    "glyph_feature_distance",
    "is_repairable_to_unicode_label",
)
