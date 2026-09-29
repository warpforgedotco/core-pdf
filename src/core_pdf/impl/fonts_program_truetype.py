from __future__ import annotations

import logging
import struct
import threading
from collections.abc import Callable
from functools import partial
from io import BytesIO
from typing import Any

import numpy

from core_adobe_fonts.cff.charstrings import cubic_point
from core_pdf._vendor.fontTools.pens.boundsPen import BoundsPen
from core_pdf._vendor.fontTools.pens.recordingPen import (
    DecomposingRecordingPen,
)
from core_pdf._vendor.fontTools.pens.transformPen import TransformPen
from core_pdf._vendor.fontTools.ttLib import TTFont
from core_pdf.impl.caches import BoundedDict
from core_pdf.impl.fonts_helpers import glyph_name_to_unicode
from core_pdf.impl.fonts_program_base import GlyphNaming, GlyphProgram
from core_pdf.impl.fonts_raster_kernel import (
    Contours,
    Point,
    scale_contours,
)
from core_pdf.impl.geometry import transform_bbox
from core_pdf.impl.types import Rectangle
from core_pdf_cythonized import truetype_contours
from core_pdf_spec.s_09_fonts.font_program_truetype import (
    is_unicode_scalar,
    symbol_character_code,
)


def parse_truetype_program(data: bytes) -> TTFont:
    font = TTFont(BytesIO(data), lazy=True)
    if not {"maxp", "glyf", "loca", "head"} <= set(font.keys()):
        raise ValueError("invalid TrueType glyph tables")
    return font


def glyph_set_of(font: Any) -> Any:
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


def raw_glyph_locations(font: TTFont) -> tuple[Any, bytes | None]:
    try:
        locations = font["loca"]
        reader = font.reader
        glyph_data = bytes(reader["glyf"]) if reader is not None else b""
    except Exception:
        return (), None
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


TrueTypeTables = tuple[bytes, numpy.ndarray[Any, Any], numpy.ndarray[Any, Any], int]


def truetype_tables(font: TTFont, glyph_data: bytes | None = None) -> TrueTypeTables | None:
    try:
        keys = set(font.keys())
        if (
            "CFF " in keys
            or "CFF2" in keys
            or "fvar" in keys
            or not {"glyf", "loca", "hmtx"} <= keys
        ):
            return None
        reader = font.reader
        if reader is None:
            return None
        glyf = bytes(reader["glyf"]) if glyph_data is None else glyph_data
        loca = numpy.asarray(font["loca"].locations, dtype=numpy.int64)
        order = font.getGlyphOrder()
        if len(set(order)) != len(order):
            return None
        metrics = font["hmtx"].metrics
        lsb = numpy.asarray([int(metrics[name][1]) for name in order], dtype=numpy.int64)
        if "vmtx" in keys:
            vertical = font["vmtx"].metrics
            if not all(name in vertical for name in order):
                return None
    except Exception:
        return None
    if len(loca):
        starts = loca[:-1]
        ends = loca[1:]
        if bool(((ends < starts) | ((ends > len(glyf)) & (ends != starts))).any()):
            return None
    return glyf, loca, lsb, len(order)


class FontToolsOutlineAccess:
    __slots__ = (
        "font",
        "glyph_count",
        "glyph_data",
        "scale",
        "truetype",
        "truetype_read",
    )

    def __init__(self, font: TTFont, glyph_data: bytes | None = None) -> None:
        self.font = font
        self.glyph_data = glyph_data
        self.glyph_count = len(font.getGlyphOrder())
        units_per_em = float(getattr(font["head"], "unitsPerEm", 1000) or 1000)
        self.scale = 1000.0 / units_per_em if units_per_em else 1.0
        self.truetype: TrueTypeTables | None = None
        self.truetype_read = False

    def glyph_id_for_name(self, glyph_name: str) -> int | None:
        return self.font.getReverseGlyphMap().get(glyph_name)

    def has_glyph_id(self, glyph_id: int) -> bool:
        return 0 <= glyph_id < self.glyph_count

    def normalized_glyph_contours(self, glyph_id: int) -> tuple[tuple[Point, ...], ...]:
        if not self.truetype_read:
            self.truetype = truetype_tables(self.font, self.glyph_data)
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


class FontToolsProgram(GlyphProgram):
    __slots__ = ("font", "outlines")

    font: TTFont
    outlines: FontToolsOutlineAccess

    def glyph_id_for_name(self, glyph_name: str) -> int | None:
        return self.outlines.glyph_id_for_name(glyph_name)

    def has_glyph_id(self, glyph_id: int) -> bool:
        return self.outlines.has_glyph_id(glyph_id)

    def normalized_glyph_contours(self, glyph_id: int) -> Contours:
        return self.outlines.normalized_glyph_contours(glyph_id)

    def glyph_bbox_for_gid(self, glyph_id: int) -> Rectangle | None:
        return self.outlines.glyph_bbox_for_gid(glyph_id)


class UnicodeGlyphMaps:
    """A font's Unicode cmap and its inverse, read on first use and shared by the
    variants of one program: a CID font with a ToUnicode never reads either."""

    __slots__ = ("font", "cmap", "inverse")

    def __init__(self, font: TTFont) -> None:
        self.font = font
        self.cmap: dict[int, int] | None = None
        self.inverse: dict[int, str] | None = None

    def unicode_cmap(self) -> dict[int, int]:
        cmap = self.cmap
        if cmap is None:
            cmap = self.cmap = best_unicode_gid_cmap(self.font)
        return cmap

    def glyph_to_unicode(self) -> dict[int, str]:
        inverse = self.inverse
        if inverse is None:
            inverse = self.inverse = invert_unicode_cmap(self.unicode_cmap())
        return inverse


class TrueTypeFontProgram(FontToolsProgram):
    __slots__ = (
        "data",
        "units_per_em",
        "cid_to_gid",
        "cmap",
        "unicode_maps",
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
        ensure_glyph_order(self.font)
        self.units_per_em = float(getattr(self.font["head"], "unitsPerEm", 1000) or 1000)
        self.glyph_locations, glyph_data = raw_glyph_locations(self.font)
        self.glyph_table_data = glyph_data if glyph_data is not None else b""
        self.outlines = FontToolsOutlineAccess(self.font, glyph_data)
        self.cid_to_gid = cid_to_gid
        self.unicode_maps = UnicodeGlyphMaps(self.font)
        self.cmap = self.selected_cmap(use_cmap=use_cmap)

    @property
    def unicode_cmap(self) -> dict[int, int]:
        return self.unicode_maps.unicode_cmap()

    def selected_cmap(self, *, use_cmap: bool) -> dict[int, int]:
        return (self.unicode_cmap or code_gid_cmap(self.font)) if use_cmap else {}

    def variant(self, cid_to_gid: bytes | None, *, use_cmap: bool) -> TrueTypeFontProgram:
        variant = object.__new__(TrueTypeFontProgram)
        for name in (*FontToolsProgram.__slots__, *TrueTypeFontProgram.__slots__):
            setattr(variant, name, getattr(self, name))
        variant.cid_to_gid = cid_to_gid
        variant.cmap = self.selected_cmap(use_cmap=use_cmap)
        return variant

    def glyph_id_for_code(self, code: int, naming: GlyphNaming) -> int | None:
        if not naming.is_cid_font and 0 <= code < 256 and self.cmap:
            glyph_text = glyph_name_to_unicode(naming.glyph_name(code))
            if len(glyph_text) == 1:
                return self.glyph_id_for_unicode(ord(glyph_text))
        return self.mapped_glyph_id(code)

    def mapped_glyph_id(self, code: int) -> int:
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

    def unicode_for_gid(self, gid: int) -> str:
        return self.unicode_maps.glyph_to_unicode().get(gid, "")

    def code_bbox(self, code: int) -> Rectangle | None:
        return self.glyph_bbox_for_gid(self.mapped_glyph_id(code))

    def glyph_bbox_for_gid(self, glyph_id: int) -> Rectangle | None:
        bbox = glyph_header_bbox(self.glyph_locations, self.glyph_table_data, glyph_id)
        if bbox is None:
            return None
        scale = 1000.0 / self.units_per_em if self.units_per_em else 1.0
        if scale == 1.0:
            return bbox
        x0, y0, x1, y1 = bbox
        return (x0 * scale, y0 * scale, x1 * scale, y1 * scale)

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
    programs: BoundedDict[object, TrueTypeFontProgram] | None = getattr(
        program_cache, "programs", None
    )
    if programs is None:
        programs = program_cache.programs = BoundedDict(PROGRAM_CACHE_LIMIT)
    key: object = (data, cid_to_gid, use_cmap)
    program = programs.get(key)
    if program is None:
        base = programs.get(data)
        if base is None:
            base = programs.put(data, TrueTypeFontProgram(data))
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


PLACEHOLDER_GLYPH_NAMES = [".notdef"]


def placeholder_glyph_order(glyph_count: int) -> list[str]:
    global PLACEHOLDER_GLYPH_NAMES
    names = PLACEHOLDER_GLYPH_NAMES
    if glyph_count > len(names):
        names = PLACEHOLDER_GLYPH_NAMES = [
            *names,
            *("glyph%.5d" % gid for gid in range(len(names), glyph_count)),
        ]
    return names[:glyph_count] if glyph_count > 0 else []


def placeholder_glyph_names(font: TTFont) -> None:
    glyph_order = placeholder_glyph_order(int(font["maxp"].numGlyphs))
    glyph_order[0] = ".notdef"
    font.glyphOrder = glyph_order
    font.__dict__["_core_pdf_placeholder_names"] = True
    if "cmap" in font:
        font["cmap"].buildReversedMin()


def ensure_glyph_order(font: TTFont) -> None:
    # A TrueType program finds glyphs by ID, so when the post table names none,
    # fontTools' names synthesized from the cmap would only be re-mapped back to
    # IDs. The placeholders it starts from serve instead; the cmap is still read
    # where fontTools reads it, so a corrupt one fails and is kept the same way.
    overrides = font.__dict__
    overrides["_getGlyphNamesFromCmap"] = partial(placeholder_glyph_names, font)
    try:
        font.getGlyphOrder()
        return
    except Exception:
        pass
    finally:
        del overrides["_getGlyphNamesFromCmap"]
    try:
        glyph_count = int(font["maxp"].numGlyphs)
    except Exception as exc:
        raise ValueError("invalid TrueType glyph order") from exc
    if glyph_count <= 0:
        raise ValueError("invalid TrueType glyph order")
    font.setGlyphOrder(placeholder_glyph_order(glyph_count))
    font.__dict__["_core_pdf_placeholder_names"] = True


def cmap_glyph_ids(font: TTFont) -> Callable[[str], int | None]:
    """Glyph ID for a glyph name a cmap subtable produced, or None when the
    font neither names it nor gives it a placeholder name."""
    if font.__dict__.get("_core_pdf_placeholder_names"):
        return placeholder_glyph_id
    reverse_glyph_map = font.getReverseGlyphMap()

    def gid_for(glyph_name: str) -> int | None:
        try:
            return reverse_glyph_map[glyph_name]
        except KeyError:
            return placeholder_glyph_id(glyph_name)

    return gid_for


def placeholder_glyph_id(glyph_name: str) -> int | None:
    if glyph_name == ".notdef":
        return 0
    if not glyph_name.startswith("glyph"):
        return None
    try:
        return int(glyph_name[5:])
    except ValueError:
        return None


def best_unicode_gid_cmap(font: TTFont) -> dict[int, int]:
    symbol_fallback = False
    try:
        cmap_table = font["cmap"]
        name_cmap = cmap_table.getBestCmap()
        if name_cmap is None:
            symbol_cmap = cmap_table.getcmap(3, 0)
            name_cmap = symbol_cmap.cmap if symbol_cmap is not None else {}
            symbol_fallback = bool(name_cmap)
        gid_for = cmap_glyph_ids(font)
    except Exception:
        return {}
    mapping: dict[int, int] = {}
    for codepoint, glyph_name in name_cmap.items():
        if not is_unicode_scalar(codepoint):
            continue
        gid = gid_for(glyph_name)
        if gid is not None and gid > 0:
            mapping[codepoint] = gid
            if symbol_fallback and 0xF000 <= codepoint <= 0xF2FF:
                mapping.setdefault(symbol_character_code(codepoint), gid)
    return mapping


def code_gid_cmap(font: TTFont) -> dict[int, int]:
    try:
        cmap_table = font["cmap"]
        gid_for = cmap_glyph_ids(font)
    except Exception:
        return {}

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
            if gid is None or gid <= 0:
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
        previous = by_gid.setdefault(gid, char)
        if previous is not char and prefer_unicode_text(char, previous):
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
    return [cubic_point(p0, p1, p2, p3, i / segments) for i in range(1, segments + 1)]


def parse_opentype_program(data: bytes) -> TTFont:
    font = TTFont(BytesIO(data), lazy=True, recalcBBoxes=False, recalcTimestamp=False)
    if not ({"CFF ", "CFF2"} & set(font.keys())):
        raise ValueError("OpenType font has no CFF outline table")
    return font


class OpenTypeFontProgram(FontToolsProgram):
    __slots__ = ()

    def __init__(self, data: bytes) -> None:
        try:
            self.font = parse_opentype_program(data)
            self.outlines = FontToolsOutlineAccess(self.font)
            if "CFF2" in self.font and "fvar" in self.font:
                del self.font["fvar"]
        except Exception as exc:
            raise ValueError("invalid OpenType CFF font program") from exc

    def glyph_id_for_code(self, code: int, naming: GlyphNaming) -> int | None:
        if naming.is_cid_font:
            return code
        return self.glyph_id_for_name(naming.glyph_name(code))
