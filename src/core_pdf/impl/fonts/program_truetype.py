from __future__ import annotations

import struct
import threading
from collections.abc import Callable
from typing import Any

lazy import numpy

from core_pdf.impl.adobe_fonts.cff.charstrings import cubic_point
from core_pdf.impl.base.caches import BoundedDict
from core_pdf.impl.base.geometry import transform_bbox
from core_pdf.impl.base.types import Rectangle
from core_pdf.impl.fonts.helpers import glyph_name_to_unicode
from core_pdf.impl.fonts.opentype_cff import (
    CffGlyphs,
    cff2_glyphs,
    cff_glyphs,
    opentype_glyph_order,
)
from core_pdf.impl.fonts.outline_pens import BoundsSink, draw_decomposed
from core_pdf.impl.fonts.program_base import GlyphNaming, GlyphProgram
from core_pdf.impl.fonts.raster_kernel import Contours, Point, scale_contours
from core_pdf.impl.fonts.sfnt import (
    GlyphOutlines,
    SfntFont,
    Table,
    best_cmap,
    cmap_subtable,
    glyf_glyph,
    placeholder_glyph_id,
)
from core_pdf.impl.spec.s_09_fonts.helpers import is_unicode_scalar, symbol_character_code
lazy from core_pdf_cythonized import truetype_contours


def parse_truetype_program(data: bytes) -> SfntFont:
    font = SfntFont(data)
    if not {"maxp", "glyf", "loca", "head"} <= font.keys():
        raise ValueError("invalid TrueType glyph tables")
    return font


def glyph_bbox(glyf: Table, glyph_name: str) -> tuple[float, float, float, float] | None:
    glyph = glyf_glyph(glyf, glyph_name)
    if glyph.numberOfContours == 0:
        return None
    return (float(glyph.xMin), float(glyph.yMin), float(glyph.xMax), float(glyph.yMax))


GLYPH_HEADER = struct.Struct(">hhhhh")


def raw_glyph_locations(font: SfntFont) -> tuple[Any, bytes | None]:
    try:
        locations = font.table("loca").locations
        glyph_data = font.table_data("glyf")
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


type TrueTypeTables = tuple[bytes, numpy.ndarray[Any, Any], numpy.ndarray[Any, Any], int]


def truetype_tables(font: Any, glyph_data: bytes | None = None) -> TrueTypeTables | None:
    try:
        keys = font.keys()
        if (
            "CFF " in keys
            or "CFF2" in keys
            or "fvar" in keys
            or not {"glyf", "loca", "hmtx"} <= keys
        ):
            return None
        glyf = font.table_data("glyf") if glyph_data is None else glyph_data
        loca = numpy.asarray(font.table("loca").locations, dtype=numpy.int64)
        order = font.glyph_order
        if len(set(order)) != len(order):
            return None
        metrics = font.table("hmtx").metrics
        lsb = numpy.asarray([int(metrics[name][1]) for name in order], dtype=numpy.int64)
        if "vmtx" in keys:
            vertical = font.table("vmtx").metrics
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


class SfntOutlineAccess:
    """A TrueType program's glyphs by ID: drawn by the compiled kernel from the raw
    tables, or, where the kernel cannot read them, glyph by glyph in Python."""

    __slots__ = (
        "font",
        "glyph_count",
        "glyph_data",
        "outline_sets",
        "scale",
        "truetype",
        "truetype_read",
    )

    def __init__(self, font: SfntFont, glyph_data: bytes | None = None) -> None:
        self.font = font
        self.glyph_data = glyph_data
        self.glyph_count = len(font.glyph_order)
        units_per_em = float(getattr(font.table("head"), "unitsPerEm", 1000) or 1000)
        self.scale = 1000.0 / units_per_em if units_per_em else 1.0
        self.truetype: TrueTypeTables | None = None
        self.truetype_read = False
        self.outline_sets: dict[int, GlyphOutlines] = {}

    def glyph_id_for_name(self, glyph_name: str) -> int | None:
        return self.font.glyph_id_for_name(glyph_name)

    def has_glyph_id(self, glyph_id: int) -> bool:
        return 0 <= glyph_id < self.glyph_count

    def outlines(self) -> GlyphOutlines:
        # Drawing tracks component depth, so each thread draws with its own.
        thread = threading.get_ident()
        outlines = self.outline_sets.get(thread)
        if outlines is None:
            outlines = self.outline_sets[thread] = GlyphOutlines(self.font)
        return outlines

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
            recording = self.outlines().record(self.font.glyph_name(glyph_id))
            contours = tuple(tuple(contour) for contour in recording_to_contours(recording))
            return contours if self.scale == 1.0 else scale_contours(contours, self.scale)
        except Exception:
            return ()


class UnicodeGlyphMaps:
    """A font's Unicode cmap and its inverse, read on first use and shared by the
    variants of one program: a CID font with a ToUnicode never reads either."""

    __slots__ = ("font", "cmap", "inverse")

    def __init__(self, font: SfntFont) -> None:
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


class TrueTypeFontProgram(GlyphProgram):
    __slots__ = (
        "font",
        "outlines",
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
        self, data: bytes, cid_to_gid: bytes | None = None, *, use_cmap: bool = False
    ) -> None:
        self.data = data
        self.composite_bbox_cache: dict[
            int, tuple[tuple[float, float, float, float] | None, bool]
        ] = {}
        self.font = tt_font_from_data(data)
        self.font.resolve_glyph_order()
        self.units_per_em = float(getattr(self.font.table("head"), "unitsPerEm", 1000) or 1000)
        self.glyph_locations, glyph_data = raw_glyph_locations(self.font)
        self.glyph_table_data = glyph_data if glyph_data is not None else b""
        self.outlines = SfntOutlineAccess(self.font, glyph_data)
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
        for name in TrueTypeFontProgram.__slots__:
            setattr(variant, name, getattr(self, name))
        variant.cid_to_gid = cid_to_gid
        variant.cmap = self.selected_cmap(use_cmap=use_cmap)
        return variant

    def glyph_id_for_name(self, glyph_name: str) -> int | None:
        return self.outlines.glyph_id_for_name(glyph_name)

    def has_glyph_id(self, glyph_id: int) -> bool:
        return self.outlines.has_glyph_id(glyph_id)

    def normalized_glyph_contours(self, glyph_id: int) -> Contours:
        return self.outlines.normalized_glyph_contours(glyph_id)

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
            glyph_name = self.font.glyph_name(gid)
            glyf = self.font.table("glyf")
            glyph = glyf_glyph(glyf, glyph_name)
            if not glyph.is_composite():
                return (None, False)
            body_bbox: tuple[float, float, float, float] | None = None
            has_dot = False
            for component in glyph.components:
                component_name = component.name
                transform = component.transform()
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


def tt_font_from_data(data: bytes) -> SfntFont:
    try:
        return parse_truetype_program(data)
    except Exception as exc:
        raise ValueError("invalid TrueType font program") from exc


def cmap_glyph_ids(font: SfntFont) -> Callable[[str], int | None]:
    """Glyph ID for a glyph name a cmap subtable produced, or None when the
    font neither names it nor gives it a placeholder name."""
    if font.placeholder:
        return placeholder_glyph_id
    reverse_glyph_map = font.glyph_id_map()

    def gid_for(glyph_name: str) -> int | None:
        try:
            return reverse_glyph_map[glyph_name]
        except KeyError:
            return placeholder_glyph_id(glyph_name)

    return gid_for


def best_unicode_gid_cmap(font: SfntFont) -> dict[int, int]:
    symbol_fallback = False
    try:
        cmap_table = font.table("cmap")
        name_cmap = best_cmap(cmap_table)
        if name_cmap is None:
            symbol_cmap = cmap_subtable(cmap_table, 3, 0)
            name_cmap = symbol_cmap.mapping() if symbol_cmap is not None else {}
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


def code_gid_cmap(font: SfntFont) -> dict[int, int]:
    try:
        cmap_table = font.table("cmap")
        gid_for = cmap_glyph_ids(font)
    except Exception:
        return {}

    for platform, encoding in ((3, 0), (1, 0)):
        try:
            subtable = cmap_subtable(cmap_table, platform, encoding)
        except Exception:
            continue
        if subtable is None:
            continue
        mapping: dict[int, int] = {}
        for code, glyph_name in subtable.mapping().items():
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


def recording_to_contours(recording: list[tuple[str, tuple[Any, ...]]]) -> list[list[Point]]:
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
    contour: list[Point], current: Point, start: Point | None, operands: tuple[Any, ...]
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


def parse_opentype_program(data: bytes) -> SfntFont:
    font = SfntFont(data)
    if not ({"CFF ", "CFF2"} & font.keys()):
        raise ValueError("OpenType font has no CFF outline table")
    return font


class OpenTypeFontProgram(GlyphProgram):
    """An OpenType font program with CFF or CFF2 outlines. A CFF2 table is read on
    first draw, and a draw fails while it cannot be read."""

    __slots__ = ("font", "glyph_count", "scale", "cff")

    def __init__(self, data: bytes) -> None:
        try:
            font = self.font = parse_opentype_program(data)
            self.cff: CffGlyphs | None = None
            if "CFF " in font:
                self.cff = cff_glyphs(font.table_data("CFF "))
                font.glyph_order = self.cff.names
            else:
                font.glyph_order = opentype_glyph_order(font)
            self.glyph_count = len(font.glyph_order)
            units_per_em = float(getattr(font.table("head"), "unitsPerEm", 1000) or 1000)
            self.scale = 1000.0 / units_per_em if units_per_em else 1.0
        except Exception as exc:
            raise ValueError("invalid OpenType CFF font program") from exc

    def glyphs(self) -> CffGlyphs:
        cff = self.cff
        if cff is None:
            cff = self.cff = cff2_glyphs(self.font.table_data("CFF2"), self.font.glyph_order)
        return cff

    def draw(
        self,
        glyph_id: int,
        transform: tuple[float, ...] | None,
        emit: Callable[[tuple[str, tuple[Any, ...]]], None],
    ) -> None:
        cff = self.glyphs()
        font = self.font
        metrics = font.table("hmtx").metrics
        vertical_table = font.optional_table("vmtx")
        vertical = getattr(vertical_table, "metrics", None) if vertical_table is not None else None

        def check_metrics(name: str) -> None:
            metrics[name]  # noqa: B018 - a glyph without metrics fails
            if vertical is not None:
                vertical[name]  # noqa: B018

        def has_glyph(name: str) -> bool:
            try:
                check_metrics(name)
            except KeyError:
                return False
            return True

        name = font.glyph_name(glyph_id)
        check_metrics(name)
        draw_decomposed(cff.record, name, transform, emit, has_glyph)

    def glyph_id_for_name(self, glyph_name: str) -> int | None:
        return self.font.glyph_id_for_name(glyph_name)

    def has_glyph_id(self, glyph_id: int) -> bool:
        return 0 <= glyph_id < self.glyph_count

    def normalized_glyph_contours(self, glyph_id: int) -> Contours:
        try:
            recording: list[tuple[str, tuple[Any, ...]]] = []
            self.draw(glyph_id, None, recording.append)
            contours = tuple(tuple(contour) for contour in recording_to_contours(recording))
            return contours if self.scale == 1.0 else scale_contours(contours, self.scale)
        except Exception:
            return ()

    def glyph_bbox_for_gid(self, glyph_id: int) -> Rectangle | None:
        try:
            sink = BoundsSink()
            self.draw(glyph_id, (self.scale, 0.0, 0.0, self.scale, 0.0, 0.0), sink)
        except Exception:
            return None
        if sink.bounds is None:
            return None
        x_min, y_min, x_max, y_max = sink.bounds
        return float(x_min), float(y_min), float(x_max), float(y_max)

    def glyph_id_for_code(self, code: int, naming: GlyphNaming) -> int | None:
        if naming.is_cid_font:
            return code
        return self.glyph_id_for_name(naming.glyph_name(code))
