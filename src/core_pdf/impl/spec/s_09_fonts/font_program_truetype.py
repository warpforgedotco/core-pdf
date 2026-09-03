"""TrueType font program parsing: cmap, glyf outlines, and metrics."""

from __future__ import annotations

import logging
import struct
from io import BytesIO
from typing import Any

from core_pdf._vendor.fontTools.pens.boundsPen import BoundsPen
from core_pdf._vendor.fontTools.pens.recordingPen import (
    DecomposingRecordingPen,
)
from core_pdf._vendor.fontTools.pens.transformPen import TransformPen
from core_pdf._vendor.fontTools.ttLib import TTFont
from core_pdf.impl.model.geometry import transform_bbox
from core_pdf.impl.spec.s_09_fonts.raster_kernel import Point, rasterize_contours, scale_contours

# fontTools validates malformed tables with bare `assert` as well as by raising,
# and it decompiles lazily, so a damaged table surfaces late and as almost any
# exception type. Embedded font programs are untrusted input, so treat every
# failure from the parser as "this font program is unusable" instead of trying
# to enumerate what a corrupt one can raise.
FONT_PROGRAM_ERRORS = Exception


def internal_fonttools_contours(font: Any, glyph_id: int) -> tuple[tuple[Point, ...], ...]:
    glyph_name = font.getGlyphName(glyph_id)
    glyph_set = font.getGlyphSet()
    pen = DecomposingRecordingPen(glyph_set, skipMissingComponents=True)
    glyph_set[glyph_name].draw(pen)
    return tuple(tuple(contour) for contour in internal_recording_to_contours(pen.value))


def internal_fonttools_bbox(
    font: Any,
    glyph_id: int,
    scale: float,
) -> tuple[float, float, float, float] | None:
    glyph_name = font.getGlyphName(glyph_id)
    glyph_set = font.getGlyphSet()
    bounds_pen = BoundsPen(glyph_set)
    glyph_set[glyph_name].draw(TransformPen(bounds_pen, (scale, 0.0, 0.0, scale, 0.0, 0.0)))
    if bounds_pen.bounds is None:
        return None
    x_min, y_min, x_max, y_max = bounds_pen.bounds
    return float(x_min), float(y_min), float(x_max), float(y_max)


class internal_RecoverableFontTableWarningFilter(logging.Filter):
    """Hide fontTools warnings for malformed fields it already repairs safely."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            message.endswith("extra bytes in post.stringData array")
            or message.endswith(" timestamp seems very low; regarding as unix timestamp")
        )


internal_FONT_TABLE_WARNING_FILTER = internal_RecoverableFontTableWarningFilter()
for logger_name in (
    "fontTools.ttLib.tables._p_o_s_t",
    "fontTools.ttLib.tables._h_e_a_d",
    "core_pdf._vendor.fontTools.ttLib.tables._p_o_s_t",
    "core_pdf._vendor.fontTools.ttLib.tables._h_e_a_d",
):
    logging.getLogger(logger_name).addFilter(internal_FONT_TABLE_WARNING_FILTER)


def is_unicode_scalar(codepoint: int) -> bool:
    return 0 <= codepoint < 0x110000 and not 0xD800 <= codepoint <= 0xDFFF


class TrueTypeFontProgram:
    __slots__ = (
        "data",
        "font",
        "units_per_em",
        "cid_to_gid",
        "cmap",
        "unicode_cmap",
        "glyph_to_unicode",
    )

    def __init__(
        self,
        data: bytes,
        cid_to_gid: bytes | None = None,
        *,
        use_cmap: bool = False,
    ) -> None:
        self.data = data
        self.font = internal_tt_font_from_data(data)
        if not {"maxp", "glyf", "loca", "head"} <= set(self.font.keys()):
            raise ValueError("invalid TrueType glyph tables")
        internal_ensure_glyph_order(self.font)
        self.units_per_em = float(getattr(self.font["head"], "unitsPerEm", 1000) or 1000)
        self.cid_to_gid = cid_to_gid
        self.unicode_cmap = internal_best_unicode_gid_cmap(self.font)
        self.glyph_to_unicode = internal_invert_unicode_cmap(self.unicode_cmap)
        if use_cmap:
            self.cmap = self.unicode_cmap or internal_code_gid_cmap(self.font)
        else:
            self.cmap = {}

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
        """Resolve a simple-font character through the embedded TrueType cmap."""
        if self.cmap:
            return self.cmap.get(codepoint, 0)
        return 0

    def has_glyph_id(self, gid: int) -> bool:
        return 0 <= gid < len(self.font.getGlyphOrder())

    def unicode_for_gid(self, gid: int) -> str:
        return self.glyph_to_unicode.get(gid, "")

    def glyph_bitmap(self, code: int, *, width: int = 24, height: int = 32) -> tuple[int, ...]:
        return self.glyph_bitmap_for_gid(self.glyph_id_for_code(code), width=width, height=height)

    def glyph_bitmap_for_gid(
        self, gid: int, *, width: int = 24, height: int = 32
    ) -> tuple[int, ...]:
        contours = self.internal_glyph_contours_for_gid(gid)
        return rasterize_contours(contours, width=width, height=height) if contours else ()

    def glyph_bbox(self, code: int) -> tuple[float, float, float, float] | None:
        return self.glyph_bbox_for_gid(self.glyph_id_for_code(code))

    def glyph_bbox_for_gid(self, gid: int) -> tuple[float, float, float, float] | None:
        try:
            glyph_name = self.font.getGlyphName(gid)
            bbox = internal_glyph_bbox(self.font["glyf"], glyph_name)
        except FONT_PROGRAM_ERRORS:
            bbox = None
        if bbox is None:
            return None
        scale = 1000.0 / self.units_per_em if self.units_per_em else 1.0
        if scale == 1.0:
            return bbox
        x0, y0, x1, y1 = bbox
        return (x0 * scale, y0 * scale, x1 * scale, y1 * scale)

    def glyph_contours(self, gid: int) -> list[list[Point]]:
        return [list(contour) for contour in self.internal_glyph_contours_for_gid(gid)]

    def normalized_glyph_contours(self, gid: int) -> tuple[tuple[Point, ...], ...]:
        """Return an immutable outline in PDF's 1000-unit glyph space."""
        contours = self.internal_glyph_contours_for_gid(gid)
        if not contours:
            return ()
        scale = 1000.0 / self.units_per_em if self.units_per_em else 1.0
        return contours if scale == 1.0 else scale_contours(contours, scale)

    def internal_glyph_contours_for_gid(self, gid: int) -> tuple[tuple[Point, ...], ...]:
        try:
            return internal_fonttools_contours(self.font, gid)
        except FONT_PROGRAM_ERRORS:
            return ()

    def composite_body_bbox(
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
                bbox = internal_glyph_bbox(glyf, component_name)
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
        except FONT_PROGRAM_ERRORS:
            return (None, False)


def internal_tt_font_from_data(data: bytes) -> TTFont:
    try:
        return TTFont(BytesIO(data), lazy=True)
    except FONT_PROGRAM_ERRORS as exc:
        raise ValueError("invalid TrueType font program") from exc


def internal_ensure_glyph_order(font: TTFont) -> None:
    """Recover a stable glyph order when an embedded ``post`` table is corrupt."""
    try:
        font.getGlyphOrder()
        return
    except FONT_PROGRAM_ERRORS:
        pass
    try:
        glyph_count = int(font["maxp"].numGlyphs)
    except FONT_PROGRAM_ERRORS as exc:
        raise ValueError("invalid TrueType glyph order") from exc
    if glyph_count <= 0:
        raise ValueError("invalid TrueType glyph order")
    font.setGlyphOrder([".notdef", *(f"glyph{gid:05d}" for gid in range(1, glyph_count))])


def internal_best_unicode_gid_cmap(font: TTFont) -> dict[int, int]:
    symbol_fallback = False
    try:
        cmap_table = font["cmap"]
        name_cmap = cmap_table.getBestCmap()
        if name_cmap is None:
            symbol_cmap = cmap_table.getcmap(3, 0)
            name_cmap = symbol_cmap.cmap if symbol_cmap is not None else {}
            symbol_fallback = bool(name_cmap)
        reverse_glyph_map = font.getReverseGlyphMap()
    except FONT_PROGRAM_ERRORS:
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
                # ISO 32000-1 9.6.6.4: with a (3, 0) subtable the codes live in
                # 0x0000-0x00FF, 0xF000-0xF0FF, 0xF100-0xF1FF or 0xF200-0xF2FF,
                # and "each byte from the string shall be prepended with the
                # high byte of the range". Its keys are not Unicode scalars, so
                # register the single-byte code that selects each glyph;
                # otherwise every code missed and resolved to GID 0.
                mapping.setdefault(codepoint & 0xFF, gid)
    return mapping


def internal_code_gid_cmap(font: TTFont) -> dict[int, int]:
    """Map raw character codes to glyphs through non-Unicode cmap subtables.

    Simple TrueType fonts written by macOS carry only a Macintosh (1,0)
    subtable — often format 6 — and symbol fonts carry (3,0) with codes
    offset into the 0xF000 private-use range. Neither is a Unicode table,
    so the best-cmap lookup finds nothing and character codes would fall
    through as glyph ids, which draws garbage from a subset. Per the PDF
    text-showing rules for symbolic TrueType fonts, resolve the raw code
    through (3,0) and then (1,0) directly.
    """
    try:
        cmap_table = font["cmap"]
        reverse_glyph_map = font.getReverseGlyphMap()
    except FONT_PROGRAM_ERRORS:
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
        except FONT_PROGRAM_ERRORS:
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


def internal_invert_unicode_cmap(cmap: dict[int, int]) -> dict[int, str]:
    by_gid: dict[int, str] = {}
    for codepoint, gid in cmap.items():
        if gid <= 0 or not is_unicode_scalar(codepoint):
            continue
        char = chr(codepoint)
        previous = by_gid.get(gid)
        if previous is None or internal_prefer_unicode_text(char, previous):
            by_gid[gid] = char
    return by_gid


def internal_prefer_unicode_text(candidate: str, current: str) -> bool:
    candidate_score = internal_unicode_text_score(candidate)
    current_score = internal_unicode_text_score(current)
    if candidate_score != current_score:
        return candidate_score > current_score
    return ord(candidate) < ord(current)


def internal_unicode_text_score(char: str) -> int:
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


def internal_glyph_bbox(glyf: Any, glyph_name: str) -> tuple[float, float, float, float] | None:
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


def internal_recording_to_contours(
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
                    contours.append(internal_close_contour(contour))
                start = internal_point(operands[0])
                current = start
                contour = [start]
            case "lineTo" if current is not None:
                current = internal_point(operands[0])
                contour.append(current)
            case "qCurveTo" if current is not None:
                current = internal_append_quadratic(contour, current, start, operands)
            case "curveTo" if current is not None:
                current = internal_append_cubic(contour, current, operands)
            case "closePath" | "endPath":
                if contour:
                    contours.append(internal_close_contour(contour))
                    contour = []
                    current = None
                    start = None
    if contour:
        contours.append(internal_close_contour(contour))
    return [contour for contour in contours if len(contour) >= 3]


def internal_point(value: Any) -> Point:
    x, y = value
    return (float(x), float(y))


def internal_append_quadratic(
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
        controls = [internal_point(point) for point in points[:-1]]
        end = start
    else:
        controls = [internal_point(point) for point in points[:-1]]
        end = internal_point(points[-1])
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
        contour.extend(internal_flatten_quadratic(segment_start, control, segment_end))
        segment_start = segment_end
    return end


def internal_append_cubic(contour: list[Point], current: Point, operands: tuple[Any, ...]) -> Point:
    if len(operands) % 3:
        return current
    segment_start = current
    for index in range(0, len(operands), 3):
        c1 = internal_point(operands[index])
        c2 = internal_point(operands[index + 1])
        end = internal_point(operands[index + 2])
        contour.extend(internal_flatten_cubic(segment_start, c1, c2, end))
        segment_start = end
    return segment_start


def internal_close_contour(contour: list[Point]) -> list[Point]:
    if contour and contour[0] != contour[-1]:
        return [*contour, contour[0]]
    return contour


def internal_flatten_quadratic(p0: Point, p1: Point, p2: Point, segments: int = 6) -> list[Point]:
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


def internal_flatten_cubic(
    p0: Point, p1: Point, p2: Point, p3: Point, segments: int = 8
) -> list[Point]:
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


__all__ = (
    "Point",
    "TrueTypeFontProgram",
    "rasterize_contours",
)
