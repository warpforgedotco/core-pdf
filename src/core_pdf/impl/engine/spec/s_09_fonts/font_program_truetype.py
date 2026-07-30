from __future__ import annotations

import logging
import struct
from functools import lru_cache
from io import BytesIO
from typing import Any

from core_pdf._vendor.fontTools.pens.recordingPen import (
    DecomposingRecordingPen,
)
from core_pdf._vendor.fontTools.ttLib import TTFont, TTLibError
from core_pdf.impl.engine.spec.s_09_fonts.raster_kernel import Point, rasterize_contours


class internal_RecoverableFontTableWarningFilter(logging.Filter):
    """Hide fontTools warnings for malformed fields it already repairs safely."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            message.endswith("extra bytes in post.stringData array")
            or message.endswith(" timestamp seems very low; regarding as unix timestamp")
        )


internal_FONT_TABLE_WARNING_FILTER = internal_RecoverableFontTableWarningFilter()
logging.getLogger("fontTools.ttLib.tables._p_o_s_t").addFilter(internal_FONT_TABLE_WARNING_FILTER)
logging.getLogger("fontTools.ttLib.tables._h_e_a_d").addFilter(internal_FONT_TABLE_WARNING_FILTER)


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
        "internal_glyph_set",
        "internal_glyph_contour_cache",
        "internal_glyph_bitmap_cache",
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
        self.units_per_em = float(getattr(self.font["head"], "unitsPerEm", 1000) or 1000)
        self.cid_to_gid = cid_to_gid
        self.unicode_cmap = internal_best_unicode_gid_cmap(self.font)
        self.glyph_to_unicode = internal_invert_unicode_cmap(self.unicode_cmap)
        self.cmap = self.unicode_cmap if use_cmap else {}
        self.internal_glyph_set: Any | None = None
        self.internal_glyph_contour_cache: dict[int, tuple[tuple[Point, ...], ...]] = {}
        self.internal_glyph_bitmap_cache: dict[tuple[int, int, int], tuple[int, ...]] = {}

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
        gid = self.glyph_id_for_code(code)
        cache_key = (gid, width, height)
        cached = self.internal_glyph_bitmap_cache.get(cache_key)
        if cached is not None:
            return cached
        contours = self.internal_glyph_contours_for_gid(gid)
        if not contours:
            self.internal_glyph_bitmap_cache[cache_key] = ()
            return ()
        bitmap = rasterize_contours(contours, width=width, height=height)
        if len(self.internal_glyph_bitmap_cache) >= 512:
            self.internal_glyph_bitmap_cache.clear()
        self.internal_glyph_bitmap_cache[cache_key] = bitmap
        return bitmap

    def glyph_bbox(self, code: int) -> tuple[float, float, float, float] | None:
        contours = self.internal_glyph_contours_for_gid(self.glyph_id_for_code(code))
        if not contours:
            return None
        points = [point for contour in contours for point in contour]
        if not points:
            return None
        scale = 1000.0 / self.units_per_em if self.units_per_em else 1.0
        xs = [point[0] * scale for point in points]
        ys = [point[1] * scale for point in points]
        return (min(xs), min(ys), max(xs), max(ys))

    def glyph_contours(self, gid: int) -> list[list[Point]]:
        return [list(contour) for contour in self.internal_glyph_contours_for_gid(gid)]

    def internal_glyph_contours_for_gid(self, gid: int) -> tuple[tuple[Point, ...], ...]:
        cached = self.internal_glyph_contour_cache.get(gid)
        if cached is not None:
            return cached
        try:
            glyph_name = self.font.getGlyphName(gid)
            glyph_set = self.internal_glyph_set
            if glyph_set is None:
                glyph_set = self.font.getGlyphSet()
                self.internal_glyph_set = glyph_set
            glyph = glyph_set[glyph_name]
            pen = DecomposingRecordingPen(  # type: ignore[call-arg]
                glyph_set, skipMissingComponents=True
            )
            glyph.draw(pen)
            contours = tuple(
                tuple(contour) for contour in internal_recording_to_contours(pen.value)
            )
        except (KeyError, TTLibError, AttributeError, IndexError, ValueError):
            contours = ()
        if len(self.internal_glyph_contour_cache) >= 512:
            self.internal_glyph_contour_cache.clear()
        self.internal_glyph_contour_cache[gid] = contours
        return contours

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
                xmin, ymin, xmax, ymax = internal_transform_bbox(bbox, transform)
                w, h = xmax - xmin, ymax - ymin
                if h > 0 and h < 600 and 0.4 < w / h < 2.5 and ymin > 900:
                    has_dot = True
                else:
                    body_bbox = (xmin, ymin, xmax, ymax)
            return (body_bbox, has_dot)
        except (KeyError, TTLibError, AttributeError, IndexError, ValueError):
            return (None, False)


def internal_tt_font_from_data(data: bytes) -> TTFont:
    try:
        return TTFont(BytesIO(data), lazy=True)
    except (TTLibError, OSError, struct.error, ValueError) as exc:
        raise ValueError("invalid TrueType font program") from exc


def internal_best_unicode_gid_cmap(font: TTFont) -> dict[int, int]:
    try:
        cmap_table = font["cmap"]
    except (KeyError, TTLibError):
        return {}
    name_cmap = cmap_table.getBestCmap()
    if name_cmap is None:
        symbol_cmap = cmap_table.getcmap(3, 0)
        name_cmap = symbol_cmap.cmap if symbol_cmap is not None else {}
    mapping: dict[int, int] = {}
    reverse_glyph_map = font.getReverseGlyphMap()
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
    return mapping


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


def internal_transform_bbox(
    bbox: tuple[float, float, float, float],
    transform: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox
    xx, xy, yx, yy, dx, dy = transform
    corners = (
        (xmin, ymin),
        (xmin, ymax),
        (xmax, ymin),
        (xmax, ymax),
    )
    transformed = [(x * xx + y * xy + dx, x * yx + y * yy + dy) for x, y in corners]
    xs = [point[0] for point in transformed]
    ys = [point[1] for point in transformed]
    return (min(xs), min(ys), max(xs), max(ys))


def internal_recording_to_contours(
    recording: list[tuple[str, tuple[Any, ...]]],
) -> list[list[Point]]:
    contours: list[list[Point]] = []
    contour: list[Point] = []
    current: Point | None = None
    start: Point | None = None
    for operator, operands in recording:
        if operator == "moveTo":
            if contour:
                contours.append(internal_close_contour(contour))
            start = internal_point(operands[0])
            current = start
            contour = [start]
        elif operator == "lineTo" and current is not None:
            current = internal_point(operands[0])
            contour.append(current)
        elif operator == "qCurveTo" and current is not None:
            current = internal_append_quadratic(contour, current, start, operands)
        elif operator == "curveTo" and current is not None:
            current = internal_append_cubic(contour, current, operands)
        elif operator in {"closePath", "endPath"}:
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


@lru_cache(maxsize=64)
def tt_font_for_data(
    data: bytes, cid_to_gid: bytes | None = None, *, use_cmap: bool = False
) -> TrueTypeFontProgram:
    return TrueTypeFontProgram(data, cid_to_gid, use_cmap=use_cmap)


__all__ = (
    "Point",
    "TrueTypeFontProgram",
    "rasterize_contours",
    "tt_font_for_data",
)
