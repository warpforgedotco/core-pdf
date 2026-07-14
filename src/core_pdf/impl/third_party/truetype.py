from __future__ import annotations

import struct
from functools import lru_cache
from io import BytesIO
from math import ceil
from typing import Any

from core_pdf.impl.third_party.fontTools.pens.recordingPen import (
    DecomposingRecordingPen,
)
from core_pdf.impl.third_party.fontTools.ttLib import TTFont, TTLibError


Point = tuple[float, float]


class TrueTypeFontProgram:
    __slots__ = (
        "data",
        "font",
        "units_per_em",
        "cid_to_gid",
        "cmap",
        "unicode_cmap",
        "glyph_to_unicode",
        "_glyph_bitmap_cache",
    )

    def __init__(
        self,
        data: bytes,
        cid_to_gid: bytes | None = None,
        *,
        use_cmap: bool = False,
    ) -> None:
        self.data = data
        self.font = _tt_font_from_data(data)
        if not {"maxp", "glyf", "loca", "head"} <= set(self.font.keys()):
            raise ValueError("invalid TrueType glyph tables")
        self.units_per_em = float(
            getattr(self.font["head"], "unitsPerEm", 1000) or 1000
        )
        self.cid_to_gid = cid_to_gid
        self.unicode_cmap = _best_unicode_gid_cmap(self.font)
        self.glyph_to_unicode = _invert_unicode_cmap(self.unicode_cmap)
        self.cmap = self.unicode_cmap if use_cmap else {}
        self._glyph_bitmap_cache: dict[tuple[int, int, int], tuple[int, ...]] = {}

    def glyph_id_for_code(self, code: int) -> int:
        if self.cid_to_gid is not None:
            pos = code * 2
            if 0 <= pos and pos + 2 <= len(self.cid_to_gid):
                return struct.unpack(">H", self.cid_to_gid[pos : pos + 2])[0]
        if self.cmap:
            return self.cmap.get(code, code)
        return code

    def unicode_for_gid(self, gid: int) -> str:
        return self.glyph_to_unicode.get(gid, "")

    def glyph_bitmap(
        self, code: int, *, width: int = 24, height: int = 32
    ) -> tuple[int, ...]:
        gid = self.glyph_id_for_code(code)
        cache_key = (gid, width, height)
        cached = self._glyph_bitmap_cache.get(cache_key)
        if cached is not None:
            return cached
        contours = self.glyph_contours(gid)
        if not contours:
            self._glyph_bitmap_cache[cache_key] = ()
            return ()
        bitmap = rasterize_contours(contours, width=width, height=height)
        if len(self._glyph_bitmap_cache) >= 512:
            self._glyph_bitmap_cache.clear()
        self._glyph_bitmap_cache[cache_key] = bitmap
        return bitmap

    def glyph_bbox(self, code: int) -> tuple[float, float, float, float] | None:
        contours = self.glyph_contours(self.glyph_id_for_code(code))
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
        try:
            glyph_name = self.font.getGlyphName(gid)
            glyph_set = self.font.getGlyphSet()
            glyph = glyph_set[glyph_name]
            pen = DecomposingRecordingPen(glyph_set, skipMissingComponents=True)
            glyph.draw(pen)
            return _recording_to_contours(pen.value)
        except KeyError, TTLibError, AttributeError, IndexError, ValueError:
            return []

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
                bbox = _glyph_bbox(glyf, component_name)
                if bbox is None:
                    continue
                xmin, ymin, xmax, ymax = _transform_bbox(bbox, transform)
                w, h = xmax - xmin, ymax - ymin
                if h > 0 and h < 600 and 0.4 < w / h < 2.5 and ymin > 900:
                    has_dot = True
                else:
                    body_bbox = (xmin, ymin, xmax, ymax)
            return (body_bbox, has_dot)
        except KeyError, TTLibError, AttributeError, IndexError, ValueError:
            return (None, False)


def _tt_font_from_data(data: bytes) -> TTFont:
    try:
        return TTFont(BytesIO(data), lazy=True)
    except (TTLibError, OSError, struct.error, ValueError) as exc:
        raise ValueError("invalid TrueType font program") from exc


def _best_unicode_gid_cmap(font: TTFont) -> dict[int, int]:
    try:
        cmap_table = font["cmap"]
    except KeyError, TTLibError:
        return {}
    name_cmap = cmap_table.getBestCmap()
    if name_cmap is None:
        symbol_cmap = cmap_table.getcmap(3, 0)
        name_cmap = symbol_cmap.cmap if symbol_cmap is not None else {}
    mapping: dict[int, int] = {}
    for codepoint, glyph_name in name_cmap.items():
        if not (0 <= codepoint < 0x110000):
            continue
        try:
            gid = font.getGlyphID(glyph_name)
        except KeyError, TTLibError, AttributeError:
            continue
        if gid > 0:
            mapping[codepoint] = gid
    return mapping


def _invert_unicode_cmap(cmap: dict[int, int]) -> dict[int, str]:
    by_gid: dict[int, str] = {}
    for codepoint, gid in cmap.items():
        if gid <= 0:
            continue
        char = chr(codepoint)
        previous = by_gid.get(gid)
        if previous is None or _prefer_unicode_text(char, previous):
            by_gid[gid] = char
    return by_gid


def _prefer_unicode_text(candidate: str, current: str) -> bool:
    candidate_score = _unicode_text_score(candidate)
    current_score = _unicode_text_score(current)
    if candidate_score != current_score:
        return candidate_score > current_score
    return ord(candidate) < ord(current)


def _unicode_text_score(char: str) -> int:
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


def _glyph_bbox(glyf: Any, glyph_name: str) -> tuple[float, float, float, float] | None:
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


def _transform_bbox(
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


def _recording_to_contours(
    recording: list[tuple[str, tuple[Any, ...]]],
) -> list[list[Point]]:
    contours: list[list[Point]] = []
    contour: list[Point] = []
    current: Point | None = None
    start: Point | None = None
    for operator, operands in recording:
        if operator == "moveTo":
            if contour:
                contours.append(_close_contour(contour))
            start = _point(operands[0])
            current = start
            contour = [start]
        elif operator == "lineTo" and current is not None:
            current = _point(operands[0])
            contour.append(current)
        elif operator == "qCurveTo" and current is not None:
            current = _append_quadratic(contour, current, start, operands)
        elif operator == "curveTo" and current is not None:
            current = _append_cubic(contour, current, operands)
        elif operator in {"closePath", "endPath"}:
            if contour:
                contours.append(_close_contour(contour))
                contour = []
                current = None
                start = None
    if contour:
        contours.append(_close_contour(contour))
    return [contour for contour in contours if len(contour) >= 3]


def _point(value: Any) -> Point:
    x, y = value
    return (float(x), float(y))


def _append_quadratic(
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
        controls = [_point(point) for point in points[:-1]]
        end = start
    else:
        controls = [_point(point) for point in points[:-1]]
        end = _point(points[-1])
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
        contour.extend(_flatten_quadratic(segment_start, control, segment_end))
        segment_start = segment_end
    return end


def _append_cubic(
    contour: list[Point], current: Point, operands: tuple[Any, ...]
) -> Point:
    if len(operands) % 3:
        return current
    segment_start = current
    for index in range(0, len(operands), 3):
        c1 = _point(operands[index])
        c2 = _point(operands[index + 1])
        end = _point(operands[index + 2])
        contour.extend(_flatten_cubic(segment_start, c1, c2, end))
        segment_start = end
    return segment_start


def _close_contour(contour: list[Point]) -> list[Point]:
    if contour and contour[0] != contour[-1]:
        return [*contour, contour[0]]
    return contour


def _flatten_quadratic(
    p0: Point, p1: Point, p2: Point, segments: int = 6
) -> list[Point]:
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


def _flatten_cubic(
    p0: Point, p1: Point, p2: Point, p3: Point, segments: int = 8
) -> list[Point]:
    out: list[Point] = []
    for i in range(1, segments + 1):
        t = i / segments
        mt = 1.0 - t
        out.append(
            (
                mt**3 * p0[0]
                + 3.0 * mt * mt * t * p1[0]
                + 3.0 * mt * t * t * p2[0]
                + t**3 * p3[0],
                mt**3 * p0[1]
                + 3.0 * mt * mt * t * p1[1]
                + 3.0 * mt * t * t * p2[1]
                + t**3 * p3[1],
            )
        )
    return out


def rasterize_contours(
    contours: list[list[Point]], *, width: int, height: int
) -> tuple[int, ...]:
    points = [point for contour in contours for point in contour]
    if not points or width <= 0 or height <= 0:
        return ()
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    glyph_width = max(max_x - min_x, 1.0)
    glyph_height = max(max_y - min_y, 1.0)
    edges: list[tuple[float, float, float, float]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        normalized = [
            (
                (px - min_x) / glyph_width * (width - 1),
                (py - min_y) / glyph_height * (height - 1),
            )
            for px, py in contour
        ]
        previous = normalized[-1]
        for point in normalized:
            x0, y0 = previous
            x1, y1 = point
            if y0 != y1:
                edges.append((x0, y0, x1, y1))
            previous = point
    if not edges:
        return ()
    rows: list[int] = []
    for y in range(height - 1, -1, -1):
        intersections: list[float] = []
        row = 0
        y_mid = y + 0.5
        for x0, y0, x1, y1 in edges:
            if (y0 > y_mid) == (y1 > y_mid):
                continue
            intersections.append(x0 + (x1 - x0) * (y_mid - y0) / (y1 - y0))
        intersections.sort()
        for index in range(0, len(intersections) - 1, 2):
            start_x = max(0, ceil(intersections[index] - 0.5))
            end_x = min(width - 1, ceil(intersections[index + 1] - 0.5) - 1)
            for x in range(start_x, end_x + 1):
                row |= 1 << x
        rows.append(row)
    return tuple(rows)


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
