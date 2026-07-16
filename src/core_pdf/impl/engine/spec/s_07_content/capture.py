# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import typing
from math import ceil, hypot
from typing import Any

from core_pdf.impl.engine.layout.geometry import RectBox
from core_pdf.impl.engine.layout.glyphs import (
    GlyphCluster,
    GlyphObservation,
    Matrix6,
    glyph_cluster_from_observations,
    glyph_unicode_confidence,
    rectbox_tuple,
    union_bboxes,
)
from core_pdf.impl.engine.layout.models import TextRun
from core_pdf.impl.engine.spec.s_07_content.text_helpers import (
    NO_SPACE_AFTER,
    NO_SPACE_BEFORE,
    can_merge_cross_font_word,
    gap_separator,
    normalize_extracted_text,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_08_graphics.matrix import Matrix
from core_pdf.impl.objects import PdfName, PdfStream, PdfString
from core_pdf.impl.third_party._vendor.fontTools.encodings.StandardEncoding import StandardEncoding

if typing.TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_09_fonts.decoder import DecodedGlyph, FontDecoder


GLYPH_BITMAP_REPAIR_LABELS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,-+/()[]{}<>|_~"
)
SUSPICIOUS_GLYPH_BITMAP_TEXT = {"\ufffd", "\ufffc"}


def should_capture_glyph_bitmap(text: str) -> bool:
    if len(text) != 1:
        return False
    if len(text) == 1 and text in GLYPH_BITMAP_REPAIR_LABELS:
        return True
    if text in SUSPICIOUS_GLYPH_BITMAP_TEXT:
        return True
    code = ord(text)
    return 0xE000 <= code <= 0xF8FF or code < 32


def glyph_bitmap_dimensions(
    glyph_bbox: tuple[float, float, float, float] | None,
    font_size: float,
) -> tuple[int, int]:
    if glyph_bbox is None:
        return (24, 32)
    x0, y0, x1, y1 = glyph_bbox
    width = x1 - x0
    height = y1 - y0
    if width <= 0.0 or height <= 0.0:
        return (24, 32)
    bitmap_h = max(16, min(64, ceil(max(font_size, 1.0) * 2.5)))
    bitmap_w = max(1, min(96, ceil(bitmap_h * width / height)))
    return (bitmap_w, bitmap_h)


def glyph_ink_rect(
    state: Any,
    glyph_bbox: tuple[float, float, float, float] | None,
    advance_start: float,
    fallback: RectBox,
) -> RectBox:
    if glyph_bbox is None:
        return fallback
    gx0, gy0, gx1, gy1 = glyph_bbox
    if gx1 <= gx0 or gy1 <= gy0:
        return fallback
    text_x0 = advance_start + gx0 * state.text_advance_scale
    text_x1 = advance_start + gx1 * state.text_advance_scale
    text_y0 = state.rise + gy0 * state.font_scale
    text_y1 = state.rise + gy1 * state.font_scale
    base_x = state.tm_e * state.ca + state.tm_f * state.cc + state.ce
    base_y = state.tm_e * state.cb + state.tm_f * state.cd + state.cf
    a = state.combined_A
    b = state.combined_B
    c = state.combined_C
    d = state.combined_D
    corners = (
        (text_x0, text_y0),
        (text_x0, text_y1),
        (text_x1, text_y0),
        (text_x1, text_y1),
    )
    points = [(base_x + x * a + y * c, base_y + x * b + y * d) for x, y in corners]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    rect = RectBox(
        min(xs),
        min(ys),
        max(xs),
        max(ys),
        seqno=fallback.seqno,
        fill=fallback.fill,
        fill_opacity=fallback.fill_opacity,
    )
    fallback_height = fallback.y1 - fallback.y0
    fallback_width = fallback.x1 - fallback.x0
    rect_height = rect.y1 - rect.y0
    rect_width = rect.x1 - rect.x0
    if rect_width <= 0.01 or rect_height <= 0.01:
        return fallback
    if fallback_width > 0.0 and rect_width > fallback_width * 4.0:
        return fallback
    if fallback_height > 0.0 and rect_height > fallback_height * 1.5:
        return fallback
    return rect


def transformed_text_rect(
    state: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> RectBox:
    base_x = state.tm_e * state.ca + state.tm_f * state.cc + state.ce
    base_y = state.tm_e * state.cb + state.tm_f * state.cd + state.cf
    a = state.combined_A
    b = state.combined_B
    c = state.combined_C
    d = state.combined_D
    points = (
        (base_x + x0 * a + y0 * c, base_y + x0 * b + y0 * d),
        (base_x + x0 * a + y1 * c, base_y + x0 * b + y1 * d),
        (base_x + x1 * a + y0 * c, base_y + x1 * b + y0 * d),
        (base_x + x1 * a + y1 * c, base_y + x1 * b + y1 * d),
    )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return RectBox(
        min(xs),
        min(ys),
        max(xs),
        max(ys),
        seqno=state.sequence,
        fill=state.fill_color,
        fill_opacity=state.fill_opacity,
    )


def transformed_text_line(
    state: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[float, float, float, float]:
    base_x = state.tm_e * state.ca + state.tm_f * state.cc + state.ce
    base_y = state.tm_e * state.cb + state.tm_f * state.cd + state.cf
    a = state.combined_A
    b = state.combined_B
    c = state.combined_C
    d = state.combined_D
    return (
        base_x + x0 * a + y0 * c,
        base_y + x0 * b + y0 * d,
        base_x + x1 * a + y1 * c,
        base_y + x1 * b + y1 * d,
    )


def glyph_text_space_boxes(
    state: Any,
    offset: float,
    advance: float,
    decoder: Any,
    position: tuple[float, float] = (0.0, 0.0),
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    Matrix6,
]:
    if decoder.is_vertical:
        position_x, position_y = position
        start_y = state.rise + position_y - offset
        end_y = start_y - advance
        ar = state.font_ascent
        dr = state.font_descent
        x0 = position_x + (dr if dr < ar else ar)
        x1 = position_x + (ar if ar > dr else dr)
        y0 = end_y if end_y < start_y else start_y
        y1 = start_y if start_y > end_y else end_y
        return (
            (x0, y0, x1, y1),
            (0.0, start_y, 0.0, end_y),
            (
                state.tm_a,
                state.tm_b,
                state.tm_c,
                state.tm_d,
                state.tm_e
                + position_x * state.tm_a
                - offset * state.tm_c
                + position_y * state.tm_c,
                state.tm_f
                + position_x * state.tm_b
                - offset * state.tm_d
                + position_y * state.tm_d,
            ),
        )
    ar = state.font_ascent + state.rise
    dr = state.font_descent + state.rise
    return (
        (offset, dr, offset + advance, ar),
        (offset, state.rise, offset + advance, state.rise),
        (
            state.tm_a,
            state.tm_b,
            state.tm_c,
            state.tm_d,
            state.tm_e + offset * state.tm_a,
            state.tm_f + offset * state.tm_b,
        ),
    )


def type3_glyph_names(font: dict[Any, Any], decoder: Any) -> dict[int, str]:
    encoding = lookup_dict_key(font, "Encoding")
    differences_obj = (
        lookup_dict_key(encoding, "Differences") if isinstance(encoding, dict) else None
    )
    glyph_names = {
        code: name for code, name in enumerate(StandardEncoding) if name != ".notdef"
    }
    if decoder.base_encoding == "MacRomanEncoding":
        from core_pdf.impl.third_party._vendor.fontTools.encodings.MacRoman import MacRoman

        glyph_names = {code: name for code, name in enumerate(MacRoman) if name != ".notdef"}
    if isinstance(differences_obj, (list, tuple)):
        code = 0
        for item in differences_obj:
            if type(item) is int:
                code = item
                continue
            name = str(item) if isinstance(item, PdfName) else None
            if name is not None and 0 <= code <= 255:
                glyph_names[code] = name
                code += 1
    return glyph_names


def apply_glyph_geometry_to_run(
    run: TextRun,
    glyphs: typing.Iterable[GlyphObservation],
    glyph_clusters: tuple[GlyphCluster, ...] = (),
) -> None:
    glyph_tuple = tuple(glyphs)
    if glyph_clusters:
        run.glyph_clusters = glyph_clusters
    if not glyph_tuple:
        return
    ink_bbox = union_bboxes(tuple(rectbox_tuple(glyph.ink_rect) for glyph in glyph_tuple))
    advance_bbox = union_bboxes(tuple(rectbox_tuple(glyph.advance_rect) for glyph in glyph_tuple))
    if advance_bbox is not None:
        run.advance_bbox = advance_bbox
    if ink_bbox is not None:
        run.ink_bbox = ink_bbox
    confidences = [glyph.confidence for glyph in glyph_tuple if glyph.confidence is not None]
    if confidences:
        run.confidence = min(confidences)


def type3_font_matrix(font: dict[str, Any]) -> Matrix:
    matrix = lookup_dict_key(font, "FontMatrix")
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 6:
        return Matrix(0.001, 0.0, 0.0, 0.001, 0.0, 0.0)
    values: list[float] = []
    for value in matrix:
        if type(value) not in (int, float):
            return Matrix(0.001, 0.0, 0.0, 0.001, 0.0, 0.0)
        values.append(float(typing.cast(Any, value)))
    return Matrix(*values)


class CapturedLine:
    __slots__ = ("x0", "y0", "x1", "y1", "line_width")

    def __init__(self, x0: float, y0: float, x1: float, y1: float, line_width: float = 1.0) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.line_width = line_width

    def replace(self, **kwargs: Any) -> CapturedLine:
        return CapturedLine(
            x0=kwargs.get("x0", self.x0),
            y0=kwargs.get("y0", self.y0),
            x1=kwargs.get("x1", self.x1),
            y1=kwargs.get("y1", self.y1),
            line_width=kwargs.get("line_width", self.line_width),
        )


class CapturedSubpath:
    __slots__ = ("points", "closed")

    def __init__(
        self,
        points: list[tuple[float, float]] | None = None,
        *,
        closed: bool = False,
    ) -> None:
        self.points = points if points is not None else []
        self.closed = closed

    def clone(self) -> CapturedSubpath:
        return CapturedSubpath(list(self.points), closed=self.closed)

    def transformed(self, matrix: Matrix) -> CapturedSubpath:
        a, b, c, d, e, f = matrix
        return CapturedSubpath(
            [(x * a + y * c + e, x * b + y * d + f) for x, y in self.points],
            closed=self.closed,
        )

    def translated(self, tx: float, ty: float) -> CapturedSubpath:
        return CapturedSubpath(
            [(x + tx, y + ty) for x, y in self.points],
            closed=self.closed,
        )

    def line_to(self, x: float, y: float) -> None:
        if self.closed:
            return
        self.points.append((x, y))

    def close(self) -> None:
        if len(self.points) > 1:
            self.closed = True

    def has_segments(self) -> bool:
        return len(self.points) > 1

    def bbox(self) -> tuple[float, float, float, float] | None:
        if not self.points:
            return None
        xs = [point[0] for point in self.points]
        ys = [point[1] for point in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def edges(self, *, close_open: bool = False) -> list[tuple[float, float, float, float]]:
        if len(self.points) < 2:
            return []
        edges = [(x0, y0, x1, y1) for (x0, y0), (x1, y1) in zip(self.points, self.points[1:])]
        first = self.points[0]
        last = self.points[-1]
        if (self.closed or close_open) and first != last:
            edges.append((last[0], last[1], first[0], first[1]))
        return edges


class CapturedPath:
    __slots__ = ("subpaths",)

    def __init__(self, subpaths: list[CapturedSubpath] | None = None) -> None:
        self.subpaths = subpaths if subpaths is not None else []

    def clone(self) -> CapturedPath:
        return CapturedPath([subpath.clone() for subpath in self.subpaths])

    def transformed(self, matrix: Matrix) -> CapturedPath:
        return CapturedPath([subpath.transformed(matrix) for subpath in self.subpaths])

    def translated(self, tx: float, ty: float) -> CapturedPath:
        return CapturedPath([subpath.translated(tx, ty) for subpath in self.subpaths])

    def clear(self) -> None:
        self.subpaths.clear()

    def move_to(self, x: float, y: float) -> None:
        self.subpaths.append(CapturedSubpath([(x, y)]))

    def line_to(self, x: float, y: float) -> None:
        if not self.subpaths:
            self.move_to(x, y)
            return
        self.subpaths[-1].line_to(x, y)

    def close(self) -> None:
        if self.subpaths:
            self.subpaths[-1].close()

    def rect(self, x: float, y: float, w: float, h: float) -> None:
        self.subpaths.append(
            CapturedSubpath(
                [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                closed=True,
            )
        )

    def has_segments(self) -> bool:
        return any(subpath.has_segments() for subpath in self.subpaths)

    def bbox(self) -> tuple[float, float, float, float] | None:
        boxes = [box for subpath in self.subpaths if (box := subpath.bbox())]
        if not boxes:
            return None
        return (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )

    def fill_edges(self) -> list[tuple[float, float, float, float]]:
        edges: list[tuple[float, float, float, float]] = []
        for subpath in self.subpaths:
            edges.extend(subpath.edges(close_open=True))
        return edges

    def stroke_subpaths(self) -> list[list[tuple[float, float, float, float]]]:
        return [edges for subpath in self.subpaths if (edges := subpath.edges(close_open=False))]

    def derived_lines(self, line_width: float) -> list[CapturedLine]:
        return [
            CapturedLine(x0, y0, x1, y1, line_width)
            for subpath in self.subpaths
            for x0, y0, x1, y1 in subpath.edges(close_open=False)
            if abs(x1 - x0) > 0.01 or abs(y1 - y0) > 0.01
        ]


DrawingItem = tuple[str, tuple[tuple[float, float], ...]]


class CapturedDrawing:
    __slots__ = (
        "seqno",
        "fill",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "stroke_opacity",
        "line_width",
        "line_cap",
        "line_join",
        "dash_pattern",
        "fill_rule",
        "blend_mode",
        "soft_mask_alpha",
        "raw_data",
        "dictionary",
        "items",
        "path",
        "bbox",
        "kind",
    )

    def __init__(
        self,
        seqno: int,
        fill: tuple[float, ...] | None,
        fill_opacity: float | None,
        fill_pattern: dict[str, Any] | None = None,
        stroke_color: tuple[float, ...] | None = None,
        stroke_pattern: dict[str, Any] | None = None,
        stroke_opacity: float | None = None,
        line_width: float = 1.0,
        line_cap: int = 0,
        line_join: int = 0,
        dash_pattern: tuple[list[float], float] | None = None,
        fill_rule: str = "nonzero",
        blend_mode: str | None = None,
        soft_mask_alpha: float | None = None,
        raw_data: bytes | None = None,
        dictionary: dict[Any, Any] | None = None,
        kind: str = "fill",
        items: list[DrawingItem] | None = None,
        path: CapturedPath | None = None,
        bbox: RectBox | None = None,
    ) -> None:
        self.seqno = seqno
        self.fill = fill
        self.fill_pattern = fill_pattern
        self.fill_opacity = fill_opacity
        self.stroke_color = stroke_color
        self.stroke_pattern = stroke_pattern
        self.stroke_opacity = stroke_opacity
        self.line_width = line_width
        self.line_cap = line_cap
        self.line_join = line_join
        self.dash_pattern = dash_pattern
        self.fill_rule = fill_rule
        self.blend_mode = blend_mode
        self.soft_mask_alpha = soft_mask_alpha
        self.raw_data = raw_data
        self.dictionary = dictionary
        self.kind = kind
        self.items = items if items is not None else []
        self.path = path
        self.bbox = bbox

    def replace(self, **kwargs: Any) -> CapturedDrawing:
        return CapturedDrawing(
            seqno=kwargs.get("seqno", self.seqno),
            fill=kwargs.get("fill", self.fill),
            fill_pattern=kwargs.get("fill_pattern", self.fill_pattern),
            fill_opacity=kwargs.get("fill_opacity", self.fill_opacity),
            stroke_color=kwargs.get("stroke_color", self.stroke_color),
            stroke_pattern=kwargs.get("stroke_pattern", self.stroke_pattern),
            stroke_opacity=kwargs.get("stroke_opacity", self.stroke_opacity),
            line_width=kwargs.get("line_width", self.line_width),
            line_cap=kwargs.get("line_cap", self.line_cap),
            line_join=kwargs.get("line_join", self.line_join),
            dash_pattern=kwargs.get("dash_pattern", self.dash_pattern),
            fill_rule=kwargs.get("fill_rule", self.fill_rule),
            blend_mode=kwargs.get("blend_mode", self.blend_mode),
            soft_mask_alpha=kwargs.get("soft_mask_alpha", self.soft_mask_alpha),
            raw_data=kwargs.get("raw_data", self.raw_data),
            dictionary=kwargs.get("dictionary", self.dictionary),
            kind=kwargs.get("kind", self.kind),
            items=kwargs.get("items", self.items),
            path=kwargs.get("path", self.path),
            bbox=kwargs.get("bbox", self.bbox),
        )

    @property
    def rect(self) -> RectBox | None:
        if self.bbox is not None:
            return self.bbox.normalize()
        if self.path is None:
            return None
        bbox = self.path.bbox()
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        return RectBox(
            x0,
            y0,
            x1,
            y1,
            seqno=self.seqno,
            fill=self.fill,
            fill_opacity=self.fill_opacity,
        )


class ContentCaptureMixin:
    """Content capture and text-run emission methods for the concrete TextState class."""

    __slots__ = ()

    pending_run: TextRun | None

    def flush_run(self: Any) -> None:
        if not self.capture_runs:
            self.pending_run = None
            return
        if self.pending_run:
            self.runs.append(self.pending_run)
            self.pending_run = None

    def transform_point(self: Any, x: float, y: float) -> tuple[float, float]:
        return (
            x * self.ca + y * self.cc + self.ce,
            x * self.cb + y * self.cd + self.cf,
        )

    def graphics_scale(self: Any) -> float:
        x_scale = hypot(self.ca, self.cb)
        y_scale = hypot(self.cc, self.cd)
        if x_scale == 0 and y_scale == 0:
            return 1.0
        if x_scale == 0:
            return y_scale
        if y_scale == 0:
            return x_scale
        return (x_scale + y_scale) * 0.5

    def transformed_line_width(self: Any) -> float:
        line_width = max(0.0, self.line_width)
        if line_width == 0:
            return 0.0
        return line_width * self.graphics_scale()

    def transformed_dash_pattern(self: Any) -> tuple[list[float], float] | None:
        dash_pattern = self.dash_pattern
        if not dash_pattern:
            return None
        dash_array, phase = dash_pattern
        scale = self.graphics_scale()
        return [max(0.0, float(value) * scale) for value in dash_array], float(phase) * scale

    def flush_drawing(self: Any, kind: str, fill_rule: str = "nonzero") -> None:
        if not self.capture_graphics or not self.is_graphics_visible():
            self.current_path.clear()
            return

        path = self.current_path.transformed(
            Matrix(self.ca, self.cb, self.cc, self.cd, self.ce, self.cf)
        )
        self.current_path.clear()
        if path.has_segments():
            line_width = self.transformed_line_width()
            self.lines.extend(path.derived_lines(line_width))
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=self.fill_color,
                    fill_pattern=self.fill_pattern,
                    fill_opacity=self.fill_opacity,
                    stroke_color=self.stroke_color,
                    stroke_pattern=self.stroke_pattern,
                    stroke_opacity=self.stroke_opacity,
                    line_width=line_width,
                    line_cap=self.line_cap,
                    line_join=self.line_join,
                    dash_pattern=self.transformed_dash_pattern(),
                    fill_rule=fill_rule,
                    blend_mode=self.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    kind=kind,
                    path=path,
                )
            )

    def alloc_run(
        self: Any,
        text: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        tx: float,
        ty: float,
        font_size: float,
        space_width: float,
        order: int,
        stream_order: int,
        xobject_depth: int,
        font_name: str | None,
        is_vertical: bool,
        rotation_angle: int,
        visible: bool,
        line_break_before: bool,
        seqno: int,
        fill_color: tuple[float, ...] | None,
        advance_bbox: tuple[float, float, float, float] | None = None,
        ink_bbox: tuple[float, float, float, float] | None = None,
        baseline: tuple[float, float, float, float] | None = None,
        provenance: tuple[tuple[str, object], ...] = (),
        confidence: float | None = None,
        glyph_clusters: tuple[GlyphCluster, ...] = (),
    ) -> TextRun:
        text = normalize_extracted_text(text)
        return self.alloc_prepared_run(
            text,
            x0,
            y0,
            x1,
            y1,
            tx,
            ty,
            font_size,
            space_width,
            order,
            stream_order,
            xobject_depth,
            font_name,
            is_vertical,
            rotation_angle,
            visible,
            line_break_before,
            seqno,
            fill_color,
            advance_bbox,
            ink_bbox,
            baseline,
            provenance,
            confidence,
            glyph_clusters,
        )

    def alloc_prepared_run(
        self: Any,
        text: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        tx: float,
        ty: float,
        font_size: float,
        space_width: float,
        order: int,
        stream_order: int,
        xobject_depth: int,
        font_name: str | None,
        is_vertical: bool,
        rotation_angle: int,
        visible: bool,
        line_break_before: bool,
        seqno: int,
        fill_color: tuple[float, ...] | None,
        advance_bbox: tuple[float, float, float, float] | None = None,
        ink_bbox: tuple[float, float, float, float] | None = None,
        baseline: tuple[float, float, float, float] | None = None,
        provenance: tuple[tuple[str, object], ...] = (),
        confidence: float | None = None,
        glyph_clusters: tuple[GlyphCluster, ...] = (),
    ) -> TextRun:
        r = TextRun.reinit(
            None,
            text,
            x0,
            y0,
            x1,
            y1,
            tx,
            ty,
            font_size,
            space_width,
            order,
            stream_order,
            xobject_depth,
            font_name,
            is_vertical,
            rotation_angle,
            visible,
            line_break_before,
            seqno,
            fill_color,
            advance_bbox,
            ink_bbox,
            baseline,
            provenance,
            confidence,
            glyph_clusters,
        )
        self.run_pool.append(r)
        self.run_pool_idx += 1
        return r

    def update_pending_run(self: Any, new_run: TextRun) -> None:
        if not self.pending_run:
            self.pending_run = new_run
            return

        p = self.pending_run
        pc = p.coords
        nc = new_run.coords
        p_text = p.text
        new_text = new_run.text
        p_font_size = pc[TextRun.FONT_SIZE]
        p_space_width = pc[TextRun.SPACE_WIDTH]
        p_rotation = p.rotation_angle
        merge_threshold = max(p_space_width * 0.45, 2.0)

        is_same_style = (
            p_rotation == new_run.rotation_angle
            and p.visible == new_run.visible
            and not new_run.line_break_before
            and p_font_size == nc[TextRun.FONT_SIZE]
            and (
                p.font_name == new_run.font_name
                or can_merge_cross_font_word(p_text, new_text)
                or can_merge_cross_font_word(new_text, p_text)
            )
            and p.fill_color == new_run.fill_color
        )

        if is_same_style and p_rotation == 90:
            y_gap = nc[TextRun.Y0] - pc[TextRun.Y1]
            max_y_gap = max(p_space_width * 0.5, p_font_size * 0.8, 2.0)
            if abs(y_gap) > max_y_gap:
                is_same_style = False
        elif is_same_style and p_rotation == 0:
            if abs(pc[TextRun.Y0] - nc[TextRun.Y0]) > p_font_size * 0.5:
                is_same_style = False

        merged = False
        if is_same_style:
            if p_rotation in (0, 90):
                if p_rotation == 0:
                    gap = nc[TextRun.X0] - pc[TextRun.X1]
                    if -2.0 <= gap < merge_threshold:
                        separator = gap_separator(p_text, new_text, gap, p)
                        p.set_text(p_text + separator + new_text)
                        p.union_ink_bbox(new_run.ink_bbox)
                        if nc[TextRun.X1] > pc[TextRun.X1]:
                            p.x1 = nc[TextRun.X1]
                        merged = True
                    else:
                        gap_rtl = pc[TextRun.X0] - nc[TextRun.X1]
                        if -2.0 <= gap_rtl < merge_threshold:
                            separator = gap_separator(new_text, p_text, gap_rtl, p)
                            p.set_text(new_text + separator + p_text)
                            p.union_ink_bbox(new_run.ink_bbox)
                            if nc[TextRun.X0] < pc[TextRun.X0]:
                                p.x0 = nc[TextRun.X0]
                            merged = True
                else:
                    gap = nc[TextRun.Y0] - pc[TextRun.Y1]
                    if -2.0 <= gap < merge_threshold:
                        separator = gap_separator(p_text, new_text, gap, p)
                        p.set_text(p_text + separator + new_text)
                        p.union_ink_bbox(new_run.ink_bbox)
                        if nc[TextRun.Y1] > pc[TextRun.Y1]:
                            p.y1 = nc[TextRun.Y1]
                        merged = True
            else:
                h_gap_inv = pc[TextRun.X0] - nc[TextRun.X1]
                if -2.0 <= h_gap_inv < merge_threshold:
                    separator = gap_separator(p_text, new_text, h_gap_inv, p)
                    p.set_text(p_text + separator + new_text)
                    p.union_ink_bbox(new_run.ink_bbox)
                    if nc[TextRun.X0] < pc[TextRun.X0]:
                        p.x0 = nc[TextRun.X0]
                    merged = True
                else:
                    h_gap_inv_rtl = nc[TextRun.X0] - pc[TextRun.X1]
                    if -2.0 <= h_gap_inv_rtl < merge_threshold:
                        separator = gap_separator(new_text, p_text, h_gap_inv_rtl, p)
                        p.set_text(new_text + separator + p_text)
                        p.union_ink_bbox(new_run.ink_bbox)
                        if nc[TextRun.X1] > pc[TextRun.X1]:
                            p.x1 = nc[TextRun.X1]
                        merged = True

        if not merged:
            self.runs.append(p)
            self.pending_run = new_run
        else:
            p.extend_glyph_clusters(new_run.glyph_clusters)

    def merge_pending_horizontal_run(
        self,
        text: str,
        x0: float,
        x1: float,
        y0: float,
        font_size: float,
        font_name: str | None,
        visible: bool,
        fill_color: tuple[float, ...] | None,
    ) -> bool:
        p = self.pending_run
        if p is None or self.pending_line_break:
            return False

        pc = p.coords
        p_text = p.text
        if (
            p.rotation_angle != 0
            or p.visible != visible
            or pc[TextRun.FONT_SIZE] != font_size
            or p.fill_color != fill_color
            or (
                p.font_name != font_name
                and not can_merge_cross_font_word(p_text, text)
                and not can_merge_cross_font_word(text, p_text)
            )
            or abs(pc[TextRun.Y0] - y0) > font_size * 0.5
        ):
            return False

        gap = x0 - pc[TextRun.X1]
        merge_threshold = max(pc[TextRun.SPACE_WIDTH] * 0.45, 2.0)
        if not (-2.0 <= gap < merge_threshold):
            return False

        p.set_text(p_text + gap_separator(p_text, text, gap, p) + text)
        if x1 > pc[TextRun.X1]:
            p.x1 = x1
            p.ink_bbox = p.advance_bbox
        return True

    def record_glyph_observations(
        self: Any,
        text: str,
        data: bytes | bytearray | memoryview,
        decoder: FontDecoder,
        rotation_angle: int,
        visible: bool,
        glyphs: tuple[DecodedGlyph, ...] | None = None,
    ) -> None:
        data = bytes(data)
        if not self.capture_glyphs:
            return
        if glyphs is None:
            glyphs = decoder.decode_glyphs(data)
        if not glyphs:
            return

        advances = [
            self.chunk_advance(glyph.width_code, decoder, char_code=glyph.char_code)
            for glyph in glyphs
        ]

        offset = 0.0
        seqno = self.sequence
        fill = self.fill_color
        append_glyph = self.glyphs.append
        clusters = getattr(self, "glyph_clusters", None)
        if clusters is None:
            clusters = []
            self.glyph_clusters = clusters
        cursor = 0
        for decoded_index, (glyph, advance) in enumerate(zip(glyphs, advances, strict=True)):
            chunk_text = glyph.unicode
            if not chunk_text:
                chunk_text = text[cursor : cursor + 1]
            cursor += max(1, len(chunk_text))
            if not chunk_text:
                offset += advance
                continue

            cluster_id = len(clusters)
            text_box, baseline_text, glyph_text_matrix = glyph_text_space_boxes(
                self,
                offset,
                advance,
                decoder,
                decoder.vertical_glyph_position(glyph.cid, font_size=self.font_size)
                if decoder.is_vertical
                else (0.0, 0.0),
            )
            advance_rect = transformed_text_rect(self, *text_box)
            baseline = transformed_text_line(self, *baseline_text)
            glyph_bbox = decoder.glyph_bbox(glyph.bitmap_code) if not decoder.is_vertical else None
            rect = glyph_ink_rect(self, glyph_bbox, offset, advance_rect)
            device_matrix = (
                self.combined_A,
                self.combined_B,
                self.combined_C,
                self.combined_D,
                baseline[0],
                baseline[1],
            )
            common_provenance = (
                ("source", "native_glyph"),
                ("seqno", seqno),
                ("font_name", self.current_font),
                ("stream_order", self.stream_order),
                ("xobject_depth", self.xobject_depth),
                ("decoded_glyph_index", decoded_index),
            )
            writing_mode = "vertical" if decoder.is_vertical else "horizontal"
            cluster_observations: list[GlyphObservation] = []
            observation_confidence = glyph_unicode_confidence(
                chunk_text,
                glyph.unicode_source,
                visible=visible,
                alternates=glyph.alternates,
            )

            if len(chunk_text) == 1:
                bitmap: tuple[int, ...] = ()
                bitmap_width = 0
                bitmap_height = 0
                if should_capture_glyph_bitmap(chunk_text):
                    bitmap_width, bitmap_height = glyph_bitmap_dimensions(
                        glyph_bbox,
                        self.font_size,
                    )
                    bitmap = decoder.glyph_bitmap(
                        glyph.bitmap_code,
                        width=bitmap_width,
                        height=bitmap_height,
                    )
                    if not bitmap:
                        bitmap_width = 0
                        bitmap_height = 0
                cluster_observations.append(
                    GlyphObservation(
                        text=chunk_text,
                        ink_rect=rect,
                        advance_rect=advance_rect,
                        seqno=seqno,
                        code_bytes=glyph.code_bytes,
                        char_code=glyph.char_code,
                        cid=glyph.cid,
                        gid=glyph.gid,
                        font_name=self.current_font,
                        font_size=self.font_size,
                        space_width=self.font_space_width,
                        text_matrix=glyph_text_matrix,
                        device_matrix=device_matrix,
                        baseline=baseline,
                        writing_mode=writing_mode,
                        rotation_angle=rotation_angle,
                        stream_order=self.stream_order,
                        xobject_depth=self.xobject_depth,
                        fill=fill,
                        visible=visible,
                        confidence=observation_confidence,
                        unicode_source=glyph.unicode_source,
                        alternates=glyph.alternates,
                        cluster_id=cluster_id,
                        cluster_index=0,
                        cluster_size=1,
                        bitmap=bitmap,
                        bitmap_width=bitmap_width,
                        bitmap_height=bitmap_height,
                        provenance=common_provenance,
                    )
                )
            elif glyph.split_unicode:
                per_char_advance = advance / len(chunk_text)
                char_offset = offset
                cluster_size = len(chunk_text)
                for cluster_index, ch in enumerate(chunk_text):
                    char_confidence = glyph_unicode_confidence(
                        ch,
                        glyph.unicode_source,
                        visible=visible,
                        alternates=glyph.alternates,
                    )
                    char_box, char_baseline_text, char_text_matrix = glyph_text_space_boxes(
                        self, char_offset, per_char_advance, decoder
                    )
                    char_advance_rect = transformed_text_rect(self, *char_box)
                    char_baseline = transformed_text_line(self, *char_baseline_text)
                    char_device_matrix = (
                        self.combined_A,
                        self.combined_B,
                        self.combined_C,
                        self.combined_D,
                        char_baseline[0],
                        char_baseline[1],
                    )
                    cluster_observations.append(
                        GlyphObservation(
                            text=ch,
                            ink_rect=char_advance_rect,
                            advance_rect=char_advance_rect,
                            seqno=seqno,
                            code_bytes=glyph.code_bytes,
                            char_code=glyph.char_code,
                            cid=glyph.cid,
                            gid=glyph.gid,
                            font_name=self.current_font,
                            font_size=self.font_size,
                            space_width=self.font_space_width,
                            text_matrix=char_text_matrix,
                            device_matrix=char_device_matrix,
                            baseline=char_baseline,
                            writing_mode=writing_mode,
                            rotation_angle=rotation_angle,
                            stream_order=self.stream_order,
                            xobject_depth=self.xobject_depth,
                            fill=fill,
                            visible=visible,
                            confidence=char_confidence,
                            unicode_source=glyph.unicode_source,
                            alternates=glyph.alternates,
                            cluster_id=cluster_id,
                            cluster_index=cluster_index,
                            cluster_size=cluster_size,
                            provenance=common_provenance,
                        )
                    )
                    char_offset += per_char_advance
            else:
                cluster_observations.append(
                    GlyphObservation(
                        text=chunk_text,
                        ink_rect=rect,
                        advance_rect=advance_rect,
                        seqno=seqno,
                        code_bytes=glyph.code_bytes,
                        char_code=glyph.char_code,
                        cid=glyph.cid,
                        gid=glyph.gid,
                        font_name=self.current_font,
                        font_size=self.font_size,
                        space_width=self.font_space_width,
                        text_matrix=glyph_text_matrix,
                        device_matrix=device_matrix,
                        baseline=baseline,
                        writing_mode=writing_mode,
                        rotation_angle=rotation_angle,
                        stream_order=self.stream_order,
                        xobject_depth=self.xobject_depth,
                        fill=fill,
                        visible=visible,
                        confidence=observation_confidence,
                        unicode_source=glyph.unicode_source,
                        alternates=glyph.alternates,
                        cluster_id=cluster_id,
                        cluster_index=0,
                        cluster_size=1,
                        provenance=common_provenance,
                    )
                )
            for observation in cluster_observations:
                append_glyph(observation)
            kind = (
                "single_glyph"
                if len(cluster_observations) == 1 and len(chunk_text) == 1
                else "ligature"
                if glyph.split_unicode
                else "multi_codepoint"
            )
            cluster = glyph_cluster_from_observations(
                cluster_id,
                chunk_text,
                tuple(cluster_observations),
                kind=kind,
                provenance=common_provenance,
            )
            if cluster is not None:
                clusters.append(cluster)
            offset += advance

    def append_text(
        self: Any,
        operand: Any = None,
        *,
        data: bytes | memoryview | None = None,
        decoder: FontDecoder | None = None,
    ) -> None:
        decoder = decoder if decoder is not None else self.get_decoder()

        if data is not None:
            if type(data) is memoryview:
                data = data.tobytes()
            glyphs = None
            if self.capture_glyphs:
                glyphs = decoder.decode_glyphs(data)
                text = "".join(glyph.unicode for glyph in glyphs)
            else:
                text = decoder.decode(data)
        else:
            text, data, glyphs = self.decode_operand(operand, decoder)
        data = bytes(data)
        data_len = len(data)
        rendered_type3_glyphs = False
        if decoder.is_type3 and self.capture_graphics and data:
            text_matrix = self.text_matrix
            line_matrix = self.line_matrix
            self.render_type3_glyphs(data, decoder)
            rendered_type3_glyphs = True
            self.text_matrix = text_matrix
            self.line_matrix = line_matrix
        if not text:
            if data and rendered_type3_glyphs:
                adv_x, adv_y = decoder.text_advance_vector(
                    data,
                    font_size=self.font_size,
                    char_space=self.char_space,
                    word_space=self.word_space,
                    horizontal_scale=self.horizontal_scale,
                    glyphs=glyphs,
                )
                te, tf = self.tm_e, self.tm_f
                ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d
                self.tm_e = te + adv_x * ta + adv_y * tc
                self.tm_f = tf + adv_x * tb + adv_y * td
                self.pending_line_break = False
            return

        simple_horizontal_run = (
            self.capture_runs
            and not self.capture_glyphs
            and self.cached_rotation == 0
            and not decoder.is_vertical
        )
        if (
            simple_horizontal_run
            and not self.marked_content_stack
            and self.render_mode != 3
            and self.font_size >= 0.1
        ):
            visible = True
        else:
            visible = self.is_text_visible(text)

        fs = self.font_size
        rise = self.rise

        if (
            glyphs is None
            and not decoder.is_cid_font
            and decoder.to_unicode is None
            and decoder.cmap is None
        ):
            widths = self.font_widths or decoder.fast_widths
            cs = self.char_space_scale
            ws = self.word_space_scale
            scale = self.text_advance_scale
            if data_len == 1:
                byte = data[0]
                total = widths[byte] + cs
                if byte == 32:
                    total += ws
            else:
                total = 0.0
                space_count = 0
                for b in data:
                    total += widths[b]
                    if b == 32:
                        space_count += 1
                total += data_len * cs + space_count * ws
            if decoder.is_vertical:
                adv_x, adv_y = 0.0, -total * scale
            else:
                adv_x, adv_y = total * scale, 0.0
        else:
            adv_x, adv_y = decoder.text_advance_vector(
                data,
                font_size=fs,
                char_space=self.char_space,
                word_space=self.word_space,
                horizontal_scale=self.horizontal_scale,
                glyphs=glyphs,
            )

        ascent = self.font_ascent
        descent = self.font_descent

        A = self.combined_A
        B = self.combined_B
        C = self.combined_C
        D = self.combined_D

        ca = self.ca
        cb = self.cb
        cc = self.cc
        cd = self.cd
        ce = self.ce
        cf = self.cf
        te, tf = self.tm_e, self.tm_f
        E = te * ca + tf * cc + ce
        F = te * cb + tf * cd + cf

        if decoder.is_vertical:
            c0_x = descent * A + rise * C + E
            c0_y = descent * B + rise * D + F
            c1_x = ascent * A + rise * C + E
            c1_y = ascent * B + rise * D + F
            adv_C = adv_y * C
            adv_D = adv_y * D
            c2_x = adv_C + c0_x
            c2_y = adv_D + c0_y
            c3_x = adv_C + c1_x
            c3_y = adv_D + c1_y
        else:
            ar = ascent + rise
            dr = descent + rise
            c0_x = dr * C + E
            c0_y = dr * D + F
            c1_x = ar * C + E
            c1_y = ar * D + F
            adv_A = adv_x * A
            adv_B = adv_x * B
            c2_x = adv_A + c0_x
            c2_y = adv_B + c0_y
            c3_x = adv_A + c1_x
            c3_y = adv_B + c1_y

        x0 = c0_x if c0_x < c1_x else c1_x
        if c2_x < x0:
            x0 = c2_x
        if c3_x < x0:
            x0 = c3_x

        y0 = c0_y if c0_y < c1_y else c1_y
        if c2_y < y0:
            y0 = c2_y
        if c3_y < y0:
            y0 = c3_y

        x1 = c0_x if c0_x > c1_x else c1_x
        if c2_x > x1:
            x1 = c2_x
        if c3_x > x1:
            x1 = c3_x

        y1 = c0_y if c0_y > c1_y else c1_y
        if c2_y > y1:
            y1 = c2_y
        if c3_y > y1:
            y1 = c3_y

        rot = self.cached_rotation
        seqno = self.sequence
        ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d
        scale_factor = hypot(C, D) if decoder.is_vertical else hypot(A, B)
        effective_font_size = fs * scale_factor
        effective_space_width = self.font_space_width * scale_factor
        baseline = (
            E,
            F,
            E + adv_x * A + adv_y * C,
            F + adv_x * B + adv_y * D,
        )
        provenance = (
            ("source", "native_text"),
            ("seqno", seqno),
            ("font_name", self.current_font),
            ("stream_order", self.stream_order),
            ("xobject_depth", self.xobject_depth),
            ("text_render_mode", self.render_mode),
            ("font_size", fs),
            ("clip_bbox", self.clip_bbox),
        )
        advance_bbox = (x0, y0, x1, y1)

        actual_text_span = self.current_actual_text_span()
        if actual_text_span is not None:
            actual_text_span.add_extents(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                nbytes=data_len,
                tx=te,
                ty=tf,
                font_size=effective_font_size,
                space_width=effective_space_width,
                order=seqno,
                stream_order=self.stream_order,
                xobject_depth=self.xobject_depth,
                font_name=self.current_font,
                is_vertical=decoder.is_vertical,
                rotation_angle=rot,
                visible=visible,
                line_break_before=self.pending_line_break,
                seqno=seqno,
                fill_color=self.fill_color,
                advance_bbox=advance_bbox,
                ink_bbox=advance_bbox,
                baseline=baseline,
                provenance=provenance,
                confidence=1.0,
            )
            self.sequence = seqno + 1
            self.tm_e = te + adv_x * ta + adv_y * tc
            self.tm_f = tf + adv_x * tb + adv_y * td
            self.pending_line_break = False
            return

        if not self.capture_runs:
            if self.capture_glyphs:
                self.record_glyph_observations(
                    text,
                    data,
                    decoder,
                    rot,
                    visible,
                    glyphs=glyphs,
                )
            self.sequence = seqno + 1
            self.pending_line_break = False
            self.tm_e = te + adv_x * ta + adv_y * tc
            self.tm_f = tf + adv_x * tb + adv_y * td
            return

        prepared_text = text
        prepared_visible = visible
        if simple_horizontal_run:
            normalized_text = normalize_extracted_text(text)
            prepared_text = normalized_text
            if normalized_text and self.merge_pending_horizontal_run(
                normalized_text,
                x0,
                x1,
                y0,
                effective_font_size,
                self.current_font,
                prepared_visible,
                self.fill_color,
            ):
                self.sequence = seqno + 1
                self.tm_e = te + adv_x * ta + adv_y * tc
                self.tm_f = tf + adv_x * tb + adv_y * td
                self.pending_line_break = False
                return

        if simple_horizontal_run:
            new_run = self.alloc_prepared_run(
                text=prepared_text,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                tx=te,
                ty=tf,
                font_size=effective_font_size,
                font_name=self.current_font,
                space_width=effective_space_width,
                order=seqno,
                stream_order=self.stream_order,
                xobject_depth=self.xobject_depth,
                is_vertical=False,
                rotation_angle=0,
                visible=prepared_visible,
                line_break_before=self.pending_line_break,
                seqno=seqno,
                fill_color=self.fill_color,
                advance_bbox=advance_bbox,
                ink_bbox=advance_bbox,
                baseline=baseline,
                provenance=provenance,
                confidence=None,
            )
        else:
            new_run = self.alloc_run(
                text=text,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                tx=te,
                ty=tf,
                font_size=effective_font_size,
                font_name=self.current_font,
                space_width=effective_space_width,
                order=seqno,
                stream_order=self.stream_order,
                xobject_depth=self.xobject_depth,
                is_vertical=decoder.is_vertical,
                rotation_angle=rot,
                visible=visible,
                line_break_before=self.pending_line_break,
                seqno=seqno,
                fill_color=self.fill_color,
                advance_bbox=advance_bbox,
                ink_bbox=advance_bbox,
                baseline=baseline,
                provenance=provenance,
                confidence=None,
            )
        if self.capture_glyphs:
            glyph_start = len(self.glyphs)
            cluster_start = len(self.glyph_clusters)
            self.record_glyph_observations(
                text,
                data,
                decoder,
                rot,
                visible,
                glyphs=glyphs,
            )
            apply_glyph_geometry_to_run(
                new_run,
                self.glyphs[glyph_start:],
                tuple(self.glyph_clusters[cluster_start:]),
            )

        self.update_pending_run(new_run)

        self.sequence = seqno + 1

        self.tm_e = te + adv_x * ta + adv_y * tc
        self.tm_f = tf + adv_x * tb + adv_y * td
        self.pending_line_break = False

    def render_type3_glyphs(self: Any, data: bytes, decoder: FontDecoder) -> None:
        font = decoder.font
        char_procs = lookup_dict_key(font, "CharProcs")
        if not isinstance(char_procs, dict):
            return
        glyph_names = decoder.type3_glyph_names
        if glyph_names is None:
            glyph_names = type3_glyph_names(font, decoder)
            decoder.type3_glyph_names = glyph_names

        resources = lookup_dict_key(font, "Resources")
        if not isinstance(resources, dict):
            resources = self.resources
        font_matrix = type3_font_matrix(font)
        widths = self.font_widths or decoder.fast_widths
        cs = self.char_space_scale
        ws = self.word_space_scale
        scale = self.text_advance_scale

        for code in data:
            glyph_name = glyph_names.get(code)
            char_proc = lookup_dict_key(char_procs, glyph_name) if glyph_name else None
            char_proc = self.resolve(char_proc)
            if isinstance(char_proc, PdfStream):
                glyph_ctm = Matrix(
                    self.combined_A,
                    self.combined_B,
                    self.combined_C,
                    self.combined_D,
                    self.tm_e * self.ca + self.tm_f * self.cc + self.ce,
                    self.tm_e * self.cb + self.tm_f * self.cd + self.cf,
                ).multiply(font_matrix)
                self.consume_stream(char_proc, resources, glyph_ctm, self.xobject_depth + 1)

            total = widths[code] + cs
            if code == 32:
                total += ws
            advance = total * scale
            if decoder.is_vertical:
                self.tm_e += -advance * self.tm_c
                self.tm_f += -advance * self.tm_d
            else:
                self.tm_e += advance * self.tm_a
                self.tm_f += advance * self.tm_b

    def append_tj_array_simple(
        self: Any, array: list[Any] | tuple[Any, ...], decoder: FontDecoder
    ) -> None:
        table = decoder.byte_decode_table
        assert table is not None

        if self.cached_rotation == 0 and not decoder.is_vertical:
            self.append_tj_array_simple_horizontal_batched(array, decoder)
            return

        pending_data: bytes | bytearray | None = None
        text_scale = self.text_advance_scale
        adjustment_scale = text_scale
        is_vert = decoder.is_vertical
        widths = self.font_widths or decoder.fast_widths
        cs = self.char_space_scale
        ws = self.word_space_scale

        fs = self.font_size
        rise = self.rise
        ascent = self.font_ascent
        descent = self.font_descent
        A, B, C, D = self.combined_A, self.combined_B, self.combined_C, self.combined_D
        ca, cb, cc, cd, ce, cf = self.ca, self.cb, self.cc, self.cd, self.ce, self.cf
        ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d
        te, tf = self.tm_e, self.tm_f
        rot = self.cached_rotation
        scale_factor = hypot(C, D) if is_vert else hypot(A, B)
        effective_font_size = fs * scale_factor
        effective_space_width = self.font_space_width * scale_factor
        stream_order = self.stream_order
        xobject_depth = self.xobject_depth
        font_name = self.current_font
        fill_color = self.fill_color
        seqno = self.sequence
        pending_line_break = self.pending_line_break

        ar = ascent + rise
        dr = descent + rise
        dr_C = dr * C
        dr_D = dr * D
        ar_C = ar * C
        ar_D = ar * D

        for item in array:
            t = type(item)
            if t is bytes:
                item_data = item
                if pending_data is None:
                    pending_data = item_data
                elif type(pending_data) is bytearray:
                    pending_data.extend(item_data)
                else:
                    merged = bytearray(pending_data)
                    merged.extend(item_data)
                    pending_data = merged
                continue
            if t is PdfString:
                item_data = item.data
                if pending_data is None:
                    pending_data = item_data
                elif type(pending_data) is bytearray:
                    pending_data.extend(item_data)
                else:
                    merged = bytearray(pending_data)
                    merged.extend(item_data)
                    pending_data = merged
                continue

            if t is int or t is float:
                if pending_data:
                    n_data = len(pending_data)
                    if n_data == 1:
                        byte = pending_data[0]
                        text = table[byte]
                        total = widths[byte] + cs
                        if byte == 32:
                            total += ws
                    elif n_data == 2:
                        b0 = pending_data[0]
                        b1 = pending_data[1]
                        text = table[b0] + table[b1]
                        total = widths[b0] + widths[b1] + (2 * cs)
                        if b0 == 32:
                            total += ws
                        if b1 == 32:
                            total += ws
                    elif n_data == 3:
                        b0 = pending_data[0]
                        b1 = pending_data[1]
                        b2 = pending_data[2]
                        text = table[b0] + table[b1] + table[b2]
                        total = widths[b0] + widths[b1] + widths[b2] + (3 * cs)
                        if b0 == 32:
                            total += ws
                        if b1 == 32:
                            total += ws
                        if b2 == 32:
                            total += ws
                    else:
                        text = "".join(map(table.__getitem__, pending_data))
                        total = 0.0
                        space_count = 0
                        for byte in pending_data:
                            total += widths[byte]
                            if byte == 32:
                                space_count += 1
                        total += n_data * cs + space_count * ws

                    if text:
                        visible = self.is_text_visible(text)
                        if is_vert:
                            adv_x, adv_y = 0.0, -total * text_scale
                        else:
                            adv_x, adv_y = total * text_scale, 0.0

                        E = te * ca + tf * cc + ce
                        F = te * cb + tf * cd + cf
                        c0_x = dr_C + E
                        c0_y = dr_D + F
                        c1_x = ar_C + E
                        c1_y = ar_D + F
                        adv_A = adv_x * A
                        adv_B = adv_x * B
                        c2_x = adv_A + c0_x
                        c2_y = adv_B + c0_y
                        c3_x = adv_A + c1_x
                        c3_y = adv_B + c1_y

                        x0 = c0_x if c0_x < c1_x else c1_x
                        if c2_x < x0:
                            x0 = c2_x
                        if c3_x < x0:
                            x0 = c3_x
                        y0 = c0_y if c0_y < c1_y else c1_y
                        if c2_y < y0:
                            y0 = c2_y
                        if c3_y < y0:
                            y0 = c3_y
                        x1 = c0_x if c0_x > c1_x else c1_x
                        if c2_x > x1:
                            x1 = c2_x
                        if c3_x > x1:
                            x1 = c3_x
                        y1 = c0_y if c0_y > c1_y else c1_y
                        if c2_y > y1:
                            y1 = c2_y
                        if c3_y > y1:
                            y1 = c3_y

                        new_run = self.alloc_run(
                            text=text,
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                            tx=te,
                            ty=tf,
                            font_size=effective_font_size,
                            font_name=font_name,
                            space_width=effective_space_width,
                            order=seqno,
                            stream_order=stream_order,
                            xobject_depth=xobject_depth,
                            is_vertical=is_vert,
                            rotation_angle=rot,
                            visible=visible,
                            line_break_before=pending_line_break,
                            seqno=seqno,
                            fill_color=fill_color,
                        )
                        self.update_pending_run(new_run)
                        seqno += 1
                        pending_line_break = False
                        te = te + adv_x * ta + adv_y * tc
                        tf = tf + adv_x * tb + adv_y * td

                    pending_data = None

                delta = -item * adjustment_scale
                if is_vert:
                    te += delta * tc
                    tf += delta * td
                else:
                    te += delta * ta
                    tf += delta * tb
                continue

            if t is str:
                item_data = item.encode("latin-1")
                if pending_data is None:
                    pending_data = item_data
                elif type(pending_data) is bytearray:
                    pending_data.extend(item_data)
                else:
                    merged = bytearray(pending_data)
                    merged.extend(item_data)
                    pending_data = merged

        if pending_data:
            n_data = len(pending_data)
            if n_data == 1:
                byte = pending_data[0]
                text = table[byte]
                total = widths[byte] + cs
                if byte == 32:
                    total += ws
            elif n_data == 2:
                b0 = pending_data[0]
                b1 = pending_data[1]
                text = table[b0] + table[b1]
                total = widths[b0] + widths[b1] + (2 * cs)
                if b0 == 32:
                    total += ws
                if b1 == 32:
                    total += ws
            elif n_data == 3:
                b0 = pending_data[0]
                b1 = pending_data[1]
                b2 = pending_data[2]
                text = table[b0] + table[b1] + table[b2]
                total = widths[b0] + widths[b1] + widths[b2] + (3 * cs)
                if b0 == 32:
                    total += ws
                if b1 == 32:
                    total += ws
                if b2 == 32:
                    total += ws
            else:
                text = "".join(map(table.__getitem__, pending_data))
                total = 0.0
                space_count = 0
                for byte in pending_data:
                    total += widths[byte]
                    if byte == 32:
                        space_count += 1
                total += n_data * cs + space_count * ws
            if text:
                visible = self.is_text_visible(text)
                if is_vert:
                    adv_x, adv_y = 0.0, -total * text_scale
                else:
                    adv_x, adv_y = total * text_scale, 0.0
                E = te * ca + tf * cc + ce
                F = te * cb + tf * cd + cf
                c0_x = dr_C + E
                c0_y = dr_D + F
                c1_x = ar_C + E
                c1_y = ar_D + F
                adv_A = adv_x * A
                adv_B = adv_x * B
                c2_x = adv_A + c0_x
                c2_y = adv_B + c0_y
                c3_x = adv_A + c1_x
                c3_y = adv_B + c1_y

                x0 = c0_x if c0_x < c1_x else c1_x
                if c2_x < x0:
                    x0 = c2_x
                if c3_x < x0:
                    x0 = c3_x
                y0 = c0_y if c0_y < c1_y else c1_y
                if c2_y < y0:
                    y0 = c2_y
                if c3_y < y0:
                    y0 = c3_y
                x1 = c0_x if c0_x > c1_x else c1_x
                if c2_x > x1:
                    x1 = c2_x
                if c3_x > x1:
                    x1 = c3_x
                y1 = c0_y if c0_y > c1_y else c1_y
                if c2_y > y1:
                    y1 = c2_y
                if c3_y > y1:
                    y1 = c3_y

                new_run = self.alloc_run(
                    text=text,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    tx=te,
                    ty=tf,
                    font_size=effective_font_size,
                    font_name=font_name,
                    space_width=effective_space_width,
                    order=seqno,
                    stream_order=stream_order,
                    xobject_depth=xobject_depth,
                    is_vertical=is_vert,
                    rotation_angle=rot,
                    visible=visible,
                    line_break_before=pending_line_break,
                    seqno=seqno,
                    fill_color=fill_color,
                )
                self.update_pending_run(new_run)
                seqno += 1
                pending_line_break = False
                te = te + adv_x * ta + adv_y * tc
                tf = tf + adv_x * tb + adv_y * td

        self.tm_e, self.tm_f = te, tf
        self.sequence = seqno
        self.pending_line_break = pending_line_break

    def append_tj_array_simple_horizontal_batched(
        self: Any, array: list[Any] | tuple[Any, ...], decoder: FontDecoder
    ) -> None:
        table = decoder.byte_decode_table
        assert table is not None

        pending_data: bytes | bytearray | None = None
        widths = self.font_widths or decoder.fast_widths
        text_scale = self.text_advance_scale
        cs = self.char_space_scale
        ws = self.word_space_scale

        fs = self.font_size
        rise = self.rise
        ascent = self.font_ascent
        descent = self.font_descent
        A, B, C, D = self.combined_A, self.combined_B, self.combined_C, self.combined_D
        ca, cb, cc, cd, ce, cf = self.ca, self.cb, self.cc, self.cd, self.ce, self.cf
        ta, tb = self.tm_a, self.tm_b
        te, tf = self.tm_e, self.tm_f
        scale_factor = hypot(A, B)
        effective_font_size = fs * scale_factor
        effective_space_width = self.font_space_width * scale_factor
        merge_threshold = max(effective_space_width * 0.45, 2.0)
        stream_order = self.stream_order
        xobject_depth = self.xobject_depth
        font_name = self.current_font
        fill_color = self.fill_color
        seqno = self.sequence
        pending_line_break = self.pending_line_break

        ar = ascent + rise
        dr = descent + rise
        dr_C = dr * C
        dr_D = dr * D
        ar_C = ar * C
        ar_D = ar * D

        batch_parts: list[str] | None = None
        batch_last_char = ""
        batch_x0 = batch_y0 = batch_x1 = batch_y1 = 0.0
        batch_tx = batch_ty = 0.0
        batch_order = batch_seqno = 0
        batch_visible = True
        batch_line_break_before = False
        alloc_run = self.alloc_run
        update_pending_run = self.update_pending_run
        is_text_visible = self.is_text_visible
        no_space_before = NO_SPACE_BEFORE
        no_space_after = NO_SPACE_AFTER

        def flush_batch() -> None:
            nonlocal batch_parts, batch_last_char
            if batch_parts is None:
                return
            new_run = alloc_run(
                text="".join(batch_parts),
                x0=batch_x0,
                y0=batch_y0,
                x1=batch_x1,
                y1=batch_y1,
                tx=batch_tx,
                ty=batch_ty,
                font_size=effective_font_size,
                font_name=font_name,
                space_width=effective_space_width,
                order=batch_order,
                stream_order=stream_order,
                xobject_depth=xobject_depth,
                is_vertical=False,
                rotation_angle=0,
                visible=batch_visible,
                line_break_before=batch_line_break_before,
                seqno=batch_seqno,
                fill_color=fill_color,
            )
            update_pending_run(new_run)
            batch_parts = None
            batch_last_char = ""

        for item in array:
            t = type(item)
            if t is bytes:
                item_data = item
                if pending_data is None:
                    pending_data = item_data
                elif type(pending_data) is bytearray:
                    pending_data.extend(item_data)
                else:
                    merged = bytearray(pending_data)
                    merged.extend(item_data)
                    pending_data = merged
                continue
            if t is PdfString:
                item_data = item.data
                if pending_data is None:
                    pending_data = item_data
                elif type(pending_data) is bytearray:
                    pending_data.extend(item_data)
                else:
                    merged = bytearray(pending_data)
                    merged.extend(item_data)
                    pending_data = merged
                continue
            if t is str:
                item_data = item.encode("latin-1")
                if pending_data is None:
                    pending_data = item_data
                elif type(pending_data) is bytearray:
                    pending_data.extend(item_data)
                else:
                    merged = bytearray(pending_data)
                    merged.extend(item_data)
                    pending_data = merged
                continue

            if t is int or t is float:
                if pending_data:
                    n_data = len(pending_data)
                    if n_data == 1:
                        byte = pending_data[0]
                        text = table[byte]
                        total = widths[byte] + cs
                        if byte == 32:
                            total += ws
                    elif n_data == 2:
                        b0 = pending_data[0]
                        b1 = pending_data[1]
                        text = table[b0] + table[b1]
                        total = widths[b0] + widths[b1] + (2 * cs)
                        if b0 == 32:
                            total += ws
                        if b1 == 32:
                            total += ws
                    elif n_data == 3:
                        b0 = pending_data[0]
                        b1 = pending_data[1]
                        b2 = pending_data[2]
                        text = table[b0] + table[b1] + table[b2]
                        total = widths[b0] + widths[b1] + widths[b2] + (3 * cs)
                        if b0 == 32:
                            total += ws
                        if b1 == 32:
                            total += ws
                        if b2 == 32:
                            total += ws
                    else:
                        text = "".join(map(table.__getitem__, pending_data))
                        total = 0.0
                        space_count = 0
                        for byte in pending_data:
                            total += widths[byte]
                            if byte == 32:
                                space_count += 1
                        total += n_data * cs + space_count * ws

                    if text:
                        visible = self.is_text_visible(text)
                        adv_x = total * text_scale

                        E = te * ca + tf * cc + ce
                        F = te * cb + tf * cd + cf
                        c0_x = dr_C + E
                        c0_y = dr_D + F
                        c1_x = ar_C + E
                        c1_y = ar_D + F
                        adv_A = adv_x * A
                        adv_B = adv_x * B
                        c2_x = adv_A + c0_x
                        c2_y = adv_B + c0_y
                        c3_x = adv_A + c1_x
                        c3_y = adv_B + c1_y

                        x0 = c0_x if c0_x < c1_x else c1_x
                        if c2_x < x0:
                            x0 = c2_x
                        if c3_x < x0:
                            x0 = c3_x
                        y0 = c0_y if c0_y < c1_y else c1_y
                        if c2_y < y0:
                            y0 = c2_y
                        if c3_y < y0:
                            y0 = c3_y
                        x1 = c0_x if c0_x > c1_x else c1_x
                        if c2_x > x1:
                            x1 = c2_x
                        if c3_x > x1:
                            x1 = c3_x
                        y1 = c0_y if c0_y > c1_y else c1_y
                        if c2_y > y1:
                            y1 = c2_y
                        if c3_y > y1:
                            y1 = c3_y

                        if batch_parts is None:
                            batch_parts = [text]
                            batch_last_char = text[-1]
                            batch_x0, batch_y0, batch_x1, batch_y1 = x0, y0, x1, y1
                            batch_tx, batch_ty = te, tf
                            batch_order = seqno
                            batch_seqno = seqno
                            batch_visible = visible
                            batch_line_break_before = pending_line_break
                        else:
                            gap = x0 - batch_x1
                            if (
                                visible == batch_visible
                                and not pending_line_break
                                and abs(batch_y0 - y0) <= effective_font_size * 0.5
                                and -2.0 <= gap < merge_threshold
                            ):
                                threshold = effective_space_width * 0.12
                                font_threshold = effective_font_size * 0.10
                                if font_threshold > threshold:
                                    threshold = font_threshold
                                if threshold < 1.0:
                                    threshold = 1.0
                                if (
                                    gap <= threshold
                                    or batch_last_char.isspace()
                                    or text[0].isspace()
                                    or text[0] in no_space_before
                                    or batch_last_char in no_space_after
                                ):
                                    separator = ""
                                else:
                                    separator = " "
                                if separator:
                                    batch_parts.append(separator)
                                batch_parts.append(text)
                                batch_last_char = text[-1]
                                if x1 > batch_x1:
                                    batch_x1 = x1
                            else:
                                flush_batch()
                                batch_parts = [text]
                                batch_last_char = text[-1]
                                batch_x0, batch_y0, batch_x1, batch_y1 = x0, y0, x1, y1
                                batch_tx, batch_ty = te, tf
                                batch_order = seqno
                                batch_seqno = seqno
                                batch_visible = visible
                                batch_line_break_before = pending_line_break

                        seqno += 1
                        pending_line_break = False
                        te = te + adv_x * ta
                        tf = tf + adv_x * tb

                    pending_data = None

                delta = -item * text_scale
                te += delta * ta
                tf += delta * tb

        if pending_data:
            n_data = len(pending_data)
            if n_data == 1:
                byte = pending_data[0]
                text = table[byte]
                total = widths[byte] + cs
                if byte == 32:
                    total += ws
            elif n_data == 2:
                b0 = pending_data[0]
                b1 = pending_data[1]
                text = table[b0] + table[b1]
                total = widths[b0] + widths[b1] + (2 * cs)
                if b0 == 32:
                    total += ws
                if b1 == 32:
                    total += ws
            elif n_data == 3:
                b0 = pending_data[0]
                b1 = pending_data[1]
                b2 = pending_data[2]
                text = table[b0] + table[b1] + table[b2]
                total = widths[b0] + widths[b1] + widths[b2] + (3 * cs)
                if b0 == 32:
                    total += ws
                if b1 == 32:
                    total += ws
                if b2 == 32:
                    total += ws
            else:
                text = "".join(map(table.__getitem__, pending_data))
                total = 0.0
                space_count = 0
                for byte in pending_data:
                    total += widths[byte]
                    if byte == 32:
                        space_count += 1
                total += n_data * cs + space_count * ws

            if text:
                visible = is_text_visible(text)
                adv_x = total * text_scale
                E = te * ca + tf * cc + ce
                F = te * cb + tf * cd + cf
                c0_x = dr_C + E
                c0_y = dr_D + F
                c1_x = ar_C + E
                c1_y = ar_D + F
                adv_A = adv_x * A
                adv_B = adv_x * B
                c2_x = adv_A + c0_x
                c2_y = adv_B + c0_y
                c3_x = adv_A + c1_x
                c3_y = adv_B + c1_y

                x0 = c0_x if c0_x < c1_x else c1_x
                if c2_x < x0:
                    x0 = c2_x
                if c3_x < x0:
                    x0 = c3_x
                y0 = c0_y if c0_y < c1_y else c1_y
                if c2_y < y0:
                    y0 = c2_y
                if c3_y < y0:
                    y0 = c3_y
                x1 = c0_x if c0_x > c1_x else c1_x
                if c2_x > x1:
                    x1 = c2_x
                if c3_x > x1:
                    x1 = c3_x
                y1 = c0_y if c0_y > c1_y else c1_y
                if c2_y > y1:
                    y1 = c2_y
                if c3_y > y1:
                    y1 = c3_y

                if batch_parts is None:
                    batch_parts = [text]
                    batch_last_char = text[-1]
                    batch_x0, batch_y0, batch_x1, batch_y1 = x0, y0, x1, y1
                    batch_tx, batch_ty = te, tf
                    batch_order = seqno
                    batch_seqno = seqno
                    batch_visible = visible
                    batch_line_break_before = pending_line_break
                else:
                    gap = x0 - batch_x1
                    if (
                        visible == batch_visible
                        and not pending_line_break
                        and abs(batch_y0 - y0) <= effective_font_size * 0.5
                        and -2.0 <= gap < merge_threshold
                    ):
                        threshold = effective_space_width * 0.12
                        font_threshold = effective_font_size * 0.10
                        if font_threshold > threshold:
                            threshold = font_threshold
                        if threshold < 1.0:
                            threshold = 1.0
                        if (
                            gap <= threshold
                            or batch_last_char.isspace()
                            or text[0].isspace()
                            or text[0] in no_space_before
                            or batch_last_char in no_space_after
                        ):
                            separator = ""
                        else:
                            separator = " "
                        if separator:
                            batch_parts.append(separator)
                        batch_parts.append(text)
                        batch_last_char = text[-1]
                        if x1 > batch_x1:
                            batch_x1 = x1
                    else:
                        flush_batch()
                        batch_parts = [text]
                        batch_last_char = text[-1]
                        batch_x0, batch_y0, batch_x1, batch_y1 = x0, y0, x1, y1
                        batch_tx, batch_ty = te, tf
                        batch_order = seqno
                        batch_seqno = seqno
                        batch_visible = visible
                        batch_line_break_before = pending_line_break

                seqno += 1
                pending_line_break = False
                te = te + adv_x * ta
                tf = tf + adv_x * tb

        flush_batch()
        self.tm_e, self.tm_f = te, tf
        self.sequence = seqno
        self.pending_line_break = pending_line_break

    def append_tj_array(self: Any, array: Any) -> None:
        if not isinstance(array, (list, tuple)):
            return
        if not array:
            return
        pending_bytes = bytearray()
        scale = self.text_advance_scale

        decoder = self.current_decoder if self.current_decoder is not None else self.get_decoder()
        if (
            self.capture_runs
            and not self.capture_glyphs
            and self.current_actual_text_span() is None
            and decoder.byte_decode_table is not None
            and not decoder.is_cid_font
            and not decoder.is_vertical
            and not (decoder.is_type3 and self.capture_graphics)
            and decoder.to_unicode is None
            and decoder.cmap is None
        ):
            self.append_tj_array_simple(array, decoder)
            return
        is_vert = decoder.is_vertical
        zero_copy_flush = (
            not decoder.is_cid_font and decoder.to_unicode is None and decoder.cmap is None
        )

        te, tf = self.tm_e, self.tm_f
        ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d

        for item in array:
            t = type(item)
            if t is PdfString:
                pending_bytes.extend(item.data)
            elif t is bytes:
                pending_bytes.extend(item)
            elif t is int or t is float:
                if pending_bytes:
                    self.tm_e, self.tm_f = te, tf
                    if zero_copy_flush:
                        self.append_text(data=memoryview(pending_bytes), decoder=decoder)
                    else:
                        self.append_text(data=bytes(pending_bytes), decoder=decoder)
                    te, tf = self.tm_e, self.tm_f
                    pending_bytes.clear()
                delta = -item * scale
                if is_vert:
                    te += delta * tc
                    tf += delta * td
                else:
                    te += delta * ta
                    tf += delta * tb
            elif t is str:
                pending_bytes.extend(item.encode("latin-1"))

        if pending_bytes:
            self.tm_e, self.tm_f = te, tf
            if zero_copy_flush:
                self.append_text(data=memoryview(pending_bytes), decoder=decoder)
            else:
                self.append_text(data=bytes(pending_bytes), decoder=decoder)
            te, tf = self.tm_e, self.tm_f

        self.tm_e, self.tm_f = te, tf

    def current_actual_text_span(self: Any) -> Any | None:
        for entry in reversed(self.marked_content_stack):
            if getattr(entry, "actual_text", None) is not None:
                return entry
        return None

    def emit_actual_text_span(self: Any, entry: Any) -> None:
        actual_text = getattr(entry, "actual_text", None)
        if (
            actual_text is None
            or not getattr(entry, "has_text_extents", False)
            or not self.capture_runs
        ):
            return
        new_run = self.alloc_run(
            text=actual_text,
            x0=entry.x0,
            y0=entry.y0,
            x1=entry.x1,
            y1=entry.y1,
            tx=entry.tx,
            ty=entry.ty,
            font_size=entry.font_size,
            font_name=entry.font_name,
            space_width=entry.space_width,
            order=entry.order,
            stream_order=entry.stream_order,
            xobject_depth=entry.xobject_depth,
            is_vertical=entry.is_vertical,
            rotation_angle=entry.rotation_angle,
            visible=entry.visible,
            line_break_before=entry.line_break_before,
            seqno=entry.seqno,
            fill_color=entry.fill_color,
            advance_bbox=entry.advance_bbox,
            ink_bbox=entry.ink_bbox,
            baseline=entry.baseline,
            provenance=entry.provenance,
            confidence=entry.confidence,
        )
        self.update_pending_run(new_run)


__all__ = ("ContentCaptureMixin",)
