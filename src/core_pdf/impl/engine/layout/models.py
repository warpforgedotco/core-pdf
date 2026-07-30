# SPDX-License-Identifier: AGPL-3.0-only
"""Shared page-space text, block, and table layout models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, TypeAlias

from core_pdf.impl.engine.layout.geometry import RectBox

if TYPE_CHECKING:
    from core_pdf.impl.engine.layout.glyphs import GlyphCluster
    from core_pdf.impl.engine.layout.text_lines import (
        LayoutLineText,
        LayoutLineTextSegment,
    )

Provenance: TypeAlias = tuple[tuple[str, object], ...]


class TextRun:
    __slots__ = (
        "internal_layout_reconstruction_cache",
        "internal_layout_words_cache",
        "internal_geometry_issues_cache",
        "internal_cache_tracking",
        "internal_revision",
        "text",
        "stripped_text",
        "has_text",
        "text_is_space",
        "text_is_upper",
        "coords",
        "mid_x_value",
        "mid_y_value",
        "height_value",
        "font_name",
        "order",
        "stream_order",
        "xobject_depth",
        "is_vertical",
        "rotation_angle",
        "visible",
        "inside_active_clip",
        "line_break_before",
        "seqno",
        "fill_color",
        "advance_bbox",
        "ink_bbox",
        "baseline",
        "provenance",
        "confidence",
        "glyph_clusters",
    )

    X0: ClassVar[int] = 0
    Y0: ClassVar[int] = 1
    X1: ClassVar[int] = 2
    Y1: ClassVar[int] = 3
    TX: ClassVar[int] = 4
    TY: ClassVar[int] = 5
    FONT_SIZE: ClassVar[int] = 6
    SPACE_WIDTH: ClassVar[int] = 7

    text: str
    stripped_text: str
    has_text: bool
    text_is_space: bool
    text_is_upper: bool
    font_name: str | None
    order: int
    stream_order: int
    xobject_depth: int
    is_vertical: bool
    rotation_angle: int
    visible: bool
    inside_active_clip: bool
    line_break_before: bool
    seqno: int
    fill_color: tuple[float, ...] | None
    advance_bbox: tuple[float, float, float, float]
    ink_bbox: tuple[float, float, float, float]
    baseline: tuple[float, float, float, float] | None
    provenance: Provenance
    confidence: float | None
    glyph_clusters: tuple[GlyphCluster, ...]
    coords: list[float]
    mid_x_value: float
    mid_y_value: float
    height_value: float
    internal_revision: int
    internal_layout_reconstruction_cache: tuple[object, LayoutLineText] | None
    internal_layout_words_cache: tuple[object, tuple[str, tuple[LayoutWordSnapshot, ...]]] | None
    internal_geometry_issues_cache: tuple[object, tuple[object, ...]] | None
    internal_cache_tracking: bool

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name == "internal_revision" or not object.__getattribute__(
            self, "internal_cache_tracking"
        ):
            return
        revision = object.__getattribute__(self, "internal_revision")
        object.__setattr__(self, "internal_revision", revision + 1)

    @property
    def x0(self) -> float:
        return self.coords[self.X0]

    @x0.setter
    def x0(self, v: float) -> None:
        self.coords[self.X0] = v
        self.mid_x_value = (v + self.coords[self.X1]) * 0.5
        self.internal_sync_advance_bbox()

    @property
    def y0(self) -> float:
        return self.coords[self.Y0]

    @y0.setter
    def y0(self, v: float) -> None:
        self.coords[self.Y0] = v
        self.mid_y_value = (v + self.coords[self.Y1]) * 0.5
        self.height_value = self.coords[self.Y1] - v
        self.internal_sync_advance_bbox()

    @property
    def x1(self) -> float:
        return self.coords[self.X1]

    @x1.setter
    def x1(self, v: float) -> None:
        self.coords[self.X1] = v
        self.mid_x_value = (self.coords[self.X0] + v) * 0.5
        self.internal_sync_advance_bbox()

    @property
    def y1(self) -> float:
        return self.coords[self.Y1]

    @y1.setter
    def y1(self, v: float) -> None:
        self.coords[self.Y1] = v
        self.mid_y_value = (self.coords[self.Y0] + v) * 0.5
        self.height_value = v - self.coords[self.Y0]
        self.internal_sync_advance_bbox()

    @property
    def tx(self) -> float:
        return self.coords[self.TX]

    @tx.setter
    def tx(self, v: float) -> None:
        self.coords[self.TX] = v
        self.internal_revision += 1

    @property
    def ty(self) -> float:
        return self.coords[self.TY]

    @ty.setter
    def ty(self, v: float) -> None:
        self.coords[self.TY] = v
        self.internal_revision += 1

    @property
    def font_size(self) -> float:
        return self.coords[self.FONT_SIZE]

    @font_size.setter
    def font_size(self, v: float) -> None:
        self.coords[self.FONT_SIZE] = v
        self.internal_revision += 1

    @property
    def space_width(self) -> float:
        return self.coords[self.SPACE_WIDTH]

    @space_width.setter
    def space_width(self, v: float) -> None:
        self.coords[self.SPACE_WIDTH] = v
        self.internal_revision += 1

    @property
    def mid_x(self) -> float:
        return self.mid_x_value

    @property
    def mid_y(self) -> float:
        return self.mid_y_value

    @property
    def height(self) -> float:
        return self.height_value

    def __init__(
        self,
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
        font_name: str | None = None,
        is_vertical: bool = False,
        rotation_angle: int = 0,
        visible: bool = True,
        inside_active_clip: bool = True,
        line_break_before: bool = False,
        seqno: int = -1,
        fill_color: tuple[float, ...] | None = None,
        advance_bbox: tuple[float, float, float, float] | None = None,
        ink_bbox: tuple[float, float, float, float] | None = None,
        baseline: tuple[float, float, float, float] | None = None,
        provenance: Provenance = (),
        confidence: float | None = None,
        glyph_clusters: tuple[GlyphCluster, ...] = (),
    ) -> None:
        set_untracked = object.__setattr__
        set_untracked(self, "internal_cache_tracking", False)
        set_untracked(self, "inside_active_clip", inside_active_clip)
        set_untracked(self, "internal_layout_reconstruction_cache", None)
        set_untracked(self, "internal_layout_words_cache", None)
        set_untracked(self, "internal_geometry_issues_cache", None)
        set_untracked(self, "internal_revision", 0)
        set_untracked(self, "coords", [x0, y0, x1, y1, tx, ty, font_size, space_width])
        set_untracked(self, "mid_x_value", (x0 + x1) * 0.5)
        set_untracked(self, "mid_y_value", (y0 + y1) * 0.5)
        set_untracked(self, "height_value", y1 - y0)
        set_untracked(self, "text", text)
        stripped_text = text.strip()
        set_untracked(self, "stripped_text", stripped_text)
        has_text = bool(stripped_text)
        set_untracked(self, "has_text", has_text)
        set_untracked(self, "text_is_space", not has_text and text.isspace())
        set_untracked(self, "text_is_upper", has_text and stripped_text.isupper())
        set_untracked(self, "font_name", font_name)
        set_untracked(self, "order", order)
        set_untracked(self, "stream_order", stream_order)
        set_untracked(self, "xobject_depth", xobject_depth)
        set_untracked(self, "is_vertical", is_vertical)
        set_untracked(self, "rotation_angle", rotation_angle)
        set_untracked(self, "visible", visible)
        set_untracked(self, "line_break_before", line_break_before)
        set_untracked(self, "seqno", seqno)
        set_untracked(self, "fill_color", fill_color)
        resolved_advance_bbox = advance_bbox or (x0, y0, x1, y1)
        set_untracked(self, "advance_bbox", resolved_advance_bbox)
        set_untracked(self, "ink_bbox", ink_bbox or resolved_advance_bbox)
        set_untracked(self, "baseline", baseline)
        set_untracked(self, "provenance", provenance)
        set_untracked(self, "confidence", confidence)
        set_untracked(self, "glyph_clusters", glyph_clusters)
        set_untracked(self, "internal_revision", 0)

    def internal_sync_advance_bbox(self) -> None:
        c = self.coords
        self.advance_bbox = (c[self.X0], c[self.Y0], c[self.X1], c[self.Y1])

    def union_ink_bbox(self, bbox: tuple[float, float, float, float]) -> None:
        x0, y0, x1, y1 = self.ink_bbox
        bx0, by0, bx1, by1 = bbox
        self.ink_bbox = (
            bx0 if bx0 < x0 else x0,
            by0 if by0 < y0 else y0,
            bx1 if bx1 > x1 else x1,
            by1 if by1 > y1 else y1,
        )

    def extend_glyph_clusters(self, clusters: tuple[GlyphCluster, ...]) -> None:
        if not clusters:
            return
        if not self.glyph_clusters:
            self.glyph_clusters = clusters
            return
        self.glyph_clusters = (*self.glyph_clusters, *clusters)

    def set_text(self, text: str) -> None:
        self.text = text
        if text and text[0] > " " and text[-1] > " ":
            stripped_text = text
        else:
            stripped_text = text.strip()
        self.stripped_text = stripped_text
        has_text = bool(stripped_text)
        self.has_text = has_text
        self.text_is_space = not has_text and text.isspace()
        self.text_is_upper = has_text and stripped_text.isupper()

    def is_bold(self) -> bool:
        if not self.font_name:
            return False
        fn = self.font_name.lower()
        return "bold" in fn or "black" in fn or "heavy" in fn

    def is_italic(self) -> bool:
        if not self.font_name:
            return False
        fn = self.font_name.lower()
        return "italic" in fn or "oblique" in fn or "slanted" in fn

    @classmethod
    def reinit(
        cls,
        existing: TextRun | None,
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
        provenance: Provenance = (),
        confidence: float | None = None,
        glyph_clusters: tuple[GlyphCluster, ...] = (),
    ) -> TextRun:
        resolved_advance_bbox = advance_bbox or (x0, y0, x1, y1)
        resolved_ink_bbox = ink_bbox or resolved_advance_bbox
        set_untracked = object.__setattr__
        if existing is not None:
            set_untracked(existing, "internal_cache_tracking", False)
            set_untracked(existing, "internal_layout_reconstruction_cache", None)
            set_untracked(existing, "internal_layout_words_cache", None)
            set_untracked(existing, "internal_geometry_issues_cache", None)
            c = existing.coords
            c[cls.X0] = x0
            c[cls.Y0] = y0
            c[cls.X1] = x1
            c[cls.Y1] = y1
            c[cls.TX] = tx
            c[cls.TY] = ty
            c[cls.FONT_SIZE] = font_size
            c[cls.SPACE_WIDTH] = space_width
            set_untracked(existing, "text", text)
            set_untracked(existing, "font_name", font_name)
            set_untracked(existing, "order", order)
            set_untracked(existing, "stream_order", stream_order)
            set_untracked(existing, "xobject_depth", xobject_depth)
            set_untracked(existing, "is_vertical", is_vertical)
            set_untracked(existing, "rotation_angle", rotation_angle)
            set_untracked(existing, "visible", visible)
            set_untracked(existing, "inside_active_clip", True)
            set_untracked(existing, "line_break_before", line_break_before)
            set_untracked(existing, "seqno", seqno)
            set_untracked(existing, "fill_color", fill_color)
            set_untracked(existing, "advance_bbox", resolved_advance_bbox)
            set_untracked(existing, "ink_bbox", resolved_ink_bbox)
            set_untracked(existing, "baseline", baseline)
            set_untracked(existing, "provenance", provenance)
            set_untracked(existing, "confidence", confidence)
            set_untracked(existing, "glyph_clusters", glyph_clusters)
            set_untracked(existing, "mid_x_value", (x0 + x1) * 0.5)
            set_untracked(existing, "mid_y_value", (y0 + y1) * 0.5)
            set_untracked(existing, "height_value", y1 - y0)
            if text and text[0] > " " and text[-1] > " ":
                stripped_text = text
            else:
                stripped_text = text.strip()
            set_untracked(existing, "stripped_text", stripped_text)
            has_text = bool(stripped_text)
            set_untracked(existing, "has_text", has_text)
            set_untracked(existing, "text_is_space", not has_text and text.isspace())
            set_untracked(existing, "text_is_upper", has_text and stripped_text.isupper())
            return existing
        r = object.__new__(cls)
        set_untracked(r, "internal_cache_tracking", False)
        set_untracked(r, "internal_layout_reconstruction_cache", None)
        set_untracked(r, "internal_layout_words_cache", None)
        set_untracked(r, "internal_geometry_issues_cache", None)
        set_untracked(r, "internal_revision", 0)
        c = [x0, y0, x1, y1, tx, ty, font_size, space_width]
        set_untracked(r, "coords", c)
        set_untracked(r, "mid_x_value", (x0 + x1) * 0.5)
        set_untracked(r, "mid_y_value", (y0 + y1) * 0.5)
        set_untracked(r, "height_value", y1 - y0)
        set_untracked(r, "text", text)
        if text and text[0] > " " and text[-1] > " ":
            stripped_text = text
        else:
            stripped_text = text.strip()
        set_untracked(r, "stripped_text", stripped_text)
        has_text = bool(stripped_text)
        set_untracked(r, "has_text", has_text)
        set_untracked(r, "text_is_space", not has_text and text.isspace())
        set_untracked(r, "text_is_upper", has_text and stripped_text.isupper())
        set_untracked(r, "font_name", font_name)
        set_untracked(r, "order", order)
        set_untracked(r, "stream_order", stream_order)
        set_untracked(r, "xobject_depth", xobject_depth)
        set_untracked(r, "is_vertical", is_vertical)
        set_untracked(r, "rotation_angle", rotation_angle)
        set_untracked(r, "visible", visible)
        set_untracked(r, "inside_active_clip", True)
        set_untracked(r, "line_break_before", line_break_before)
        set_untracked(r, "seqno", seqno)
        set_untracked(r, "fill_color", fill_color)
        set_untracked(r, "advance_bbox", resolved_advance_bbox)
        set_untracked(r, "ink_bbox", resolved_ink_bbox)
        set_untracked(r, "baseline", baseline)
        set_untracked(r, "provenance", provenance)
        set_untracked(r, "confidence", confidence)
        set_untracked(r, "glyph_clusters", glyph_clusters)
        set_untracked(r, "internal_revision", 0)
        return r

    def replace(self, **kwargs: Any) -> TextRun:
        x0 = kwargs.get("x0", self.x0)
        y0 = kwargs.get("y0", self.y0)
        x1 = kwargs.get("x1", self.x1)
        y1 = kwargs.get("y1", self.y1)
        coords_changed = any(key in kwargs for key in ("x0", "y0", "x1", "y1"))
        advance_bbox = kwargs.get(
            "advance_bbox",
            (x0, y0, x1, y1) if coords_changed else self.advance_bbox,
        )
        ink_bbox = kwargs.get(
            "ink_bbox",
            (x0, y0, x1, y1) if coords_changed else self.ink_bbox,
        )
        text_changed = "text" in kwargs and kwargs["text"] != self.text
        glyph_clusters = kwargs.get(
            "glyph_clusters",
            () if coords_changed or text_changed else self.glyph_clusters,
        )
        r = TextRun(
            text=kwargs.get("text", self.text),
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            tx=kwargs.get("tx", self.tx),
            ty=kwargs.get("ty", self.ty),
            font_size=kwargs.get("font_size", self.font_size),
            space_width=kwargs.get("space_width", self.space_width),
            order=kwargs.get("order", self.order),
            stream_order=kwargs.get("stream_order", self.stream_order),
            xobject_depth=kwargs.get("xobject_depth", self.xobject_depth),
            is_vertical=kwargs.get("is_vertical", self.is_vertical),
            rotation_angle=kwargs.get("rotation_angle", self.rotation_angle),
            visible=kwargs.get("visible", self.visible),
            inside_active_clip=kwargs.get("inside_active_clip", self.inside_active_clip),
            line_break_before=kwargs.get("line_break_before", self.line_break_before),
            seqno=kwargs.get("seqno", self.seqno),
            fill_color=kwargs.get("fill_color", self.fill_color),
            advance_bbox=advance_bbox,
            ink_bbox=ink_bbox,
            baseline=kwargs.get("baseline", self.baseline),
            provenance=kwargs.get("provenance", self.provenance),
            confidence=kwargs.get("confidence", self.confidence),
            glyph_clusters=glyph_clusters,
        )
        return r

    def with_coords(self, x0: float, y0: float, x1: float, y1: float) -> TextRun:
        r = object.__new__(TextRun)
        r.internal_cache_tracking = False
        r.internal_layout_reconstruction_cache = None
        r.internal_layout_words_cache = None
        r.internal_geometry_issues_cache = None
        r.internal_revision = 0
        coords = self.coords
        r.coords = [
            x0,
            y0,
            x1,
            y1,
            coords[self.TX],
            coords[self.TY],
            coords[self.FONT_SIZE],
            coords[self.SPACE_WIDTH],
        ]
        r.mid_x_value = (x0 + x1) * 0.5
        r.mid_y_value = (y0 + y1) * 0.5
        r.height_value = y1 - y0
        r.text = self.text
        r.inside_active_clip = self.inside_active_clip
        r.stripped_text = self.stripped_text
        r.has_text = self.has_text
        r.text_is_space = self.text_is_space
        r.text_is_upper = self.text_is_upper
        r.font_name = self.font_name
        r.order = self.order
        r.stream_order = self.stream_order
        r.xobject_depth = self.xobject_depth
        r.is_vertical = self.is_vertical
        r.rotation_angle = self.rotation_angle
        r.visible = self.visible
        r.line_break_before = self.line_break_before
        r.seqno = self.seqno
        r.fill_color = self.fill_color
        r.advance_bbox = (x0, y0, x1, y1)
        r.ink_bbox = (x0, y0, x1, y1)
        r.baseline = self.baseline
        r.provenance = self.provenance
        r.confidence = self.confidence
        r.glyph_clusters = ()
        r.internal_revision = 0
        return r


LayoutLineReconstructionKey: TypeAlias = tuple[
    bool,
    tuple[tuple[TextRun, int, tuple[float, ...]], ...],
]


class LayoutWordSnapshot(NamedTuple):
    text: str
    bbox: tuple[float, float, float, float]
    start_index: int


class LayoutWord:
    __slots__ = ("text", "bbox", "start_index")

    text: str
    bbox: tuple[float, float, float, float]
    start_index: int

    def __init__(
        self,
        text: str,
        bbox: tuple[float, float, float, float],
        start_index: int,
    ) -> None:
        self.text = text
        self.bbox = bbox
        self.start_index = start_index


class LayoutLine:
    __slots__ = (
        "internal_reconstructed_cache",
        "internal_reconstructed_cache_key",
        "runs",
        "x0",
        "y0",
        "x1",
        "y1",
        "is_vertical",
        "rotation_angle",
        "max_order",
        "max_depth",
        "min_order",
        "mid_y",
        "height",
        "max_font_size",
        "is_all_caps_text",
    )

    runs: list[TextRun]
    x0: float
    y0: float
    x1: float
    y1: float
    is_vertical: bool
    rotation_angle: int
    max_order: int
    max_depth: int
    min_order: int
    mid_y: float
    max_font_size: float
    is_all_caps_text: bool

    def __init__(
        self,
        runs: list[TextRun] | None = None,
        x0: float = 0.0,
        y0: float = 0.0,
        x1: float = 0.0,
        y1: float = 0.0,
        is_vertical: bool = False,
        rotation_angle: int = 0,
        max_order: int = -1,
        max_depth: int = -1,
        min_order: int = 999999,
        mid_y: float = 0.0,
        height: float = 0.0,
        max_font_size: float = 0.0,
        is_all_caps_text: bool = True,
    ) -> None:
        self.internal_reconstructed_cache: LayoutLineText | None = None
        self.internal_reconstructed_cache_key: LayoutLineReconstructionKey | None = None
        compute_from_runs = (
            runs is not None
            and len(runs) > 0
            and x0 == 0.0
            and y0 == 0.0
            and x1 == 0.0
            and y1 == 0.0
            and not is_vertical
            and rotation_angle == 0
            and max_order == -1
            and max_depth == -1
            and min_order == 999999
            and mid_y == 0.0
            and height == 0.0
            and max_font_size == 0.0
            and is_all_caps_text
        )
        if compute_from_runs and runs is not None:
            run_list = runs
            first_run = run_list[0]
            first_coords = first_run.coords
            x0 = first_coords[TextRun.X0]
            y0 = first_coords[TextRun.Y0]
            x1 = first_coords[TextRun.X1]
            y1 = first_coords[TextRun.Y1]
            max_order = first_run.order
            min_order = first_run.order
            max_depth = first_run.xobject_depth
            max_font_size = first_coords[TextRun.FONT_SIZE]
            is_all_caps_text = not first_run.has_text or first_run.text_is_upper

            text_run_x0 = TextRun.X0
            text_run_y0 = TextRun.Y0
            text_run_x1 = TextRun.X1
            text_run_y1 = TextRun.Y1
            text_run_font_size = TextRun.FONT_SIZE

            for run in run_list[1:]:
                coords = run.coords
                run_x0 = coords[text_run_x0]
                run_y0 = coords[text_run_y0]
                run_x1 = coords[text_run_x1]
                run_y1 = coords[text_run_y1]
                font_size = coords[text_run_font_size]
                if run_x0 < x0:
                    x0 = run_x0
                if run_y0 < y0:
                    y0 = run_y0
                if run_x1 > x1:
                    x1 = run_x1
                if run_y1 > y1:
                    y1 = run_y1
                if run.order > max_order:
                    max_order = run.order
                if run.order < min_order:
                    min_order = run.order
                if run.xobject_depth > max_depth:
                    max_depth = run.xobject_depth
                if font_size > max_font_size:
                    max_font_size = font_size
                if is_all_caps_text and run.has_text and not run.text_is_upper:
                    is_all_caps_text = False

            is_vertical = first_run.is_vertical
            rotation_angle = first_run.rotation_angle
            mid_y = (y0 + y1) * 0.5
            height = y1 - y0

        self.runs = runs if runs is not None else []
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.is_vertical = is_vertical
        self.rotation_angle = rotation_angle
        self.max_order = max_order
        self.max_depth = max_depth
        self.min_order = min_order
        self.mid_y = mid_y
        self.height = height
        self.max_font_size = max_font_size
        self.is_all_caps_text = is_all_caps_text

    def replace(self, **kwargs: Any) -> LayoutLine:
        return LayoutLine(
            runs=kwargs.get("runs", self.runs),
            x0=kwargs.get("x0", self.x0),
            y0=kwargs.get("y0", self.y0),
            x1=kwargs.get("x1", self.x1),
            y1=kwargs.get("y1", self.y1),
            is_vertical=kwargs.get("is_vertical", self.is_vertical),
            rotation_angle=kwargs.get("rotation_angle", self.rotation_angle),
            max_order=kwargs.get("max_order", self.max_order),
            max_depth=kwargs.get("max_depth", self.max_depth),
            min_order=kwargs.get("min_order", self.min_order),
            mid_y=kwargs.get("mid_y", self.mid_y),
            height=kwargs.get("height", self.height),
            max_font_size=kwargs.get("max_font_size", self.max_font_size),
            is_all_caps_text=kwargs.get("is_all_caps_text", self.is_all_caps_text),
        )

    def reconstruction_key(self) -> LayoutLineReconstructionKey:
        return (
            self.is_all_caps_text,
            tuple((run, run.internal_revision, tuple(run.coords)) for run in self.runs),
        )

    def reconstructed_text(self) -> LayoutLineText:
        from core_pdf.impl.engine.layout.text_lines import (
            reconstruct_layout_line_text,
        )

        key = self.reconstruction_key()
        cached = self.internal_reconstructed_cache
        if cached is not None and key == self.internal_reconstructed_cache_key:
            return cached
        first_run = self.runs[0] if self.runs else None
        shared_cache = (
            first_run.internal_layout_reconstruction_cache if first_run is not None else None
        )
        if shared_cache is not None and shared_cache[0] == key:
            reconstructed = shared_cache[1]
        else:
            reconstructed = reconstruct_layout_line_text(
                self.runs,
                is_all_caps_text=self.is_all_caps_text,
            )
            if first_run is not None:
                object.__setattr__(first_run, "internal_cache_tracking", True)
                object.__setattr__(
                    first_run, "internal_layout_reconstruction_cache", (key, reconstructed)
                )
        self.internal_reconstructed_cache = reconstructed
        self.internal_reconstructed_cache_key = key
        return reconstructed

    def text(self) -> str:
        return self.reconstructed_text().text

    def internal_build_text_and_words(self) -> tuple[str, tuple[LayoutWordSnapshot, ...]]:
        reconstructed = self.reconstructed_text()
        parts: list[str] = []
        words: list[LayoutWordSnapshot] = []
        word = ""
        word_start = 0
        text_len = 0
        word_x0 = word_y0 = word_x1 = word_y1 = 0.0
        append_part = parts.append
        append_word = words.append

        def flush_word() -> None:
            nonlocal word, word_x0, word_y0, word_x1, word_y1
            if not word:
                return
            append_word(LayoutWordSnapshot(word, (word_x0, word_y0, word_x1, word_y1), word_start))
            word = ""

        def extend_word(char: str, bbox: tuple[float, float, float, float]) -> None:
            nonlocal word, word_start, word_x0, word_y0, word_x1, word_y1
            if not word:
                word_start = text_len
                word_x0, word_y0, word_x1, word_y1 = bbox
            else:
                word_x0 = min(word_x0, bbox[0])
                word_y0 = min(word_y0, bbox[1])
                word_x1 = max(word_x1, bbox[2])
                word_y1 = max(word_y1, bbox[3])
            word += char

        def append_space() -> None:
            nonlocal text_len
            if parts and parts[-1] == " ":
                return
            flush_word()
            append_part(" ")
            text_len += 1

        for segment in reconstructed.segments:
            if segment.separator_before:
                append_space()
            text = segment.text
            text_length = len(text)
            for index, char in enumerate(text):
                bbox = layout_line_segment_char_bbox(segment, index, text_length)
                if char.isspace():
                    append_space()
                    continue
                if word and char.isalnum() != word[-1].isalnum():
                    flush_word()
                extend_word(char, bbox)
                append_part(char)
                text_len += 1

        flush_word()
        return "".join(parts).rstrip(), tuple(words)

    def cached_text_and_words(self) -> tuple[str, tuple[LayoutWordSnapshot, ...]]:
        key = self.reconstruction_key()
        first_run = self.runs[0] if self.runs else None
        cache = first_run.internal_layout_words_cache if first_run is not None else None
        if cache is not None and cache[0] == key:
            return cache[1]
        result = self.internal_build_text_and_words()
        if first_run is not None:
            object.__setattr__(first_run, "internal_cache_tracking", True)
            object.__setattr__(first_run, "internal_layout_words_cache", (key, result))
        return result

    def text_and_words(self) -> tuple[str, list[LayoutWord]]:
        text, snapshots = self.cached_text_and_words()
        return text, [LayoutWord(word.text, word.bbox, word.start_index) for word in snapshots]

    def words(self) -> list[LayoutWord]:
        return self.text_and_words()[1]


def layout_line_segment_char_bbox(
    segment: LayoutLineTextSegment,
    index: int,
    text_length: int,
) -> tuple[float, float, float, float]:
    if text_length <= 1:
        return segment.advance_bbox
    x0, y0, x1, y1 = segment.advance_bbox
    if segment.rotation_angle in (90, 270):
        step = (y1 - y0) / text_length
        char_y0 = y0 + step * index
        return (x0, char_y0, x1, char_y0 + step)
    step = (x1 - x0) / text_length
    char_x0 = x0 + step * index
    return (char_x0, y0, char_x0 + step, y1)


class LayoutBox:
    __slots__ = ("lines", "x0", "y0", "x1", "y1", "max_depth", "mid_y")

    lines: list[LayoutLine]
    x0: float
    y0: float
    x1: float
    y1: float
    max_depth: int
    mid_y: float

    def __init__(
        self,
        lines: list[LayoutLine] | None = None,
        x0: float = 0.0,
        y0: float = 0.0,
        x1: float = 0.0,
        y1: float = 0.0,
        max_depth: int = -1,
        mid_y: float = 0.0,
    ) -> None:
        self.lines = lines if lines is not None else []
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.max_depth = max_depth
        self.mid_y = mid_y

    def replace(self, **kwargs: Any) -> LayoutBox:
        return LayoutBox(
            lines=kwargs.get("lines", self.lines),
            x0=kwargs.get("x0", self.x0),
            y0=kwargs.get("y0", self.y0),
            x1=kwargs.get("x1", self.x1),
            y1=kwargs.get("y1", self.y1),
            max_depth=kwargs.get("max_depth", self.max_depth),
            mid_y=kwargs.get("mid_y", self.mid_y),
        )

    @property
    def bbox_rect(self) -> RectBox:
        return RectBox(self.x0, self.y0, self.x1, self.y1)

    def add(self, line: LayoutLine) -> None:
        if not self.lines:
            self.x0, self.y0, self.x1, self.y1 = (line.x0, line.y0, line.x1, line.y1)
            self.max_depth = line.max_depth
            self.mid_y = line.mid_y
        else:
            if line.x0 < self.x0:
                self.x0 = line.x0
            if line.y0 < self.y0:
                self.y0 = line.y0
            if line.x1 > self.x1:
                self.x1 = line.x1
            if line.y1 > self.y1:
                self.y1 = line.y1
            if line.max_depth > self.max_depth:
                self.max_depth = line.max_depth
            self.mid_y = (self.y0 + self.y1) * 0.5
        self.lines.append(line)


class TableGrid:
    __slots__ = ("cols", "rows")

    cols: list[float]
    rows: list[float]

    def __init__(self, cols: list[float], rows: list[float]) -> None:
        if len(cols) < 2 or len(rows) < 2:
            raise ValueError("invalid table grid")
        if not all(type(v) in (int, float) for v in cols):
            raise ValueError("invalid table grid")
        if not all(type(v) in (int, float) for v in rows):
            raise ValueError("invalid table grid")
        if any(cols[i] > cols[i + 1] for i in range(len(cols) - 1)):
            raise ValueError("invalid table grid")
        if any(rows[i] < rows[i + 1] for i in range(len(rows) - 1)):
            raise ValueError("invalid table grid")
        self.cols = cols
        self.rows = rows

    def is_valid(self) -> bool:
        return len(self.cols) >= 2 and len(self.rows) >= 2
