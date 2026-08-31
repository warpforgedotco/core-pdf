# SPDX-License-Identifier: AGPL-3.0-only
"""The capture text-run records: TextRun and its revision-tracked subclass.

TextRun memoizes two results the layout heuristics compute
(``internal_layout_reconstruction_cache``, ``internal_layout_words_cache``). Their
record types live beside this module in ``model/line_text.py``, so nothing here
names a type from ``layout/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

from core_pdf.impl.model.line_text import LayoutLineText, LayoutWordSnapshot

if TYPE_CHECKING:
    from core_pdf.impl.model.glyphs import GlyphCluster

Provenance: TypeAlias = tuple[tuple[str, object], ...]


class TextRun:
    __slots__ = (
        "internal_layout_reconstruction_cache",
        "internal_layout_words_cache",
        "internal_geometry_issues_cache",
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
    # A pending run may hold a list while merges accumulate; finalized runs
    # always carry the tuple form (see freeze_glyph_clusters).
    glyph_clusters: tuple[GlyphCluster, ...] | list[GlyphCluster]
    coords: list[float]
    mid_x_value: float
    mid_y_value: float
    height_value: float
    internal_revision: int
    internal_layout_reconstruction_cache: tuple[object, LayoutLineText] | None
    internal_layout_words_cache: tuple[object, tuple[str, tuple[LayoutWordSnapshot, ...]]] | None
    internal_geometry_issues_cache: tuple[object, tuple[object, ...]] | None

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
        glyph_clusters: tuple[GlyphCluster, ...] | list[GlyphCluster] = (),
    ) -> None:
        self.internal_revision = 0
        self.inside_active_clip = inside_active_clip
        self.internal_layout_reconstruction_cache = None
        self.internal_layout_words_cache = None
        self.internal_geometry_issues_cache = None
        self.coords = [x0, y0, x1, y1, tx, ty, font_size, space_width]
        self.mid_x_value = (x0 + x1) * 0.5
        self.mid_y_value = (y0 + y1) * 0.5
        self.height_value = y1 - y0
        self.text = text
        stripped_text = text.strip()
        self.stripped_text = stripped_text
        has_text = bool(stripped_text)
        self.has_text = has_text
        self.text_is_space = not has_text and text.isspace()
        self.text_is_upper = has_text and stripped_text.isupper()
        self.font_name = font_name
        self.order = order
        self.stream_order = stream_order
        self.xobject_depth = xobject_depth
        self.is_vertical = is_vertical
        self.rotation_angle = rotation_angle
        self.visible = visible
        self.line_break_before = line_break_before
        self.seqno = seqno
        self.fill_color = fill_color
        resolved_advance_bbox = advance_bbox or (x0, y0, x1, y1)
        self.advance_bbox = resolved_advance_bbox
        self.ink_bbox = ink_bbox or resolved_advance_bbox
        self.baseline = baseline
        self.provenance = provenance
        self.confidence = confidence
        self.glyph_clusters = glyph_clusters

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
        # While a run is pending, accumulate clusters in a list so repeated
        # merges stay linear; freeze_glyph_clusters restores the tuple form
        # (and a fresh identity for id()-keyed caches) at finalization.
        if not clusters:
            return
        existing = self.glyph_clusters
        if not existing:
            self.glyph_clusters = clusters
        elif type(existing) is list:
            existing.extend(clusters)
        else:
            combined = list(existing)
            combined.extend(clusters)
            self.glyph_clusters = combined

    def freeze_glyph_clusters(self) -> None:
        if type(self.glyph_clusters) is list:
            self.glyph_clusters = tuple(self.glyph_clusters)

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
        glyph_clusters: tuple[GlyphCluster, ...] | list[GlyphCluster] = (),
    ) -> TextRun:
        resolved_advance_bbox = advance_bbox or (x0, y0, x1, y1)
        resolved_ink_bbox = ink_bbox or resolved_advance_bbox
        if existing is not None:
            if type(existing) is not TextRun:
                existing.__class__ = TextRun
            existing.internal_layout_reconstruction_cache = None
            existing.internal_layout_words_cache = None
            existing.internal_geometry_issues_cache = None
            existing.internal_revision += 1
            c = existing.coords
            c[cls.X0] = x0
            c[cls.Y0] = y0
            c[cls.X1] = x1
            c[cls.Y1] = y1
            c[cls.TX] = tx
            c[cls.TY] = ty
            c[cls.FONT_SIZE] = font_size
            c[cls.SPACE_WIDTH] = space_width
            existing.text = text
            existing.font_name = font_name
            existing.order = order
            existing.stream_order = stream_order
            existing.xobject_depth = xobject_depth
            existing.is_vertical = is_vertical
            existing.rotation_angle = rotation_angle
            existing.visible = visible
            existing.inside_active_clip = True
            existing.line_break_before = line_break_before
            existing.seqno = seqno
            existing.fill_color = fill_color
            existing.advance_bbox = resolved_advance_bbox
            existing.ink_bbox = resolved_ink_bbox
            existing.baseline = baseline
            existing.provenance = provenance
            existing.confidence = confidence
            existing.glyph_clusters = glyph_clusters
            existing.mid_x_value = (x0 + x1) * 0.5
            existing.mid_y_value = (y0 + y1) * 0.5
            existing.height_value = y1 - y0
            if text and text[0] > " " and text[-1] > " ":
                stripped_text = text
            else:
                stripped_text = text.strip()
            existing.stripped_text = stripped_text
            has_text = bool(stripped_text)
            existing.has_text = has_text
            existing.text_is_space = not has_text and text.isspace()
            existing.text_is_upper = has_text and stripped_text.isupper()
            return existing
        r = object.__new__(TextRun)
        r.internal_revision = 0
        r.internal_layout_reconstruction_cache = None
        r.internal_layout_words_cache = None
        r.internal_geometry_issues_cache = None
        c = [x0, y0, x1, y1, tx, ty, font_size, space_width]
        r.coords = c
        r.mid_x_value = (x0 + x1) * 0.5
        r.mid_y_value = (y0 + y1) * 0.5
        r.height_value = y1 - y0
        r.text = text
        if text and text[0] > " " and text[-1] > " ":
            stripped_text = text
        else:
            stripped_text = text.strip()
        r.stripped_text = stripped_text
        has_text = bool(stripped_text)
        r.has_text = has_text
        r.text_is_space = not has_text and text.isspace()
        r.text_is_upper = has_text and stripped_text.isupper()
        r.font_name = font_name
        r.order = order
        r.stream_order = stream_order
        r.xobject_depth = xobject_depth
        r.is_vertical = is_vertical
        r.rotation_angle = rotation_angle
        r.visible = visible
        r.inside_active_clip = True
        r.line_break_before = line_break_before
        r.seqno = seqno
        r.fill_color = fill_color
        r.advance_bbox = resolved_advance_bbox
        r.ink_bbox = resolved_ink_bbox
        r.baseline = baseline
        r.provenance = provenance
        r.confidence = confidence
        r.glyph_clusters = glyph_clusters
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


class TrackedTextRun(TextRun):
    """A ``TextRun`` that maintains ``internal_revision`` for memo invalidation.

    Promoted into, never instantiated directly: ``__setattr__`` reads
    ``internal_revision``, which only exists on an already-initialized run.
    """

    __slots__ = ()

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name == "internal_revision":
            return
        revision = object.__getattribute__(self, "internal_revision")
        object.__setattr__(self, "internal_revision", revision + 1)


def internal_track_text_run(run: TextRun) -> None:
    """Promote a run so later attribute writes bump ``internal_revision``."""
    if type(run) is TextRun:
        run.__class__ = TrackedTextRun
