# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, Self, TypeAlias

from core_pdf.impl.geometry import extend_baseline, union_bbox
from core_pdf.impl.glyphs import GlyphClusterLike, min_optional_confidence
from core_pdf.impl.types import GeneratedRecord, RecordType, ReprFields


class LayoutLineTextSegment(GeneratedRecord):
    text: str
    separator_before: str
    advance_bbox: tuple[float, float, float, float]
    rotation_angle: int


class LayoutLineText(GeneratedRecord):
    text: str
    segments: tuple[LayoutLineTextSegment, ...]


EMPTY_LAYOUT_LINE_TEXT = LayoutLineText("", ())


Provenance: TypeAlias = tuple[tuple[str, object], ...]


class DerivedRunText:
    """Values derived from a run's text once, since layout reads them per comparison.
    Change a run's text with TextRun.set_text so they stay current."""

    __slots__ = ("stripped_text", "has_text")

    stripped_text: str
    has_text: bool


class TextRun(DerivedRunText, ReprFields, metaclass=RecordType, frozen=False, eq=False):
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    tx: float
    ty: float
    font_size: float
    space_width: float
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
    glyph_clusters: tuple[GlyphClusterLike, ...]

    def __replace__(self, /, **changes: Any) -> Self:
        values = {name: changes.pop(name, getattr(self, name)) for name in self.__fields__}
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(**values)

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def text_is_space(self) -> bool:
        return not self.has_text and self.text.isspace()

    @property
    def text_is_upper(self) -> bool:
        stripped = self.stripped_text
        return bool(stripped) and stripped.isupper()

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
        glyph_clusters: tuple[GlyphClusterLike, ...] = (),
    ) -> None:
        self.inside_active_clip = inside_active_clip
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.tx = tx
        self.ty = ty
        self.font_size = font_size
        self.space_width = space_width
        self.set_text(text)
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

    def set_text(self, text: str) -> None:
        self.text = text
        stripped = text if text and text[0] > " " and text[-1] > " " else text.strip()
        self.stripped_text = stripped
        self.has_text = bool(stripped)

    def absorb_extent(self, other: TextRun) -> None:
        self.x0 = min(self.x0, other.x0)
        self.y0 = min(self.y0, other.y0)
        self.x1 = max(self.x1, other.x1)
        self.y1 = max(self.y1, other.y1)
        self.advance_bbox = union_bbox(self.advance_bbox, other.advance_bbox)
        self.baseline = extend_baseline(self.baseline, other.baseline)
        self.confidence = min_optional_confidence(self.confidence, other.confidence)

    def union_ink_bbox(self, bbox: tuple[float, float, float, float]) -> None:
        self.ink_bbox = union_bbox(self.ink_bbox, bbox)

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

    def with_font_name(self, font_name: str | None) -> TextRun:
        return type(self)(
            self.text,
            self.x0,
            self.y0,
            self.x1,
            self.y1,
            self.tx,
            self.ty,
            self.font_size,
            self.space_width,
            self.order,
            self.stream_order,
            self.xobject_depth,
            font_name,
            self.is_vertical,
            self.rotation_angle,
            self.visible,
            self.inside_active_clip,
            self.line_break_before,
            self.seqno,
            self.fill_color,
            self.advance_bbox,
            self.ink_bbox,
            self.baseline,
            self.provenance,
            self.confidence,
            self.glyph_clusters,
        )

    def replace(self, **kwargs: Any) -> TextRun:
        coords_changed = any(key in kwargs for key in ("x0", "y0", "x1", "y1"))
        if coords_changed:
            box = (
                kwargs.get("x0", self.x0),
                kwargs.get("y0", self.y0),
                kwargs.get("x1", self.x1),
                kwargs.get("y1", self.y1),
            )
            kwargs.setdefault("advance_bbox", box)
            kwargs.setdefault("ink_bbox", box)
        position_changed = coords_changed or any(
            key in kwargs for key in ("tx", "ty", "rotation_angle")
        )
        if position_changed:
            kwargs.setdefault("baseline", None)
        text_changed = "text" in kwargs and kwargs["text"] != self.text
        if (
            position_changed
            or text_changed
            or any(key in kwargs for key in ("advance_bbox", "ink_bbox", "baseline"))
        ):
            kwargs.setdefault("glyph_clusters", ())
        take = kwargs.pop
        replaced = type(self)(
            take("text", self.text),
            take("x0", self.x0),
            take("y0", self.y0),
            take("x1", self.x1),
            take("y1", self.y1),
            take("tx", self.tx),
            take("ty", self.ty),
            take("font_size", self.font_size),
            take("space_width", self.space_width),
            take("order", self.order),
            take("stream_order", self.stream_order),
            take("xobject_depth", self.xobject_depth),
            take("font_name", self.font_name),
            take("is_vertical", self.is_vertical),
            take("rotation_angle", self.rotation_angle),
            take("visible", self.visible),
            take("inside_active_clip", self.inside_active_clip),
            take("line_break_before", self.line_break_before),
            take("seqno", self.seqno),
            take("fill_color", self.fill_color),
            take("advance_bbox", self.advance_bbox),
            take("ink_bbox", self.ink_bbox),
            take("baseline", self.baseline),
            take("provenance", self.provenance),
            take("confidence", self.confidence),
            take("glyph_clusters", self.glyph_clusters),
        )
        if kwargs:
            raise TypeError(f"unexpected TextRun field(s): {', '.join(sorted(kwargs))}")
        return replaced
