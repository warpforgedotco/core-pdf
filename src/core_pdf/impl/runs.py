# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, ClassVar, Self, TypeAlias

from core_pdf.impl.geometry import extend_baseline
from core_pdf.impl.glyphs import GlyphClusterLike, min_optional_confidence
from core_pdf.impl.types import Record, ReprFields, frozen_setattr


class LayoutLineTextSegment(Record):
    __slots__ = ("text", "separator_before", "advance_bbox", "rotation_angle")

    text: str
    separator_before: str
    advance_bbox: tuple[float, float, float, float]
    rotation_angle: int

    __fields__: ClassVar[tuple[str, ...]] = (
        "text",
        "separator_before",
        "advance_bbox",
        "rotation_angle",
    )
    __match_args__ = ("text", "separator_before", "advance_bbox", "rotation_angle")

    def __init__(
        self,
        text: str,
        separator_before: str,
        advance_bbox: tuple[float, float, float, float],
        rotation_angle: int,
    ) -> None:
        frozen_setattr(self, "text", text)
        frozen_setattr(self, "separator_before", separator_before)
        frozen_setattr(self, "advance_bbox", advance_bbox)
        frozen_setattr(self, "rotation_angle", rotation_angle)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.text == other.text
            and self.separator_before == other.separator_before
            and self.advance_bbox == other.advance_bbox
            and self.rotation_angle == other.rotation_angle
        )

    def __hash__(self) -> int:
        return hash((self.text, self.separator_before, self.advance_bbox, self.rotation_angle))


class LayoutLineText(Record):
    __slots__ = ("text", "segments")

    text: str
    segments: tuple[LayoutLineTextSegment, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("text", "segments")
    __match_args__ = ("text", "segments")

    def __init__(self, text: str, segments: tuple[LayoutLineTextSegment, ...]) -> None:
        frozen_setattr(self, "text", text)
        frozen_setattr(self, "segments", segments)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.text == other.text and self.segments == other.segments

    def __hash__(self) -> int:
        return hash((self.text, self.segments))


EMPTY_LAYOUT_LINE_TEXT = LayoutLineText("", ())


Provenance: TypeAlias = tuple[tuple[str, object], ...]


class TextRun(ReprFields):
    __slots__ = (
        "text",
        "x0",
        "y0",
        "x1",
        "y1",
        "tx",
        "ty",
        "font_size",
        "space_width",
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

    __fields__: ClassVar[tuple[str, ...]] = (
        "text",
        "x0",
        "y0",
        "x1",
        "y1",
        "tx",
        "ty",
        "font_size",
        "space_width",
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
    __match_args__ = (
        "text",
        "x0",
        "y0",
        "x1",
        "y1",
        "tx",
        "ty",
        "font_size",
        "space_width",
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

    def __replace__(self, /, **changes: Any) -> Self:
        text = changes.pop("text", self.text)
        x0 = changes.pop("x0", self.x0)
        y0 = changes.pop("y0", self.y0)
        x1 = changes.pop("x1", self.x1)
        y1 = changes.pop("y1", self.y1)
        tx = changes.pop("tx", self.tx)
        ty = changes.pop("ty", self.ty)
        font_size = changes.pop("font_size", self.font_size)
        space_width = changes.pop("space_width", self.space_width)
        font_name = changes.pop("font_name", self.font_name)
        order = changes.pop("order", self.order)
        stream_order = changes.pop("stream_order", self.stream_order)
        xobject_depth = changes.pop("xobject_depth", self.xobject_depth)
        is_vertical = changes.pop("is_vertical", self.is_vertical)
        rotation_angle = changes.pop("rotation_angle", self.rotation_angle)
        visible = changes.pop("visible", self.visible)
        inside_active_clip = changes.pop("inside_active_clip", self.inside_active_clip)
        line_break_before = changes.pop("line_break_before", self.line_break_before)
        seqno = changes.pop("seqno", self.seqno)
        fill_color = changes.pop("fill_color", self.fill_color)
        advance_bbox = changes.pop("advance_bbox", self.advance_bbox)
        ink_bbox = changes.pop("ink_bbox", self.ink_bbox)
        baseline = changes.pop("baseline", self.baseline)
        provenance = changes.pop("provenance", self.provenance)
        confidence = changes.pop("confidence", self.confidence)
        glyph_clusters = changes.pop("glyph_clusters", self.glyph_clusters)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            text=text,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            tx=tx,
            ty=ty,
            font_size=font_size,
            space_width=space_width,
            font_name=font_name,
            order=order,
            stream_order=stream_order,
            xobject_depth=xobject_depth,
            is_vertical=is_vertical,
            rotation_angle=rotation_angle,
            visible=visible,
            inside_active_clip=inside_active_clip,
            line_break_before=line_break_before,
            seqno=seqno,
            fill_color=fill_color,
            advance_bbox=advance_bbox,
            ink_bbox=ink_bbox,
            baseline=baseline,
            provenance=provenance,
            confidence=confidence,
            glyph_clusters=glyph_clusters,
        )

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def stripped_text(self) -> str:
        text = self.text
        if text and text[0] > " " and text[-1] > " ":
            return text
        return text.strip()

    @property
    def has_text(self) -> bool:
        return bool(self.stripped_text)

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
        self.text = text
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

    def absorb_extent(self, other: TextRun) -> None:
        """Grow the box, advance box, baseline and confidence to cover `other`."""
        self.x0 = min(self.x0, other.x0)
        self.y0 = min(self.y0, other.y0)
        self.x1 = max(self.x1, other.x1)
        self.y1 = max(self.y1, other.y1)
        x0, y0, x1, y1 = self.advance_bbox
        bx0, by0, bx1, by1 = other.advance_bbox
        self.advance_bbox = (min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1))
        self.baseline = extend_baseline(self.baseline, other.baseline)
        self.confidence = min_optional_confidence(self.confidence, other.confidence)

    def union_ink_bbox(self, bbox: tuple[float, float, float, float]) -> None:
        x0, y0, x1, y1 = self.ink_bbox
        bx0, by0, bx1, by1 = bbox
        self.ink_bbox = (
            min(x0, bx0),
            min(y0, by0),
            max(x1, bx1),
            max(y1, by1),
        )

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
