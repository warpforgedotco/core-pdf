# SPDX-License-Identifier: AGPL-3.0-only
"""Text-run records emitted by content capture."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

if TYPE_CHECKING:
    from core_pdf.impl._impl.model.glyphs import GlyphCluster


@dataclass(frozen=True, slots=True)
class LayoutLineTextSegment:
    text: str
    separator_before: str
    advance_bbox: tuple[float, float, float, float]
    rotation_angle: int


@dataclass(frozen=True, slots=True)
class LayoutLineText:
    text: str
    segments: tuple[LayoutLineTextSegment, ...]


EMPTY_LAYOUT_LINE_TEXT = LayoutLineText("", ())


Provenance: TypeAlias = tuple[tuple[str, object], ...]


@dataclass(slots=True, eq=False, init=False)
class TextRun:
    X0: ClassVar[int] = 0
    Y0: ClassVar[int] = 1
    X1: ClassVar[int] = 2
    Y1: ClassVar[int] = 3
    TX: ClassVar[int] = 4
    TY: ClassVar[int] = 5
    FONT_SIZE: ClassVar[int] = 6
    SPACE_WIDTH: ClassVar[int] = 7

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
    glyph_clusters: tuple[GlyphCluster, ...]

    @property
    def coords(self) -> tuple[float, float, float, float, float, float, float, float]:
        """Compatibility view of the former packed coordinate storage."""
        return (
            self.x0,
            self.y0,
            self.x1,
            self.y1,
            self.tx,
            self.ty,
            self.font_size,
            self.space_width,
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
        glyph_clusters: tuple[GlyphCluster, ...] = (),
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

    def union_ink_bbox(self, bbox: tuple[float, float, float, float]) -> None:
        x0, y0, x1, y1 = self.ink_bbox
        bx0, by0, bx1, by1 = bbox
        self.ink_bbox = (
            bx0 if bx0 < x0 else x0,
            by0 if by0 < y0 else y0,
            bx1 if bx1 > x1 else x1,
            by1 if by1 > y1 else y1,
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
        """Copy a run, discarding evidence invalidated by text or geometry edits."""
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
        return replace(self, **kwargs)
