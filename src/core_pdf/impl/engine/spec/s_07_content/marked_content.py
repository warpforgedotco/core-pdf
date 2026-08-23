# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass, field

from core_pdf.impl.engine.spec.s_07_content.geometry import (
    extend_baseline,
    min_optional_confidence,
    union_bbox,
)
from core_pdf.impl.types import Rectangle

Provenance = tuple[tuple[str, object], ...]


@dataclass(slots=True)
class MarkedContentEntry:
    layer: str | None = None
    actual_text: str | None = None
    mcid: int | None = None
    has_text_extents: bool = False
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    nbytes: int = 0
    tx: float = 0.0
    ty: float = 0.0
    font_size: float = 0.0
    space_width: float = 0.0
    order: int = 0
    stream_order: int = 0
    xobject_depth: int = 0
    font_name: str | None = None
    is_vertical: bool = False
    rotation_angle: int = 0
    visible: bool = True
    line_break_before: bool = False
    seqno: int = 0
    fill_color: tuple[float, ...] | None = None
    advance_bbox: Rectangle | None = None
    ink_bbox: Rectangle | None = None
    baseline: Rectangle | None = None
    provenance: Provenance = ()
    confidence: float | None = None
    font_decoder: object | None = None
    effective_font_height: float = 0.0
    compatibility_text: str = ""
    compatibility_glyphs: list[object] = field(default_factory=list)

    def add_extents(
        self,
        *,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        nbytes: int,
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
        advance_bbox: Rectangle | None = None,
        ink_bbox: Rectangle | None = None,
        baseline: Rectangle | None = None,
        provenance: Provenance = (),
        confidence: float | None = None,
        font_decoder: object | None = None,
        effective_font_height: float = 0.0,
        compatibility_text: str = "",
    ) -> None:
        if self.has_text_extents:
            self.x0 = min(self.x0, x0)
            self.y0 = min(self.y0, y0)
            self.x1 = max(self.x1, x1)
            self.y1 = max(self.y1, y1)
            self.advance_bbox = union_bbox(self.advance_bbox, advance_bbox)
            self.ink_bbox = union_bbox(self.ink_bbox, ink_bbox)
            self.baseline = extend_baseline(self.baseline, baseline)
            self.confidence = min_optional_confidence(self.confidence, confidence)
            self.compatibility_text += compatibility_text
        else:
            self.x0 = x0
            self.y0 = y0
            self.x1 = x1
            self.y1 = y1
            self.tx = tx
            self.ty = ty
            self.font_size = font_size
            self.space_width = space_width
            self.order = order
            self.stream_order = stream_order
            self.xobject_depth = xobject_depth
            self.font_name = font_name
            self.is_vertical = is_vertical
            self.rotation_angle = rotation_angle
            self.visible = visible
            self.line_break_before = line_break_before
            self.seqno = seqno
            self.fill_color = fill_color
            self.advance_bbox = advance_bbox
            self.ink_bbox = ink_bbox
            self.baseline = baseline
            self.provenance = provenance
            self.confidence = confidence
            self.font_decoder = font_decoder
            self.effective_font_height = effective_font_height
            self.compatibility_text = compatibility_text
            self.has_text_extents = True
        self.nbytes += nbytes
