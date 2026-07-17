# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass

from core_pdf.impl.engine.layout.geometry import RectBox

BBox = tuple[float, float, float, float]
Matrix6 = tuple[float, float, float, float, float, float]

UNICODE_SOURCE_CONFIDENCE = {
    "to_unicode": 1.0,
    "cff_glyph_repair": 0.96,
    "ligature_override": 0.94,
    "glyph_name": 0.92,
    "truetype_cmap": 0.90,
    "predefined_cmap": 0.94,
    "cid_collection": 0.88,
    "encoding": 0.84,
    "identity": 0.58,
    "fallback_nul": 0.05,
    "replacement": 0.05,
}


@dataclass(frozen=True, slots=True)
class GlyphObservation:
    text: str
    ink_rect: RectBox
    advance_rect: RectBox
    seqno: int
    code_bytes: bytes = b""
    char_code: int | None = None
    cid: int | None = None
    gid: int | None = None
    font_name: str | None = None
    font_size: float = 0.0
    space_width: float = 0.0
    text_matrix: Matrix6 | None = None
    device_matrix: Matrix6 | None = None
    baseline: BBox | None = None
    writing_mode: str = "horizontal"
    rotation_angle: int = 0
    stream_order: int = -1
    xobject_depth: int = 0
    fill: tuple[float, ...] | None = None
    visible: bool = True
    confidence: float | None = None
    unicode_source: str = ""
    alternates: tuple[str, ...] = ()
    cluster_id: int = -1
    cluster_index: int = 0
    cluster_size: int = 1
    bitmap: tuple[int, ...] = ()
    bitmap_width: int = 0
    bitmap_height: int = 0
    provenance: tuple[tuple[str, object], ...] = ()

    @property
    def ink_bbox(self) -> BBox:
        return rectbox_tuple(self.ink_rect)

    @property
    def advance_bbox(self) -> BBox:
        return rectbox_tuple(self.advance_rect)


@dataclass(frozen=True, slots=True)
class GlyphCluster:
    cluster_id: int
    text: str
    glyphs: tuple[GlyphObservation, ...]
    kind: str
    advance_bbox: BBox
    ink_bbox: BBox
    baseline: BBox | None
    writing_mode: str
    rotation_angle: int
    font_name: str | None
    seqno: int
    confidence: float | None
    provenance: tuple[tuple[str, object], ...] = ()


def rectbox_tuple(rect: RectBox) -> BBox:
    return (rect.x0, rect.y0, rect.x1, rect.y1)


def glyph_unicode_confidence(
    text: str,
    unicode_source: str,
    *,
    visible: bool = True,
    alternates: tuple[str, ...] = (),
) -> float:
    """Estimate Unicode decoding confidence independently of paint visibility.

    ``visible`` is retained for call compatibility and provenance symmetry.  A
    text-rendering mode or optional-content state says whether a glyph is
    painted, not whether its character mapping is correct.
    """
    del visible
    if not text:
        confidence = 0.0
    else:
        confidence = UNICODE_SOURCE_CONFIDENCE.get(unicode_source, 0.50)
        if any(
            alternate and not glyph_text_has_unsupported_codepoint(alternate)
            for alternate in alternates
        ):
            confidence = max(confidence, 0.68)
        if glyph_text_has_unsupported_codepoint(text):
            confidence = min(confidence, 0.20)
    return confidence


def glyph_text_has_unsupported_codepoint(text: str) -> bool:
    for char in text:
        codepoint = ord(char)
        if char in {"\ufffd", "\ufffc"}:
            return True
        if char == "\xad":
            return True
        if codepoint < 32 and char not in "\t\n\r":
            return True
        if 0xD800 <= codepoint <= 0xDFFF:
            return True
        if 0xE000 <= codepoint <= 0xF8FF:
            return True
    return False


def union_bboxes(boxes: tuple[BBox, ...]) -> BBox | None:
    if not boxes:
        return None
    x0, y0, x1, y1 = boxes[0]
    for bx0, by0, bx1, by1 in boxes[1:]:
        if bx0 < x0:
            x0 = bx0
        if by0 < y0:
            y0 = by0
        if bx1 > x1:
            x1 = bx1
        if by1 > y1:
            y1 = by1
    return (x0, y0, x1, y1)


def glyph_cluster_from_observations(
    cluster_id: int,
    text: str,
    glyphs: tuple[GlyphObservation, ...],
    *,
    kind: str,
    provenance: tuple[tuple[str, object], ...] = (),
) -> GlyphCluster | None:
    if not glyphs:
        return None
    first = glyphs[0]
    if len(glyphs) == 1:
        advance_bbox = first.advance_bbox
        ink_bbox = first.ink_bbox
        confidence = first.confidence
    else:
        aggregated_advance_bbox = union_bboxes(tuple(glyph.advance_bbox for glyph in glyphs))
        aggregated_ink_bbox = union_bboxes(tuple(glyph.ink_bbox for glyph in glyphs))
        if aggregated_advance_bbox is None or aggregated_ink_bbox is None:
            return None
        advance_bbox = aggregated_advance_bbox
        ink_bbox = aggregated_ink_bbox
        confidences = [glyph.confidence for glyph in glyphs if glyph.confidence is not None]
        confidence = min(confidences) if confidences else None
    return GlyphCluster(
        cluster_id=cluster_id,
        text=text,
        glyphs=glyphs,
        kind=kind,
        advance_bbox=advance_bbox,
        ink_bbox=ink_bbox,
        baseline=first.baseline,
        writing_mode=first.writing_mode,
        rotation_angle=first.rotation_angle,
        font_name=first.font_name,
        seqno=first.seqno,
        confidence=confidence,
        provenance=provenance,
    )
