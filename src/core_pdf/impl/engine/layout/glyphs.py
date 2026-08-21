# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from core_pdf.impl.engine.layout.geometry import bbox_union

BBox = tuple[float, float, float, float]
Matrix6 = tuple[float, float, float, float, float, float]


UNICODE_SOURCE_CONFIDENCE = {
    "actual_text": 1.0,
    "to_unicode": 1.0,
    "cff_glyph_repair": 0.96,
    "ligature_override": 0.94,
    "glyph_name": 0.92,
    "truetype_cmap": 0.90,
    "truetype_glyph_shape": 0.90,
    "predefined_cmap": 0.94,
    "learned_ocr": 0.93,
    "cid_collection": 0.88,
    "encoding": 0.84,
    "identity": 0.58,
    "fallback_nul": 0.05,
    "replacement": 0.05,
}

AUTHORITATIVE_UNICODE_SOURCES = frozenset(
    {
        "actual_text",
        "to_unicode",
        "cff_glyph_repair",
        "ligature_override",
        "glyph_name",
        "truetype_cmap",
        "truetype_glyph_shape",
        "predefined_cmap",
    }
)
HEURISTIC_UNICODE_SOURCES = frozenset({"cid_collection", "encoding", "learned_ocr"})


class GlyphUnicodeSemantics(StrEnum):
    """Whether glyph text is semantic Unicode or only a PDF character identifier."""

    AUTHORITATIVE = "authoritative"
    HEURISTIC = "heuristic"
    UNKNOWN_IDENTIFIER = "unknown-identifier"
    UNSUPPORTED = "unsupported"


@dataclass(slots=True)
class GlyphObservation:
    text: str
    ink_bbox: BBox
    advance_bbox: BBox
    seqno: int
    code_bytes: bytes = b""
    char_code: int | None = None
    cid: int | None = None
    gid: int | None = None
    font_name: str | None = None
    font_size: float = 0.0
    baseline: BBox | None = None
    rotation_angle: int = 0
    fill: tuple[float, ...] | None = None
    visible: bool = True
    confidence: float | None = None
    unicode_source: str = ""
    alternates: tuple[str, ...] = ()
    bitmap: tuple[int, ...] = ()
    bitmap_width: int = 0
    bitmap_height: int = 0
    bitmap_code: int | None = None
    font_decoder: object | None = None
    effective_font_size: float = 0.0
    effective_font_height: float = 0.0
    provenance: tuple[tuple[str, object], ...] = ()
    glyph_transform: Matrix6 | None = None
    text_render_mode: int = 0
    fill_opacity: float | None = None
    stroke_color: tuple[float, ...] | None = None
    stroke_opacity: float | None = None
    line_width: float = 1.0
    blend_mode: str | None = None
    soft_mask_alpha: float | None = None
    paint_glyph: bool = True

    @property
    def has_paint(self) -> bool:
        """Whether this glyph can contribute paint to a text-inclusive render."""
        return bool(
            self.bitmap
            or (
                self.paint_glyph
                and self.glyph_transform is not None
                and self.font_decoder is not None
            )
            or (
                self.font_decoder is not None
                and self.bitmap_code is not None
                and self.bitmap_width > 0
                and self.bitmap_height > 0
            )
        )

    def resolved_bitmap(self) -> tuple[int, ...]:
        """Resolve a glyph shape only when a text-inclusive renderer needs it.

        FontDecoder owns a per-font/glyph/size cache, so repeated glyph occurrences
        share one rasterized shape without page capture eagerly touching the cache.
        """
        if self.bitmap:
            return self.bitmap
        decoder = self.font_decoder
        code = self.bitmap_code
        resolver = getattr(decoder, "glyph_bitmap", None)
        if code is None or not callable(resolver):
            return ()
        return resolver(code, width=self.bitmap_width, height=self.bitmap_height)


@dataclass(slots=True)
class GlyphCluster:
    cluster_id: int
    text: str
    glyphs: tuple[GlyphObservation, ...]
    advance_bbox: BBox
    ink_bbox: BBox
    baseline: BBox | None
    confidence: float | None


@lru_cache(maxsize=512)
def glyph_unicode_confidence(
    text: str,
    unicode_source: str,
    alternates: tuple[str, ...] = (),
) -> float:
    """Estimate Unicode decoding confidence from mapping evidence."""
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


@lru_cache(maxsize=512)
def glyph_unicode_semantics(text: str, unicode_source: str) -> GlyphUnicodeSemantics:
    """Classify a decoded value without treating raw CIDs as real Unicode.

    Identity CMaps map character codes to glyph identifiers.  Their numeric value
    can happen to be a Unicode scalar, but that coincidence is not semantic text.
    Keeping that distinction explicit lets extraction retain identifiers for
    diagnostics and learning while preventing them from suppressing OCR.
    """
    if not text or glyph_text_has_unsupported_codepoint(text):
        return GlyphUnicodeSemantics.UNSUPPORTED
    if unicode_source in AUTHORITATIVE_UNICODE_SOURCES:
        return GlyphUnicodeSemantics.AUTHORITATIVE
    if unicode_source in HEURISTIC_UNICODE_SOURCES:
        return GlyphUnicodeSemantics.HEURISTIC
    return GlyphUnicodeSemantics.UNKNOWN_IDENTIFIER


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


def glyph_cluster_from_observations(
    cluster_id: int,
    text: str,
    glyphs: tuple[GlyphObservation, ...],
) -> GlyphCluster | None:
    if not glyphs:
        return None
    first = glyphs[0]
    if len(glyphs) == 1:
        advance_bbox = first.advance_bbox
        ink_bbox = first.ink_bbox
        confidence = first.confidence
    else:
        aggregated_advance_bbox = bbox_union(tuple(glyph.advance_bbox for glyph in glyphs))
        aggregated_ink_bbox = bbox_union(tuple(glyph.ink_bbox for glyph in glyphs))
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
        advance_bbox=advance_bbox,
        ink_bbox=ink_bbox,
        baseline=first.baseline,
        confidence=confidence,
    )
