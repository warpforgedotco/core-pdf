# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from core_pdf.impl._impl.model.geometry import bbox_union
from core_pdf.impl.types import Rectangle

Matrix6 = tuple[float, float, float, float, float, float]


class UnicodeSource(StrEnum):
    ACTUAL_TEXT = "actual_text"
    TO_UNICODE = "to_unicode"
    CFF_GLYPH_REPAIR = "cff_glyph_repair"
    LIGATURE_OVERRIDE = "ligature_override"
    PREDEFINED_CMAP = "predefined_cmap"
    GLYPH_NAME = "glyph_name"
    TRUETYPE_CMAP = "truetype_cmap"
    TRUETYPE_GLYPH_SHAPE = "truetype_glyph_shape"
    CID_COLLECTION = "cid_collection"
    ENCODING = "encoding"
    IDENTITY = "identity"
    UNDEFINED = "undefined"
    FALLBACK_NUL = "fallback_nul"
    REPLACEMENT = "replacement"


UNICODE_SOURCE_CONFIDENCE: dict[str, float] = {
    UnicodeSource.ACTUAL_TEXT: 1.0,
    UnicodeSource.TO_UNICODE: 1.0,
    UnicodeSource.CFF_GLYPH_REPAIR: 0.96,
    UnicodeSource.LIGATURE_OVERRIDE: 0.94,
    UnicodeSource.GLYPH_NAME: 0.92,
    UnicodeSource.TRUETYPE_CMAP: 0.90,
    UnicodeSource.TRUETYPE_GLYPH_SHAPE: 0.90,
    UnicodeSource.PREDEFINED_CMAP: 0.94,
    UnicodeSource.CID_COLLECTION: 0.88,
    UnicodeSource.ENCODING: 0.84,
    UnicodeSource.IDENTITY: 0.58,
    UnicodeSource.UNDEFINED: 0.05,
    UnicodeSource.FALLBACK_NUL: 0.05,
    UnicodeSource.REPLACEMENT: 0.05,
}

AUTHORITATIVE_UNICODE_SOURCES: frozenset[str] = frozenset(
    {
        UnicodeSource.ACTUAL_TEXT,
        UnicodeSource.TO_UNICODE,
        UnicodeSource.CFF_GLYPH_REPAIR,
        UnicodeSource.LIGATURE_OVERRIDE,
        UnicodeSource.GLYPH_NAME,
        UnicodeSource.TRUETYPE_CMAP,
        UnicodeSource.TRUETYPE_GLYPH_SHAPE,
        UnicodeSource.PREDEFINED_CMAP,
    }
)
HEURISTIC_UNICODE_SOURCES: frozenset[str] = frozenset(
    {UnicodeSource.CID_COLLECTION, UnicodeSource.ENCODING}
)


class GlyphUnicodeSemantics(StrEnum):
    AUTHORITATIVE = "authoritative"
    HEURISTIC = "heuristic"
    UNKNOWN_IDENTIFIER = "unknown-identifier"
    UNSUPPORTED = "unsupported"


@dataclass(slots=True, eq=False)
class GlyphObservation:
    text: str
    ink_bbox: Rectangle
    advance_bbox: Rectangle
    seqno: int
    code_bytes: bytes = b""
    char_code: int | None = None
    cid: int | None = None
    gid: int | None = None
    font_name: str | None = None
    font_size: float = 0.0
    baseline: Rectangle | None = None
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
    text_object_id: int = 0
    line_cap: int = 0
    line_join: int = 0
    dash_pattern: tuple[list[float], float] | None = None
    cluster_key: tuple[int, int] | None = None
    clip_glyph: bool = False
    alpha_is_shape: bool = False
    paint_from_program: bool = False
    graphics_soft_mask: object | None = None

    @property
    def has_paint(self) -> bool:
        if self.paint_from_program:
            return False
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
        if self.bitmap:
            return self.bitmap
        decoder = self.font_decoder
        code = self.bitmap_code
        resolver = getattr(decoder, "glyph_bitmap", None)
        if code is None or not callable(resolver):
            return ()
        return resolver(code, width=self.bitmap_width, height=self.bitmap_height)


@dataclass(slots=True, eq=False)
class GlyphCluster:
    cluster_id: int
    text: str
    glyphs: tuple[GlyphObservation, ...]
    advance_bbox: Rectangle
    ink_bbox: Rectangle
    baseline: Rectangle | None
    confidence: float | None


internal_CONFIDENCE_CACHE: dict[tuple[str, str, tuple[str, ...]], float] = {}
internal_CONFIDENCE_CACHE_LIMIT = 8192


def glyph_unicode_confidence(
    text: str,
    unicode_source: str,
    alternates: tuple[str, ...] = (),
) -> float:
    key = (text, unicode_source, alternates)
    try:
        return internal_CONFIDENCE_CACHE[key]
    except KeyError:
        pass
    except TypeError:
        return internal_glyph_unicode_confidence(text, unicode_source, alternates)
    if len(internal_CONFIDENCE_CACHE) >= internal_CONFIDENCE_CACHE_LIMIT:
        internal_CONFIDENCE_CACHE.clear()
    confidence = internal_CONFIDENCE_CACHE[key] = internal_glyph_unicode_confidence(
        text, unicode_source, alternates
    )
    return confidence


def internal_glyph_unicode_confidence(
    text: str,
    unicode_source: str,
    alternates: tuple[str, ...],
) -> float:
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


def glyph_unicode_semantics(text: str, unicode_source: str) -> GlyphUnicodeSemantics:
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
