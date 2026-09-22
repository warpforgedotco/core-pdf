# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar, Self

from core_pdf.impl.model.geometry import bbox_union
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


class GlyphObservation:
    __slots__ = (
        "text",
        "ink_bbox",
        "advance_bbox",
        "seqno",
        "code_bytes",
        "char_code",
        "cid",
        "gid",
        "font_name",
        "font_size",
        "baseline",
        "rotation_angle",
        "fill",
        "visible",
        "confidence",
        "unicode_source",
        "alternates",
        "bitmap",
        "bitmap_width",
        "bitmap_height",
        "bitmap_code",
        "font_decoder",
        "effective_font_size",
        "effective_font_height",
        "provenance",
        "glyph_transform",
        "text_render_mode",
        "fill_opacity",
        "stroke_color",
        "stroke_opacity",
        "line_width",
        "blend_mode",
        "soft_mask_alpha",
        "paint_glyph",
        "text_object_id",
        "line_cap",
        "line_join",
        "dash_pattern",
        "cluster_key",
        "clip_glyph",
        "alpha_is_shape",
        "paint_from_program",
        "graphics_soft_mask",
    )

    text: str
    ink_bbox: Rectangle
    advance_bbox: Rectangle
    seqno: int
    code_bytes: bytes
    char_code: int | None
    cid: int | None
    gid: int | None
    font_name: str | None
    font_size: float
    baseline: Rectangle | None
    rotation_angle: int
    fill: tuple[float, ...] | None
    visible: bool
    confidence: float | None
    unicode_source: str
    alternates: tuple[str, ...]
    bitmap: tuple[int, ...]
    bitmap_width: int
    bitmap_height: int
    bitmap_code: int | None
    font_decoder: object | None
    effective_font_size: float
    effective_font_height: float
    provenance: tuple[tuple[str, object], ...]
    glyph_transform: Matrix6 | None
    text_render_mode: int
    fill_opacity: float | None
    stroke_color: tuple[float, ...] | None
    stroke_opacity: float | None
    line_width: float
    blend_mode: str | None
    soft_mask_alpha: float | None
    paint_glyph: bool
    text_object_id: int
    line_cap: int
    line_join: int
    dash_pattern: tuple[list[float], float] | None
    cluster_key: tuple[int, int] | None
    clip_glyph: bool
    alpha_is_shape: bool
    paint_from_program: bool
    graphics_soft_mask: object | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "text",
        "ink_bbox",
        "advance_bbox",
        "seqno",
        "code_bytes",
        "char_code",
        "cid",
        "gid",
        "font_name",
        "font_size",
        "baseline",
        "rotation_angle",
        "fill",
        "visible",
        "confidence",
        "unicode_source",
        "alternates",
        "bitmap",
        "bitmap_width",
        "bitmap_height",
        "bitmap_code",
        "font_decoder",
        "effective_font_size",
        "effective_font_height",
        "provenance",
        "glyph_transform",
        "text_render_mode",
        "fill_opacity",
        "stroke_color",
        "stroke_opacity",
        "line_width",
        "blend_mode",
        "soft_mask_alpha",
        "paint_glyph",
        "text_object_id",
        "line_cap",
        "line_join",
        "dash_pattern",
        "cluster_key",
        "clip_glyph",
        "alpha_is_shape",
        "paint_from_program",
        "graphics_soft_mask",
    )
    __match_args__ = (
        "text",
        "ink_bbox",
        "advance_bbox",
        "seqno",
        "code_bytes",
        "char_code",
        "cid",
        "gid",
        "font_name",
        "font_size",
        "baseline",
        "rotation_angle",
        "fill",
        "visible",
        "confidence",
        "unicode_source",
        "alternates",
        "bitmap",
        "bitmap_width",
        "bitmap_height",
        "bitmap_code",
        "font_decoder",
        "effective_font_size",
        "effective_font_height",
        "provenance",
        "glyph_transform",
        "text_render_mode",
        "fill_opacity",
        "stroke_color",
        "stroke_opacity",
        "line_width",
        "blend_mode",
        "soft_mask_alpha",
        "paint_glyph",
        "text_object_id",
        "line_cap",
        "line_join",
        "dash_pattern",
        "cluster_key",
        "clip_glyph",
        "alpha_is_shape",
        "paint_from_program",
        "graphics_soft_mask",
    )

    def __init__(
        self,
        text: str,
        ink_bbox: Rectangle,
        advance_bbox: Rectangle,
        seqno: int,
        code_bytes: bytes = b"",
        char_code: int | None = None,
        cid: int | None = None,
        gid: int | None = None,
        font_name: str | None = None,
        font_size: float = 0.0,
        baseline: Rectangle | None = None,
        rotation_angle: int = 0,
        fill: tuple[float, ...] | None = None,
        visible: bool = True,
        confidence: float | None = None,
        unicode_source: str = "",
        alternates: tuple[str, ...] = (),
        bitmap: tuple[int, ...] = (),
        bitmap_width: int = 0,
        bitmap_height: int = 0,
        bitmap_code: int | None = None,
        font_decoder: object | None = None,
        effective_font_size: float = 0.0,
        effective_font_height: float = 0.0,
        provenance: tuple[tuple[str, object], ...] = (),
        glyph_transform: Matrix6 | None = None,
        text_render_mode: int = 0,
        fill_opacity: float | None = None,
        stroke_color: tuple[float, ...] | None = None,
        stroke_opacity: float | None = None,
        line_width: float = 1.0,
        blend_mode: str | None = None,
        soft_mask_alpha: float | None = None,
        paint_glyph: bool = True,
        text_object_id: int = 0,
        line_cap: int = 0,
        line_join: int = 0,
        dash_pattern: tuple[list[float], float] | None = None,
        cluster_key: tuple[int, int] | None = None,
        clip_glyph: bool = False,
        alpha_is_shape: bool = False,
        paint_from_program: bool = False,
        graphics_soft_mask: object | None = None,
    ) -> None:
        self.text = text
        self.ink_bbox = ink_bbox
        self.advance_bbox = advance_bbox
        self.seqno = seqno
        self.code_bytes = code_bytes
        self.char_code = char_code
        self.cid = cid
        self.gid = gid
        self.font_name = font_name
        self.font_size = font_size
        self.baseline = baseline
        self.rotation_angle = rotation_angle
        self.fill = fill
        self.visible = visible
        self.confidence = confidence
        self.unicode_source = unicode_source
        self.alternates = alternates
        self.bitmap = bitmap
        self.bitmap_width = bitmap_width
        self.bitmap_height = bitmap_height
        self.bitmap_code = bitmap_code
        self.font_decoder = font_decoder
        self.effective_font_size = effective_font_size
        self.effective_font_height = effective_font_height
        self.provenance = provenance
        self.glyph_transform = glyph_transform
        self.text_render_mode = text_render_mode
        self.fill_opacity = fill_opacity
        self.stroke_color = stroke_color
        self.stroke_opacity = stroke_opacity
        self.line_width = line_width
        self.blend_mode = blend_mode
        self.soft_mask_alpha = soft_mask_alpha
        self.paint_glyph = paint_glyph
        self.text_object_id = text_object_id
        self.line_cap = line_cap
        self.line_join = line_join
        self.dash_pattern = dash_pattern
        self.cluster_key = cluster_key
        self.clip_glyph = clip_glyph
        self.alpha_is_shape = alpha_is_shape
        self.paint_from_program = paint_from_program
        self.graphics_soft_mask = graphics_soft_mask

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"text={self.text!r}, "
            f"ink_bbox={self.ink_bbox!r}, "
            f"advance_bbox={self.advance_bbox!r}, "
            f"seqno={self.seqno!r}, "
            f"code_bytes={self.code_bytes!r}, "
            f"char_code={self.char_code!r}, "
            f"cid={self.cid!r}, "
            f"gid={self.gid!r}, "
            f"font_name={self.font_name!r}, "
            f"font_size={self.font_size!r}, "
            f"baseline={self.baseline!r}, "
            f"rotation_angle={self.rotation_angle!r}, "
            f"fill={self.fill!r}, "
            f"visible={self.visible!r}, "
            f"confidence={self.confidence!r}, "
            f"unicode_source={self.unicode_source!r}, "
            f"alternates={self.alternates!r}, "
            f"bitmap={self.bitmap!r}, "
            f"bitmap_width={self.bitmap_width!r}, "
            f"bitmap_height={self.bitmap_height!r}, "
            f"bitmap_code={self.bitmap_code!r}, "
            f"font_decoder={self.font_decoder!r}, "
            f"effective_font_size={self.effective_font_size!r}, "
            f"effective_font_height={self.effective_font_height!r}, "
            f"provenance={self.provenance!r}, "
            f"glyph_transform={self.glyph_transform!r}, "
            f"text_render_mode={self.text_render_mode!r}, "
            f"fill_opacity={self.fill_opacity!r}, "
            f"stroke_color={self.stroke_color!r}, "
            f"stroke_opacity={self.stroke_opacity!r}, "
            f"line_width={self.line_width!r}, "
            f"blend_mode={self.blend_mode!r}, "
            f"soft_mask_alpha={self.soft_mask_alpha!r}, "
            f"paint_glyph={self.paint_glyph!r}, "
            f"text_object_id={self.text_object_id!r}, "
            f"line_cap={self.line_cap!r}, "
            f"line_join={self.line_join!r}, "
            f"dash_pattern={self.dash_pattern!r}, "
            f"cluster_key={self.cluster_key!r}, "
            f"clip_glyph={self.clip_glyph!r}, "
            f"alpha_is_shape={self.alpha_is_shape!r}, "
            f"paint_from_program={self.paint_from_program!r}, "
            f"graphics_soft_mask={self.graphics_soft_mask!r}"
            ")"
        )

    def __replace__(self, /, **changes: Any) -> Self:
        text = changes.pop("text", self.text)
        ink_bbox = changes.pop("ink_bbox", self.ink_bbox)
        advance_bbox = changes.pop("advance_bbox", self.advance_bbox)
        seqno = changes.pop("seqno", self.seqno)
        code_bytes = changes.pop("code_bytes", self.code_bytes)
        char_code = changes.pop("char_code", self.char_code)
        cid = changes.pop("cid", self.cid)
        gid = changes.pop("gid", self.gid)
        font_name = changes.pop("font_name", self.font_name)
        font_size = changes.pop("font_size", self.font_size)
        baseline = changes.pop("baseline", self.baseline)
        rotation_angle = changes.pop("rotation_angle", self.rotation_angle)
        fill = changes.pop("fill", self.fill)
        visible = changes.pop("visible", self.visible)
        confidence = changes.pop("confidence", self.confidence)
        unicode_source = changes.pop("unicode_source", self.unicode_source)
        alternates = changes.pop("alternates", self.alternates)
        bitmap = changes.pop("bitmap", self.bitmap)
        bitmap_width = changes.pop("bitmap_width", self.bitmap_width)
        bitmap_height = changes.pop("bitmap_height", self.bitmap_height)
        bitmap_code = changes.pop("bitmap_code", self.bitmap_code)
        font_decoder = changes.pop("font_decoder", self.font_decoder)
        effective_font_size = changes.pop("effective_font_size", self.effective_font_size)
        effective_font_height = changes.pop("effective_font_height", self.effective_font_height)
        provenance = changes.pop("provenance", self.provenance)
        glyph_transform = changes.pop("glyph_transform", self.glyph_transform)
        text_render_mode = changes.pop("text_render_mode", self.text_render_mode)
        fill_opacity = changes.pop("fill_opacity", self.fill_opacity)
        stroke_color = changes.pop("stroke_color", self.stroke_color)
        stroke_opacity = changes.pop("stroke_opacity", self.stroke_opacity)
        line_width = changes.pop("line_width", self.line_width)
        blend_mode = changes.pop("blend_mode", self.blend_mode)
        soft_mask_alpha = changes.pop("soft_mask_alpha", self.soft_mask_alpha)
        paint_glyph = changes.pop("paint_glyph", self.paint_glyph)
        text_object_id = changes.pop("text_object_id", self.text_object_id)
        line_cap = changes.pop("line_cap", self.line_cap)
        line_join = changes.pop("line_join", self.line_join)
        dash_pattern = changes.pop("dash_pattern", self.dash_pattern)
        cluster_key = changes.pop("cluster_key", self.cluster_key)
        clip_glyph = changes.pop("clip_glyph", self.clip_glyph)
        alpha_is_shape = changes.pop("alpha_is_shape", self.alpha_is_shape)
        paint_from_program = changes.pop("paint_from_program", self.paint_from_program)
        graphics_soft_mask = changes.pop("graphics_soft_mask", self.graphics_soft_mask)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            text,
            ink_bbox,
            advance_bbox,
            seqno,
            code_bytes,
            char_code,
            cid,
            gid,
            font_name,
            font_size,
            baseline,
            rotation_angle,
            fill,
            visible,
            confidence,
            unicode_source,
            alternates,
            bitmap,
            bitmap_width,
            bitmap_height,
            bitmap_code,
            font_decoder,
            effective_font_size,
            effective_font_height,
            provenance,
            glyph_transform,
            text_render_mode,
            fill_opacity,
            stroke_color,
            stroke_opacity,
            line_width,
            blend_mode,
            soft_mask_alpha,
            paint_glyph,
            text_object_id,
            line_cap,
            line_join,
            dash_pattern,
            cluster_key,
            clip_glyph,
            alpha_is_shape,
            paint_from_program,
            graphics_soft_mask,
        )

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


class GlyphCluster:
    __slots__ = (
        "cluster_id",
        "text",
        "glyphs",
        "advance_bbox",
        "ink_bbox",
        "baseline",
        "confidence",
    )

    cluster_id: int
    text: str
    glyphs: tuple[GlyphObservation, ...]
    advance_bbox: Rectangle
    ink_bbox: Rectangle
    baseline: Rectangle | None
    confidence: float | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "cluster_id",
        "text",
        "glyphs",
        "advance_bbox",
        "ink_bbox",
        "baseline",
        "confidence",
    )
    __match_args__ = (
        "cluster_id",
        "text",
        "glyphs",
        "advance_bbox",
        "ink_bbox",
        "baseline",
        "confidence",
    )

    def __init__(
        self,
        cluster_id: int,
        text: str,
        glyphs: tuple[GlyphObservation, ...],
        advance_bbox: Rectangle,
        ink_bbox: Rectangle,
        baseline: Rectangle | None,
        confidence: float | None,
    ) -> None:
        self.cluster_id = cluster_id
        self.text = text
        self.glyphs = glyphs
        self.advance_bbox = advance_bbox
        self.ink_bbox = ink_bbox
        self.baseline = baseline
        self.confidence = confidence

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"cluster_id={self.cluster_id!r}, "
            f"text={self.text!r}, "
            f"glyphs={self.glyphs!r}, "
            f"advance_bbox={self.advance_bbox!r}, "
            f"ink_bbox={self.ink_bbox!r}, "
            f"baseline={self.baseline!r}, "
            f"confidence={self.confidence!r}"
            ")"
        )

    def __replace__(self, /, **changes: Any) -> Self:
        cluster_id = changes.pop("cluster_id", self.cluster_id)
        text = changes.pop("text", self.text)
        glyphs = changes.pop("glyphs", self.glyphs)
        advance_bbox = changes.pop("advance_bbox", self.advance_bbox)
        ink_bbox = changes.pop("ink_bbox", self.ink_bbox)
        baseline = changes.pop("baseline", self.baseline)
        confidence = changes.pop("confidence", self.confidence)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            cluster_id,
            text,
            glyphs,
            advance_bbox,
            ink_bbox,
            baseline,
            confidence,
        )


CONFIDENCE_CACHE: dict[tuple[str, str, tuple[str, ...]], float] = {}
CONFIDENCE_CACHE_LIMIT = 8192


def glyph_unicode_confidence(
    text: str,
    unicode_source: str,
    alternates: tuple[str, ...] = (),
) -> float:
    key = (text, unicode_source, alternates)
    try:
        return CONFIDENCE_CACHE[key]
    except KeyError:
        pass
    except TypeError:
        return compute_glyph_unicode_confidence(text, unicode_source, alternates)
    if len(CONFIDENCE_CACHE) >= CONFIDENCE_CACHE_LIMIT:
        CONFIDENCE_CACHE.clear()
    confidence = CONFIDENCE_CACHE[key] = compute_glyph_unicode_confidence(
        text, unicode_source, alternates
    )
    return confidence


def compute_glyph_unicode_confidence(
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
    if text.isascii() and text.isprintable():
        return False
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
        cluster_id, text, glyphs, advance_bbox, ink_bbox, first.baseline, confidence
    )
