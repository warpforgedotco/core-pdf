# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic raster-only font fallback for unembedded PDF fonts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Protocol, cast

from core_pdf.impl.engine.spec.s_09_fonts.font_program_truetype import TrueTypeFontProgram


@dataclass(frozen=True, slots=True)
class PdfRasterFontRequest:
    """Description passed to a custom renderer font provider."""

    font_name: str | None
    text: str
    is_cid_font: bool
    is_vertical: bool


@dataclass(frozen=True, slots=True)
class PdfRasterFontFace:
    """An in-memory TrueType font used only for raster rendering."""

    identifier: str
    data: bytes


class PdfRasterFontProvider(Protocol):
    """Resolve an unembedded font without changing extraction semantics."""

    def resolve_raster_font(self, request: PdfRasterFontRequest) -> PdfRasterFontFace | None: ...


RasterFontProviderLike = (
    PdfRasterFontProvider | Callable[[PdfRasterFontRequest], PdfRasterFontFace | None]
)


def internal_provider_face(
    provider: RasterFontProviderLike | None, request: PdfRasterFontRequest
) -> PdfRasterFontFace | None:
    if provider is None:
        return None
    resolver = getattr(provider, "resolve_raster_font", None)
    if resolver is not None:
        return resolver(request)
    callback = cast(Callable[[PdfRasterFontRequest], PdfRasterFontFace | None], provider)
    return callback(request)


def internal_builtin_face_name(font_name: str | None) -> str:
    name = (font_name or "").split("+", 1)[-1].lower()
    bold = "bold" in name
    italic = "italic" in name or "oblique" in name
    if bold and italic:
        style = "BoldItalic"
    elif bold:
        style = "Bold"
    elif italic:
        style = "Italic"
    else:
        style = "Regular"
    if "courier" in name or "mono" in name:
        family = "LiberationMono"
    elif "times" in name or "serif" in name or "symbol" in name or "zapfdingbats" in name:
        family = "LiberationSerif"
    else:
        family = "LiberationSans"
    return f"{family}-{style}.ttf"


@lru_cache(maxsize=12)
def internal_builtin_font(face_name: str) -> TrueTypeFontProgram:
    resource = files(__package__).joinpath("data", "raster_fonts", face_name)
    return TrueTypeFontProgram(resource.read_bytes(), use_cmap=True)


@lru_cache(maxsize=16)
def internal_custom_font(identifier: str, data: bytes) -> TrueTypeFontProgram:
    del identifier
    return TrueTypeFontProgram(data, use_cmap=True)


def fallback_glyph_outline(
    font_name: str | None,
    text: str,
    *,
    is_cid_font: bool,
    is_vertical: bool,
    provider: RasterFontProviderLike | None = None,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Resolve one Unicode scalar through a custom provider, then bundled fonts."""
    if len(text) != 1:
        return ()
    request = PdfRasterFontRequest(font_name, text, is_cid_font, is_vertical)
    face = internal_provider_face(provider, request)
    try:
        program = (
            internal_custom_font(face.identifier, face.data)
            if face is not None
            else internal_builtin_font(internal_builtin_face_name(font_name))
        )
    except (OSError, ValueError):
        return ()
    glyph_id = program.glyph_id_for_unicode(ord(text))
    if glyph_id == 0:
        return ()
    return program.normalized_glyph_contours(glyph_id)


__all__ = ("PdfRasterFontFace", "PdfRasterFontProvider", "PdfRasterFontRequest")
