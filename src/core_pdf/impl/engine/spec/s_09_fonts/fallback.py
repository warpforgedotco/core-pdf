# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic raster-only font fallback for unembedded PDF fonts."""

from __future__ import annotations

import contextlib
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
    cid_registry: str | None = None
    cid_ordering: str | None = None


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


def internal_builtin_face_names(font_name: str | None) -> tuple[str, ...]:
    name = (font_name or "").split("+", 1)[-1].lower()
    if "zapfdingbats" in name:
        return ("NotoSansSymbols2-Regular.ttf",)
    if "symbol" in name:
        return (
            "NotoSansSymbols-Regular.ttf",
            "NotoSansSymbols2-Regular.ttf",
            "LiberationSerif-Regular.ttf",
        )
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
    elif "times" in name or "serif" in name:
        family = "LiberationSerif"
    else:
        family = "LiberationSans"
    return (f"{family}-{style}.ttf",)


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
    cid_registry: str | None = None,
    cid_ordering: str | None = None,
    provider: RasterFontProviderLike | None = None,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Resolve one Unicode scalar through a custom provider, then bundled fonts."""
    if len(text) != 1:
        return ()
    request = PdfRasterFontRequest(
        font_name,
        text,
        is_cid_font,
        is_vertical,
        cid_registry,
        cid_ordering,
    )
    face = internal_provider_face(provider, request)
    programs: list[TrueTypeFontProgram] = []
    if face is not None:
        with contextlib.suppress(OSError, ValueError):
            programs.append(internal_custom_font(face.identifier, face.data))
    for face_name in internal_builtin_face_names(font_name):
        try:
            programs.append(internal_builtin_font(face_name))
        except (OSError, ValueError):
            continue
    for program in programs:
        glyph_id = program.glyph_id_for_unicode(ord(text))
        if glyph_id != 0:
            return program.normalized_glyph_contours(glyph_id)
    return ()


__all__ = ("PdfRasterFontFace", "PdfRasterFontProvider", "PdfRasterFontRequest")
