# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic raster-only font fallback for unembedded PDF fonts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from typing import Protocol, cast

from core_pdf.impl.spec.s_09_fonts.font_program_truetype import TrueTypeFontProgram
from core_pdf.impl.spec.s_09_fonts.helpers import strip_subset_tag


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


class internal_RasterFontRepository:
    """Document-owned parsed fonts used by raster fallback.

    Font programs are immutable after construction. Keeping them beside the
    document avoids both process-global state and reparsing a font for every
    painted glyph.
    """

    __slots__ = ("provider", "internal_builtin_programs", "internal_provider_programs")

    def __init__(self, provider: RasterFontProviderLike | None = None) -> None:
        self.provider = provider
        self.internal_builtin_programs: dict[str, TrueTypeFontProgram | None] = {}
        self.internal_provider_programs: dict[str, TrueTypeFontProgram | None] = {}

    def internal_provider_program(
        self, request: PdfRasterFontRequest
    ) -> TrueTypeFontProgram | None:
        face = internal_provider_face(self.provider, request)
        if face is None:
            return None
        if face.identifier not in self.internal_provider_programs:
            try:
                program = TrueTypeFontProgram(face.data, use_cmap=True)
            except (OSError, ValueError):
                program = None
            self.internal_provider_programs[face.identifier] = program
        return self.internal_provider_programs[face.identifier]

    def internal_builtin_program(self, face_name: str) -> TrueTypeFontProgram | None:
        if face_name not in self.internal_builtin_programs:
            try:
                program = internal_builtin_font(face_name)
            except (OSError, ValueError):
                program = None
            self.internal_builtin_programs[face_name] = program
        return self.internal_builtin_programs[face_name]

    def close(self) -> None:
        """Release all document-scoped references to parsed font programs."""
        self.internal_builtin_programs.clear()
        self.internal_provider_programs.clear()


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
    name = strip_subset_tag(font_name or "").lower()
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


def internal_builtin_font(face_name: str) -> TrueTypeFontProgram:
    resource = files(__package__).joinpath("data", "raster_fonts", face_name)
    return TrueTypeFontProgram(resource.read_bytes(), use_cmap=True)


def fallback_glyph_outline(
    font_name: str | None,
    text: str,
    *,
    is_cid_font: bool,
    is_vertical: bool,
    cid_registry: str | None = None,
    cid_ordering: str | None = None,
    provider: RasterFontProviderLike | internal_RasterFontRepository | None = None,
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
    repository = (
        provider
        if isinstance(provider, internal_RasterFontRepository)
        else internal_RasterFontRepository(provider)
    )
    programs: list[TrueTypeFontProgram] = []
    provider_program = repository.internal_provider_program(request)
    if provider_program is not None:
        programs.append(provider_program)
    for face_name in internal_builtin_face_names(font_name):
        program = repository.internal_builtin_program(face_name)
        if program is not None:
            programs.append(program)
    for program in programs:
        glyph_id = program.glyph_id_for_unicode(ord(text))
        if glyph_id != 0:
            return program.normalized_glyph_contours(glyph_id)
    return ()


__all__ = ("PdfRasterFontFace", "PdfRasterFontProvider", "PdfRasterFontRequest")
