# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any, Protocol
lazy from importlib.resources import files

from core_pdf.impl.base.types import GeneratedRecord
from core_pdf.impl.fonts.helpers import recover_strip_subset_tag
from core_pdf.impl.spec.s_09_fonts.descriptor_flags import (
    FIXED_PITCH,
    NO_DESCRIPTOR_STYLE,
    SERIF,
    FontDescriptorStyle,
    descriptor_bold,
    descriptor_italic,
)
lazy from core_pdf.impl.fonts.program_truetype import TrueTypeFontProgram

# By name, not files(__package__): the fonts live in core_pdf.impl.data, not beside this module.
DATA_PACKAGE = "core_pdf.impl.data"


class PdfRasterFontRequest(GeneratedRecord):
    font_name: str | None
    text: str
    is_cid_font: bool
    is_vertical: bool
    cid_registry: str | None = None
    cid_ordering: str | None = None
    # What the font descriptor shows (ISO 32000-2 9.8.2): its flags, FontWeight and
    # ItalicAngle. False where it shows nothing; the font name may still say so.
    bold: bool = False
    italic: bool = False
    fixed_pitch: bool = False
    serif: bool = False


class PdfRasterFontFace(GeneratedRecord):
    identifier: str
    data: bytes


class PdfRasterFontProvider(Protocol):
    def resolve_raster_font(self, request: PdfRasterFontRequest) -> PdfRasterFontFace | None: ...


RasterFontProviderLike = (
    PdfRasterFontProvider | Callable[[PdfRasterFontRequest], PdfRasterFontFace | None]
)


class RasterFontRepository:
    __slots__ = ("provider", "builtin_programs", "provider_programs")

    def __init__(self, provider: RasterFontProviderLike | None = None) -> None:
        self.provider = provider
        self.builtin_programs: dict[str, TrueTypeFontProgram | None] = {}
        self.provider_programs: dict[str, TrueTypeFontProgram | None] = {}

    def load_provider_program(self, request: PdfRasterFontRequest) -> TrueTypeFontProgram | None:
        face = provider_face(self.provider, request)
        if face is None:
            return None
        if face.identifier not in self.provider_programs:
            try:
                program = TrueTypeFontProgram(face.data, use_cmap=True)
            except OSError, ValueError:
                program = None
            self.provider_programs[face.identifier] = program
        return self.provider_programs[face.identifier]

    def builtin_program(self, face_name: str) -> TrueTypeFontProgram | None:
        if face_name not in self.builtin_programs:
            try:
                program = builtin_font(face_name)
            except OSError, ValueError:
                program = None
            self.builtin_programs[face_name] = program
        return self.builtin_programs[face_name]

    def close(self) -> None:
        self.builtin_programs.clear()
        self.provider_programs.clear()


def provider_face(
    provider: RasterFontProviderLike | None, request: PdfRasterFontRequest
) -> PdfRasterFontFace | None:
    if provider is None:
        return None
    resolve: Any = getattr(provider, "resolve_raster_font", None) or provider
    face: PdfRasterFontFace | None = resolve(request)
    return face


def builtin_face_names(
    font_name: str | None, style: FontDescriptorStyle = NO_DESCRIPTOR_STYLE
) -> tuple[str, ...]:
    """The bundled faces to draw a font that is not embedded with: by the name, and, where
    the name gives no family or style, by the descriptor (ISO 32000-2 9.8.2): FixedPitch for a
    monospaced face, Serif for a serif one, ForceBold, a FontWeight of 600 or more, the Italic
    flag or an ItalicAngle for the style. The descriptor's FontFamily counts as name."""
    name = recover_strip_subset_tag(font_name or "").lower()
    if style.font_family:
        name = f"{name} {style.font_family.lower()}"
    if "zapfdingbats" in name:
        return ("NotoSansSymbols2-Regular.ttf",)
    if "symbol" in name:
        return (
            "NotoSansSymbols-Regular.ttf",
            "NotoSansSymbols2-Regular.ttf",
            "LiberationSerif-Regular.ttf",
        )
    bold = "bold" in name or descriptor_bold(style)
    italic = "italic" in name or "oblique" in name or descriptor_italic(style)
    if bold and italic:
        face_style = "BoldItalic"
    elif bold:
        face_style = "Bold"
    elif italic:
        face_style = "Italic"
    else:
        face_style = "Regular"
    if "courier" in name or "mono" in name:
        family = "LiberationMono"
    elif "times" in name or "serif" in name:
        family = "LiberationSerif"
    elif "helvetica" in name or "arial" in name or "sans" in name:
        family = "LiberationSans"
    elif style.has_flag(FIXED_PITCH):
        family = "LiberationMono"
    elif style.has_flag(SERIF):
        family = "LiberationSerif"
    else:
        family = "LiberationSans"
    return (f"{family}-{face_style}.ttf",)


@cache
def builtin_font(face_name: str) -> TrueTypeFontProgram:
    resource = files(DATA_PACKAGE).joinpath("raster_fonts", face_name)
    return TrueTypeFontProgram(resource.read_bytes(), use_cmap=True)


def fallback_glyph_outline(
    font_name: str | None,
    text: str,
    *,
    is_cid_font: bool,
    is_vertical: bool,
    cid_registry: str | None = None,
    cid_ordering: str | None = None,
    provider: RasterFontProviderLike | RasterFontRepository | None = None,
    style: FontDescriptorStyle = NO_DESCRIPTOR_STYLE,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    if len(text) != 1:
        return ()
    request = PdfRasterFontRequest(
        font_name,
        text,
        is_cid_font,
        is_vertical,
        cid_registry,
        cid_ordering,
        descriptor_bold(style),
        descriptor_italic(style),
        style.has_flag(FIXED_PITCH),
        style.has_flag(SERIF),
    )
    repository = (
        provider if isinstance(provider, RasterFontRepository) else RasterFontRepository(provider)
    )
    programs: list[TrueTypeFontProgram] = []
    provider_program = repository.load_provider_program(request)
    if provider_program is not None:
        programs.append(provider_program)
    for face_name in builtin_face_names(font_name, style):
        program = repository.builtin_program(face_name)
        if program is not None:
            programs.append(program)
    for program in programs:
        glyph_id = program.glyph_id_for_unicode(ord(text))
        if glyph_id != 0:
            return program.normalized_glyph_contours(glyph_id)
    return ()


__all__ = ("PdfRasterFontFace", "PdfRasterFontProvider", "PdfRasterFontRequest")
