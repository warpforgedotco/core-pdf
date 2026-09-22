# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from importlib.resources import files
from typing import ClassVar, Protocol, cast

from core_pdf.impl.fonts.font_program import TrueTypeFontProgram
from core_pdf.impl.fonts.helpers import strip_subset_tag
from core_pdf.impl.records import Record

frozen_setattr = object.__setattr__


class PdfRasterFontRequest(Record):
    __slots__ = ("font_name", "text", "is_cid_font", "is_vertical", "cid_registry", "cid_ordering")

    font_name: str | None
    text: str
    is_cid_font: bool
    is_vertical: bool
    cid_registry: str | None
    cid_ordering: str | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "font_name",
        "text",
        "is_cid_font",
        "is_vertical",
        "cid_registry",
        "cid_ordering",
    )
    __match_args__ = (
        "font_name",
        "text",
        "is_cid_font",
        "is_vertical",
        "cid_registry",
        "cid_ordering",
    )

    def __init__(
        self,
        font_name: str | None,
        text: str,
        is_cid_font: bool,
        is_vertical: bool,
        cid_registry: str | None = None,
        cid_ordering: str | None = None,
    ) -> None:
        frozen_setattr(self, "font_name", font_name)
        frozen_setattr(self, "text", text)
        frozen_setattr(self, "is_cid_font", is_cid_font)
        frozen_setattr(self, "is_vertical", is_vertical)
        frozen_setattr(self, "cid_registry", cid_registry)
        frozen_setattr(self, "cid_ordering", cid_ordering)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.font_name == other.font_name
            and self.text == other.text
            and self.is_cid_font == other.is_cid_font
            and self.is_vertical == other.is_vertical
            and self.cid_registry == other.cid_registry
            and self.cid_ordering == other.cid_ordering
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.font_name,
                self.text,
                self.is_cid_font,
                self.is_vertical,
                self.cid_registry,
                self.cid_ordering,
            )
        )


class PdfRasterFontFace(Record):
    __slots__ = ("identifier", "data")

    identifier: str
    data: bytes

    __fields__: ClassVar[tuple[str, ...]] = ("identifier", "data")
    __match_args__ = ("identifier", "data")

    def __init__(self, identifier: str, data: bytes) -> None:
        frozen_setattr(self, "identifier", identifier)
        frozen_setattr(self, "data", data)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.identifier == other.identifier and self.data == other.data

    def __hash__(self) -> int:
        return hash((self.identifier, self.data))


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
    resolver = getattr(provider, "resolve_raster_font", None)
    if resolver is not None:
        return resolver(request)
    callback = cast(Callable[[PdfRasterFontRequest], PdfRasterFontFace | None], provider)
    return callback(request)


def builtin_face_names(font_name: str | None) -> tuple[str, ...]:
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


@cache
def builtin_font(face_name: str) -> TrueTypeFontProgram:
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
    provider: RasterFontProviderLike | RasterFontRepository | None = None,
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
    )
    repository = (
        provider if isinstance(provider, RasterFontRepository) else RasterFontRepository(provider)
    )
    programs: list[TrueTypeFontProgram] = []
    provider_program = repository.load_provider_program(request)
    if provider_program is not None:
        programs.append(provider_program)
    for face_name in builtin_face_names(font_name):
        program = repository.builtin_program(face_name)
        if program is not None:
            programs.append(program)
    for program in programs:
        glyph_id = program.glyph_id_for_unicode(ord(text))
        if glyph_id != 0:
            return program.normalized_glyph_contours(glyph_id)
    return ()


__all__ = ("PdfRasterFontFace", "PdfRasterFontProvider", "PdfRasterFontRequest")
