from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core_pdf import PdfPage
from core_pdf.impl.capture.glyphs import GlyphPaint
from core_pdf.impl.capture.program import CapturedProgram, CaptureOptions
from core_pdf.impl.capture.recording import TextState
from core_pdf.impl.capture.recovery import CaptureRecovery
from core_pdf.impl.document.recovery.lexer import PdfLexer
from core_pdf.impl.exceptions import PdfError, PdfParseError
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.types import PdfString, Rectangle
from core_pdf_spec.s_07_content.operations import ContentOperands
from core_pdf_spec.s_07_content.streams import ContentStreamFrame
from core_pdf_spec.s_07_syntax.lexer import PdfLexer as SyntaxLexer
from core_pdf_spec.s_07_syntax.types import PdfDict

from ._fonts import (
    pdfminer_embedded_cmap_is_unusable,
    pdfminer_normalized_width,
)


class PdfminerContentLexer(PdfLexer):
    def read_string(
        self,
        *,
        drop_unknown_escapes: bool = True,
        unknown_escape: Callable[[int], bytes] | None = None,
        eol_pair: Callable[[int, int], bool] | None = None,
    ) -> bytes:
        return super().read_string(
            drop_unknown_escapes=drop_unknown_escapes,
            unknown_escape=unknown_escape,
            eol_pair=eol_pair,
        )

    def parse_dictionary_or_stream(self) -> PdfDict:
        return self.parse_dictionary()

    def handle_dictionary_key_error(self) -> bool:
        raise PdfParseError("invalid content dictionary")

    def handle_dictionary_entry_error(self, value_start: int) -> bool:
        raise PdfParseError("invalid content dictionary")


class PdfminerRecovery(CaptureRecovery):
    def resume(
        self,
        lexer: SyntaxLexer,
        error: PdfParseError,
        kind: str,
        start: int,
        is_operator: Callable[[bytes], bool] | None = None,
    ) -> int | None:
        if str(error) == "invalid content dictionary":
            raise PdfError(str(error)) from error
        return super().resume(lexer, error, kind, start, is_operator)


class PdfminerTextState(TextState):
    def __init__(self, document: Any, *, page_clip: Rectangle | None = None) -> None:
        super().__init__(
            document, page_clip=page_clip, options=CaptureOptions(ink_bounds=False, text_runs=False)
        )
        self.cursor = 0.0
        self.frame_cursors: dict[int, float] = {}

    def enter_stream(self, state: object, frame: ContentStreamFrame) -> None:
        self.frame_cursors[id(frame)] = self.cursor
        if frame.is_form:
            self.cursor = 0.0
        super().enter_stream(state, frame)

    def exit_stream(self, state: object, frame: ContentStreamFrame) -> None:
        self.cursor = self.frame_cursors.pop(id(frame))
        super().exit_stream(state, frame)

    def current_capture_actual_text_span(self) -> None:
        return None

    def text_boundary(self, state: object, kind: str) -> None:
        if kind in {"begin", "move", "matrix"}:
            self.cursor = 0.0
        super().text_boundary(state, kind)

    def op_Tj(self, operands: ContentOperands, depth: int) -> None:
        if operands:
            self.append_tj_array([operands[-1]])

    def append_tj_array(self, array: Any) -> None:
        decoder = self.get_decoder()
        if decoder.is_vertical:
            super().append_tj_array(array)
            return
        if not isinstance(array, (list, tuple)):
            return
        scale = self.graphics.horizontal_scale * 0.01
        adjustment_scale = 0.001 * self.graphics.font_size * scale
        char_space = self.graphics.char_space * scale
        word_space = 0.0 if decoder.is_cid_font else self.graphics.word_space * scale  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        needs_spacing = False
        paint: GlyphPaint | None = None
        for value in array:
            if isinstance(value, (int, float)):
                self.cursor -= value * adjustment_scale
                needs_spacing = True
                continue
            if not isinstance(value, PdfString):
                continue
            for decoded in decoder.decode_glyphs(value.data):
                if needs_spacing:
                    self.cursor += char_space
                self.text_matrix = self.text_matrix._replace(
                    e=self.cursor * self.line_matrix.a + self.line_matrix.e,
                    f=self.cursor * self.line_matrix.b + self.line_matrix.f,
                )
                start = len(self.glyphs)
                advance_x, advance_y = decoder.glyph_advance_vector(
                    decoded.width_code,
                    font_size=self.graphics.font_size,
                    char_space=self.graphics.char_space,
                    word_space=self.graphics.word_space,
                    horizontal_scale=self.graphics.horizontal_scale,
                    encoded_space=decoded.code_bytes == b" ",
                )
                if paint is None:
                    paint = self.glyph_paint(self.capture_color(stroke=False))
                self.show_text(
                    self,
                    decoded.unicode,
                    decoded.code_bytes,
                    (decoded,),
                    decoder,
                    advance_x,
                    advance_y,
                    glyph_paint=paint,
                )
                if len(self.glyphs) > start:
                    glyph = self.glyphs[start]
                    width = pdfminer_normalized_width(glyph)
                    if pdfminer_embedded_cmap_is_unusable(glyph):
                        width = 0.0
                    self.cursor += width * self.graphics.font_size * scale
                if decoded.width_code == 32:
                    self.cursor += word_space
                needs_spacing = True
        self.text_matrix = self.text_matrix._replace(
            e=self.cursor * self.line_matrix.a + self.line_matrix.e,
            f=self.cursor * self.line_matrix.b + self.line_matrix.f,
        )
        self.text_boundary(self, "shown")


def pdfminer_page_program(page: PdfPage) -> CapturedProgram:
    state = PdfminerTextState(page.document, page_clip=page.effective_page_clip())
    state.lexer_factory = PdfminerContentLexer
    state.recovery = PdfminerRecovery()
    page.consume_contents(state)
    state.run_accumulator.flush()
    return CapturedProgram(
        runs=tuple(state.runs),
        glyphs=tuple(state.glyphs),
        drawings=tuple(state.drawings),
        inline_images=tuple(state.inline_images),
        lines=tuple(state.lines),
        text_boundaries=tuple(state.text_boundaries),
        options=state.options,
    )


def pdfminer_validate_page_resources(page: PdfPage) -> None:
    resources = page.resources
    fonts = page.document.resolver.resolve(resources.get("Font"))
    if isinstance(fonts, dict):
        for font_value in fonts.values():
            font = page.document.resolver.resolve(font_value)
            if not isinstance(font, dict):
                continue
            if recover_pdf_name(font.get("Subtype")) == "Type0":
                descendants = page.document.resolver.resolve(font.get("DescendantFonts"))
                if not isinstance(descendants, list) or not descendants:
                    raise PdfError("Type0 font is missing /DescendantFonts")

    color_spaces = page.document.resolver.resolve(resources.get("ColorSpace"))
    if not isinstance(color_spaces, dict):
        return
    for color_space_value in color_spaces.values():
        color_space = page.document.resolver.resolve(color_space_value)
        if not isinstance(color_space, list) or len(color_space) < 2:
            continue
        if recover_pdf_name(color_space[0]) != "ICCBased":
            continue
        profile = page.document.resolver.resolve(color_space[1])
        dictionary = getattr(profile, "dictionary", profile)
        if not isinstance(dictionary, dict) or dictionary.get("N") is None:
            raise PdfError("ICCBased color profile is missing /N")
