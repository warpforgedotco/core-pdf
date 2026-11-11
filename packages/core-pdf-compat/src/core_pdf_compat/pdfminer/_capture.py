from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field  # noqa: TID251
from typing import Any, NamedTuple

import core_pdf_spec.s_07_syntax.lexer
from core_pdf import PdfPage
from core_pdf.impl.capture_glyphs import GlyphPaint
from core_pdf.impl.capture_program import CapturedProgram, CaptureOptions
from core_pdf.impl.capture_recording import TextState
from core_pdf.impl.capture_recovery import CaptureRecovery
from core_pdf.impl.exceptions import PdfError, PdfParseError
from core_pdf.impl.pdf_values import recover_pdf_name
from core_pdf.impl.recovery_lexer import PdfLexer
from core_pdf.impl.types import PdfReference, PdfString, Rectangle
from core_pdf_spec.s_07_content.inline_images import InlineImage
from core_pdf_spec.s_07_content.model import PdfPath
from core_pdf_spec.s_07_content.operations import ContentOperands
from core_pdf_spec.s_07_content.streams import ContentStreamFrame
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import Matrix

from ._fonts import (
    EMPTY_GLYPH_TEXT,
    pdfminer_embedded_cmap_is_unusable,
    pdfminer_normalized_width,
)
from ._parsing import pdfminer_content_error
from .pdfexceptions import PDFKeyError


class InvalidContentDictionaryError(PdfParseError): ...


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
        raise InvalidContentDictionaryError("invalid content dictionary")

    def handle_dictionary_entry_error(self, value_start: int) -> bool:
        raise InvalidContentDictionaryError("invalid content dictionary")


class PdfminerRecovery(CaptureRecovery):
    def resume(
        self,
        lexer: core_pdf_spec.s_07_syntax.lexer.PdfLexer,
        error: PdfParseError,
        kind: str,
        start: int,
        is_operator: Callable[[bytes], bool] | None = None,
    ) -> int | None:
        if isinstance(error, InvalidContentDictionaryError):
            # pdfminer's PSStackParser raises PSSyntaxError naming the dictionary's entries.
            pdfminer_error = pdfminer_content_error(lexer.raw_data)
            if pdfminer_error is not None:
                raise pdfminer_error from error
            raise PdfError(str(error)) from error
        return super().resume(lexer, error, kind, start, is_operator)


class PaintedPath(NamedTuple):
    ops: bytes
    coords: tuple[float, ...]
    ctm: Any
    kind: str
    fill_rule: str
    line_width: float
    stroke_color: Any
    fill_color: Any
    dash: Any
    mcid: int | None
    tag: str | None
    sequence: int


# One step of the walk pdfminer's layout analyzer sees, in content order:
# ("glyphs", first, end) for the glyphs self.glyphs[first:end],
# ("path", index) for painted_paths[index],
# ("begin", name, bbox) and ("end",) around a form XObject's content, and
# ("image", name, ctm, stream) for an image XObject or inline image.
LayoutEvent = tuple[Any, ...]


PdfminerMatrix = tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class PdfminerProgram(CapturedProgram):
    layout_events: tuple[LayoutEvent, ...] = field(default=(), kw_only=True)
    painted_paths: tuple[PaintedPath, ...] = field(default=(), kw_only=True)
    # A horizontal glyph's matrix as pdfminer computes it, in its layout space, by index.
    glyph_matrices: dict[int, PdfminerMatrix] = field(default_factory=dict, kw_only=True)


def mult_matrix(m1: Any, m0: Any) -> PdfminerMatrix:
    """pdfminer.six's utils.mult_matrix, Copyright (c) 2004-2016 Yusuke Shinyama, MIT."""
    (a1, b1, c1, d1, e1, f1) = m1
    (a0, b0, c0, d0, e0, f0) = m0
    return (
        a0 * a1 + c0 * b1,
        b0 * a1 + d0 * b1,
        a0 * c1 + c0 * d1,
        b0 * c1 + d0 * d1,
        a0 * e1 + c0 * f1 + e0,
        b0 * e1 + d0 * f1 + f0,
    )


US_LETTER = (0.0, 0.0, 612.0, 792.0)


def pdfminer_media_box(page: PdfPage) -> tuple[float, float, float, float]:
    """The page's MediaBox; pdfminer's PDFPage takes US Letter for a missing or unreadable one."""
    return page.media_box or US_LETTER


def pdfminer_base_ctm(page: PdfPage) -> PdfminerMatrix:
    """The matrix pdfminer's interpreter starts a page with, from its PDFPageInterpreter."""
    x0, y0, x1, y1 = pdfminer_media_box(page)
    rotation = int(page.rotation) % 360
    if rotation == 90:
        return (0, -1, 1, 0, -y0, x1)
    if rotation == 180:
        return (-1, 0, 0, -1, x1, y1)
    if rotation == 270:
        return (0, 1, -1, 0, y1, -x0)
    return (1, 0, 0, 1, -x0, -y0)


class PdfminerGlyphProbe(NamedTuple):
    """The fields of a glyph observation pdfminer's width depends on."""

    font_decoder: Any
    char_code: int | None
    cid: int | None
    code_bytes: bytes
    unicode_source: str
    text: str
    alternates: tuple[str, ...]


class PdfminerTextState(TextState):
    def __init__(self, document: Any, *, page_clip: Rectangle | None = None) -> None:
        super().__init__(
            document,
            page_clip=page_clip,
            options=CaptureOptions(ink_bounds=False, text_runs=False, render_details=False),
        )
        self.cursor = 0.0
        # pdfminer's text line matrix holds a vertical cursor beside the horizontal one:
        # vertical writing moves it, and horizontal writing leaves it.
        self.vertical_cursor = 0.0
        self.strict_cmaps = True
        self.frame_cursors: dict[int, tuple[float, float]] = {}
        # A glyph's pdfminer advance depends only on its decoder and these fields.
        self.glyph_widths: dict[tuple[Any, ...], float] = {}
        self.decoder_glyph_widths: dict[int, tuple[Any, dict[int, tuple[Any, float]]]] = {}
        # Flattening loses the operators pdfminer classifies paths by.
        self.painted_paths: list[PaintedPath] = []
        self.vertical_line: Any = None
        self.vertical_offset = (0.0, 0.0)
        self.current_mcid: int | None = None
        self.current_tag: str | None = None
        self.layout_events: list[LayoutEvent] = []
        self.pending_xobject: str | None = None
        # Above zero inside content pdfminer does not walk: Type3 glyph procedures, and forms
        # without a /BBox.
        self.hidden_depth = 0
        self.hidden_frames: set[int] = set()
        # pdfminer multiplies its matrices from a page matrix of its own, so its rounding
        # differs from the engine's; each engine ctm maps to the matrix pdfminer holds.
        self.base_ctm: PdfminerMatrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        self.pdfminer_ctms: dict[int, tuple[Any, PdfminerMatrix]] = {}
        self.glyph_matrices: dict[int, PdfminerMatrix] = {}

    def op_BDC(self, operands: ContentOperands, depth: int) -> None:
        super().op_BDC(operands, depth)
        self.current_tag = self.resolver.resolve_name(operands[0]) if operands else None
        self.current_mcid = self.marked_content_stack[-1].mcid

    def op_BMC(self, operands: ContentOperands, depth: int) -> None:
        super().op_BMC(operands, depth)
        self.current_tag = self.resolver.resolve_name(operands[0]) if operands else None
        self.current_mcid = None

    def op_EMC(self, operands: ContentOperands, depth: int) -> None:
        super().op_EMC(operands, depth)
        self.current_tag = None
        self.current_mcid = None

    def show_text(
        self,
        state: object,
        text: str,
        data: bytes | memoryview,
        glyphs: tuple[Any, ...],
        decoder: Any,
        adv_x: float,
        adv_y: float,
        *,
        glyph_paint: GlyphPaint | None = None,
    ) -> None:
        # Named parameters rather than *args: this runs once per glyph.
        captured = self.glyphs
        start = len(captured)
        super().show_text(state, text, data, glyphs, decoder, adv_x, adv_y, glyph_paint=glyph_paint)
        if len(captured) > start and not self.hidden_depth:
            self.layout_events.append(("glyphs", start, len(captured)))

    def op_Do(self, operands: ContentOperands, depth: int) -> ContentStreamFrame | None:
        name = self.resolver.resolve_name(operands[0]) if operands else None
        self.pending_xobject = name
        frame = super().op_Do(operands, depth)
        if frame is None and name and not self.hidden_depth:
            # pdfminer opens a figure for a form it then refuses to run, as when it recurses.
            xobj = self.resolver.resolve(self.lookup_page_resource("XObject", name))
            dictionary = getattr(xobj, "dictionary", None)
            if (
                isinstance(dictionary, dict)
                and recover_pdf_name(dictionary.get("Subtype")) == "Form"
                and dictionary.get("BBox") is not None
            ):
                ctm = self.matrix_operand(dictionary.get("Matrix"), "form").multiply(
                    self.graphics.ctm
                )
                self.layout_events.append(
                    ("begin", name, self.figure_bbox(dictionary.get("BBox"), None, ctm))
                )
                self.layout_events.append(("end",))
        return frame

    def paint_image(self, state: object, xobj: PdfStream) -> None:
        # pdfminer places every image XObject that has a size, whether or not it is visible.
        dictionary = xobj.dictionary
        if not self.hidden_depth and "Width" in dictionary and "Height" in dictionary:
            self.layout_events.append(
                ("image", self.pending_xobject or "", self.graphics.ctm, xobj)
            )
        super().paint_image(state, xobj)

    def paint_inline_image(self, state: object, image: InlineImage) -> None:
        dictionary = image.dictionary
        if not self.hidden_depth and "Width" in dictionary and "Height" in dictionary:
            self.layout_events.append(("image", str(id(image)), self.graphics.ctm, image))
        super().paint_inline_image(state, image)

    def paint_path(self, state: object, source: PdfPath, kind: str, fill_rule: str) -> None:
        if source.ops:
            graphics = self.graphics
            if not self.hidden_depth:
                self.layout_events.append(("path", len(self.painted_paths)))
            self.painted_paths.append(
                PaintedPath(
                    bytes(source.ops),
                    tuple(source.coords),
                    graphics.ctm,
                    kind,
                    fill_rule,
                    graphics.line_width,
                    self.capture_color(stroke=True),
                    self.capture_color(stroke=False),
                    self.transformed_dash_pattern(),
                    self.current_mcid,
                    self.current_tag,
                    self.sequence,
                )
            )
        super().paint_path(state, source, kind, fill_rule)

    def pdfminer_ctm(self, ctm: Any) -> PdfminerMatrix:
        known = self.pdfminer_ctms.get(id(ctm))
        if known is not None and known[0] is ctm:
            return known[1]
        return mult_matrix(ctm, self.base_ctm)

    def op_cm(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 6)) is None:
            return
        before = self.graphics.ctm
        after = self.graphics.ctm = Matrix(*values).multiply(before)
        if after is not before:
            self.pdfminer_ctms[id(after)] = (after, mult_matrix(values, self.pdfminer_ctm(before)))

    def enter_stream(self, state: object, frame: ContentStreamFrame) -> None:
        self.frame_cursors[id(frame)] = (self.cursor, self.vertical_cursor)
        if not self.hidden_depth:
            if frame.is_form:
                raw_matrix = self.resolver.deep_resolve(frame.stream.dictionary.get("Matrix"))
                form_matrix: Any = (1, 0, 0, 1, 0, 0)
                if isinstance(raw_matrix, (list, tuple)) and len(raw_matrix) == 6:
                    form_matrix = raw_matrix
                try:
                    pdfminer_ctm = mult_matrix(form_matrix, self.pdfminer_ctm(self.graphics.ctm))
                except TypeError:
                    pdfminer_ctm = mult_matrix(frame.ctm, self.base_ctm)
            else:
                known = self.pdfminer_ctms.get(id(frame.ctm))
                if known is not None and known[0] is frame.ctm:
                    pdfminer_ctm = known[1]
                elif frame.depth == 0:
                    pdfminer_ctm = self.base_ctm
                else:
                    pdfminer_ctm = self.pdfminer_ctm(frame.ctm)
            self.pdfminer_ctms[id(frame.ctm)] = (frame.ctm, pdfminer_ctm)
        if frame.is_form:
            self.cursor = 0.0
            if self.hidden_depth or frame.form_bbox_operand is None:
                self.hidden_depth += 1
                self.hidden_frames.add(id(frame))
            else:
                self.layout_events.append(
                    (
                        "begin",
                        self.pending_xobject or "",
                        self.figure_bbox(frame.form_bbox_operand, frame.form_bbox, frame.ctm),
                    )
                )
        self.pending_xobject = None
        super().enter_stream(state, frame)

    def exit_stream(self, state: object, frame: ContentStreamFrame) -> None:
        self.cursor, self.vertical_cursor = self.frame_cursors.pop(id(frame))
        if id(frame) in self.hidden_frames:
            self.hidden_frames.discard(id(frame))
            self.hidden_depth -= 1
        elif frame.is_form:
            self.layout_events.append(("end",))
        super().exit_stream(state, frame)

    def figure_bbox(
        self, raw: object, form_bbox: Rectangle | None, matrix: Any
    ) -> tuple[float, float, float, float]:
        """The form's box as pdfminer's LTFigure reads it: x, y, width and height."""
        values: list[float] = []
        if isinstance(raw, (list, tuple)) and len(raw) >= 4:
            resolved = [self.resolver.resolve_float(value, default=None) for value in raw[:4]]
            values = [value for value in resolved if value is not None]
        if len(values) == 4:
            x, y, width, height = values
        elif form_bbox is not None:
            x, y, x1, y1 = form_bbox
            width, height = x1 - x, y1 - y
        else:
            x = y = width = height = 0.0
        corners = [
            (matrix.a * px + matrix.c * py + matrix.e, matrix.b * px + matrix.d * py + matrix.f)
            for px, py in ((x, y), (x + width, y), (x, y + height), (x + width, y + height))
        ]
        return (
            min(point[0] for point in corners),
            min(point[1] for point in corners),
            max(point[0] for point in corners),
            max(point[1] for point in corners),
        )

    def current_capture_actual_text_span(self) -> None:
        return None

    def text_boundary(self, state: object, kind: str) -> None:
        if kind in {"begin", "move", "matrix"}:
            self.cursor = 0.0
            self.vertical_cursor = 0.0
        elif kind == "type3-glyph-begin":
            self.hidden_depth += 1
        elif kind == "type3-glyph-end":
            self.hidden_depth -= 1
        super().text_boundary(state, kind)

    def op_Tj(self, operands: ContentOperands, depth: int) -> None:
        if operands:
            self.append_tj_array([operands[-1]])

    def op_double_quote(self, operands: ContentOperands, depth: int) -> None:
        # pdfminer's " sets the spacing and shows the text, but stays on the same line.
        if len(operands) < 3 or (values := self.as_floats(operands, 2)) is None:
            return
        self.graphics.word_space, self.graphics.char_space = values
        self.op_Tj((operands[2],), depth)

    def record_vertical_matrices(self, array: Any, decoder: Any, start: int) -> None:
        """Each vertical glyph's matrix as pdfminer's render_string_vertical computes it.

        pdfminer translates the line matrix times the ctm by its line position (x, y), and
        moves y by each glyph's vertical advance; its rounding decides which boxes touch,
        and so which lines group.
        """
        if self.hidden_depth or not isinstance(array, (list, tuple)):
            return
        scale = self.graphics.horizontal_scale * 0.01
        font_size = self.graphics.font_size
        char_space = self.graphics.char_space * scale
        adjustment_scale = 0.001 * font_size * scale
        a, b, c, d, e, f = mult_matrix(self.line_matrix, self.pdfminer_ctm(self.graphics.ctm))
        x = self.cursor
        y = self.vertical_cursor
        matrices: list[PdfminerMatrix] = []
        needs_spacing = False
        for value in array:
            if isinstance(value, (int, float)):
                y -= value * adjustment_scale
                needs_spacing = True
                continue
            if not isinstance(value, PdfString):
                continue
            for decoded in decoder.decode_glyphs(value.data):
                if needs_spacing:
                    y += char_space
                matrices.append((a, b, c, d, x * a + y * c + e, x * b + y * d + f))
                if decoded.cid is None:
                    return
                width = decoder.vertical_glyph_metric(decoded.cid)[0] * 0.001
                y += width * font_size * scale
                needs_spacing = True
        self.vertical_cursor = y
        if len(self.glyphs) - start == len(matrices):
            for index, matrix in enumerate(matrices, start):
                self.glyph_matrices[index] = matrix

    def append_tj_array(self, array: Any) -> None:
        decoder = self.get_decoder()
        if self.vertical_line is not self.line_matrix:
            self.vertical_line = self.line_matrix
            self.vertical_offset = (0.0, 0.0)
        if decoder.is_vertical:
            before = self.text_matrix
            start = len(self.glyphs)
            super().append_tj_array(array)
            self.record_vertical_matrices(array, decoder, start)
            after = self.text_matrix
            # Later horizontal text continues from where this vertical run ended.
            self.vertical_offset = (
                self.vertical_offset[0] + after.e - before.e,
                self.vertical_offset[1] + after.f - before.f,
            )
            return
        if not isinstance(array, (list, tuple)):
            return
        self.append_glyphs(array, decoder)
        self.text_matrix = self.text_matrix._replace(
            e=self.cursor * self.line_matrix.a + self.line_matrix.e + self.vertical_offset[0],
            f=self.cursor * self.line_matrix.b + self.line_matrix.f + self.vertical_offset[1],
        )
        self.text_boundary(self, "shown")

    def append_glyphs(self, array: list[Any] | tuple[Any, ...], decoder: Any) -> None:
        """Show a horizontal TJ array's glyphs where pdfminer's render_string places them.

        Each glyph sits at the cursor on the line matrix and moves it by its pdfminer width,
        which depends only on its decoder and decoded fields. Glyphs go to show_glyph_run
        together. A glyph whose text splits is shown alone between runs, its width taken
        from what it captured, as pdfminer measures the first part; a run show_glyph_run
        declines is shown a glyph at a time, at the same places.
        """
        graphics = self.graphics
        scale = graphics.horizontal_scale * 0.01
        font_size = graphics.font_size
        adjustment_scale = 0.001 * font_size * scale
        char_space = graphics.char_space * scale
        word_space = 0.0 if decoder.is_cid_font else graphics.word_space * scale
        line = self.line_matrix
        la, lb, le, lf = line.a, line.b, line.e, line.f
        offset_x, offset_y = self.vertical_offset
        # pdfminer's char matrix: the line matrix times the ctm, translated by the cursor.
        line_ctm = (
            mult_matrix(line, self.pdfminer_ctm(graphics.ctm))
            if self.vertical_offset == (0.0, 0.0) and not self.hidden_depth
            else None
        )
        widths = self.glyph_widths
        known_decoder_widths = self.decoder_glyph_widths.get(id(decoder))
        if known_decoder_widths is None or known_decoder_widths[0] is not decoder:
            known_decoder_widths = self.decoder_glyph_widths[id(decoder)] = (decoder, {})
        decoder_widths = known_decoder_widths[1]
        decoder_widths_get = decoder_widths.get
        cursor = self.cursor
        needs_spacing = False
        run: list[Any] = []
        texts: list[str] = []
        origins: list[float] = []
        cursors: list[float] = []
        run_append = run.append
        texts_append = texts.append
        origins_append = origins.append
        cursors_append = cursors.append
        for value in array:
            if isinstance(value, (int, float)):
                cursor -= value * adjustment_scale
                needs_spacing = True
                continue
            if not isinstance(value, PdfString):
                continue
            for decoded in decoder.decode_glyphs(value.data):
                text = decoded.unicode or EMPTY_GLYPH_TEXT
                if needs_spacing:
                    cursor += char_space
                e = cursor * la + le + offset_x
                f = cursor * lb + lf + offset_y
                if decoded.split_unicode and len(text) > 1:
                    self.show_run(run, texts, origins, cursors, decoder, line_ctm)
                    run, texts, origins, cursors = [], [], [], []
                    run_append = run.append
                    texts_append = texts.append
                    origins_append = origins.append
                    cursors_append = cursors.append
                    start = self.show_alone(decoded, text, e, f, cursor, decoder, line_ctm)
                    if start is not None:
                        glyph = self.glyphs[start]
                        key = (
                            glyph.font_decoder,
                            glyph.char_code,
                            glyph.cid,
                            glyph.code_bytes,
                            glyph.unicode_source,
                            glyph.text,
                            glyph.alternates,
                        )
                        width = widths.get(key)
                        if width is None:
                            width = self.pdfminer_width(key)
                        cursor += width * font_size * scale
                else:
                    run_append(decoded)
                    texts_append(text)
                    origins_append(e)
                    origins_append(f)
                    cursors_append(cursor)
                    decoded_id = id(decoded)
                    known_width = decoder_widths_get(decoded_id)
                    if known_width is not None and known_width[0] is decoded:
                        width = known_width[1]
                    else:
                        key = (
                            decoder,
                            decoded.char_code,
                            decoded.cid,
                            decoded.code_bytes,
                            decoded.unicode_source,
                            text,
                            decoded.alternates,
                        )
                        width = widths.get(key)
                        if width is None:
                            width = self.pdfminer_width(key)
                        decoder_widths[decoded_id] = (decoded, width)
                    cursor += width * font_size * scale
                if decoded.width_code == 32:
                    cursor += word_space
                needs_spacing = True
        self.show_run(run, texts, origins, cursors, decoder, line_ctm)
        self.cursor = cursor

    def show_run(
        self,
        run: list[Any],
        texts: list[str],
        origins: list[float],
        cursors: list[float],
        decoder: Any,
        line_ctm: PdfminerMatrix | None,
    ) -> None:
        """Show placed glyphs in one show_glyph_run call, or one at a time if it declines."""
        if not run:
            return
        captured = self.glyphs
        start = len(captured)
        if (
            self.show_glyph_run(
                tuple(run), texts, origins, decoder, glyph_paint=self.shared_paint()
            )
            is None
        ):
            for index, decoded in enumerate(run):
                self.show_alone(
                    decoded,
                    texts[index],
                    origins[2 * index],
                    origins[2 * index + 1],
                    cursors[index],
                    decoder,
                    line_ctm,
                )
            return
        if not self.hidden_depth:
            # One event for the run, which the layout walk reads glyph by glyph.
            self.layout_events.append(("glyphs", start, len(captured)))
        if line_ctm is not None:
            a, b, c, d, e, f = line_ctm
            matrices = self.glyph_matrices
            for index, x in enumerate(cursors, start):
                matrices[index] = (a, b, c, d, x * a + 0 * c + e, x * b + 0 * d + f)

    def show_alone(
        self,
        decoded: Any,
        text: str,
        e: float,
        f: float,
        x: float,
        decoder: Any,
        line_ctm: PdfminerMatrix | None,
    ) -> int | None:
        """Show one placed glyph with show_text; the index of its first capture, if any."""
        self.text_matrix = self.text_matrix._replace(e=e, f=f)
        start = len(self.glyphs)
        # This capture records no text runs, and show_text reads the glyph advance only for
        # runs, so it is not computed.
        self.show_text(
            self,
            text,
            decoded.code_bytes,
            (decoded,),
            decoder,
            0.0,
            0.0,
            glyph_paint=self.shared_paint(),
        )
        if len(self.glyphs) == start:
            return None
        if line_ctm is not None:
            a, b, c, d, le, lf = line_ctm
            self.glyph_matrices[start] = (a, b, c, d, x * a + 0 * c + le, x * b + 0 * d + lf)
        return start

    def pdfminer_width(self, key: tuple[Any, ...]) -> float:
        """The advance pdfminer gives a glyph, in text space units per font size unit,
        cached under key: the PdfminerGlyphProbe fields it depends on."""
        glyph = PdfminerGlyphProbe._make(key)
        width = pdfminer_normalized_width(glyph)
        if self.strict_cmaps and pdfminer_embedded_cmap_is_unusable(glyph):
            width = 0.0
        self.glyph_widths[key] = width
        return width


def pdfminer_page_program(page: PdfPage, *, strict_cmaps: bool = True) -> PdfminerProgram:
    """strict_cmaps: whether glyphs of a font whose embedded CMap pdfminer cannot find
    advance nothing, as in pdfminer; unstructured parses those CMaps itself."""
    state = PdfminerTextState(page.document, page_clip=page.effective_page_clip())
    state.strict_cmaps = strict_cmaps
    state.lexer_factory = PdfminerContentLexer
    state.content_recovery = PdfminerRecovery()
    state.base_ctm = pdfminer_base_ctm(page)
    page.consume_contents(state)
    state.run_accumulator.flush()
    return PdfminerProgram(
        runs=tuple(state.runs),
        glyphs=tuple(state.glyphs),
        drawings=tuple(state.drawings),
        inline_images=tuple(state.inline_images),
        lines=state.lines.since(0),
        text_boundaries=tuple(state.text_boundaries),
        options=state.options,
        layout_events=tuple(state.layout_events),
        painted_paths=tuple(state.painted_paths),
        glyph_matrices=state.glyph_matrices,
    )


def pdfminer_validate_page_resources(page: PdfPage) -> None:
    """Raise what pdfminer's PDFPageInterpreter.init_resources raises on the page's resources.

    It reads the resources in their order: each font of /Font (a Type0 font takes its
    /DescendantFonts with spec["DescendantFonts"] and asserts the list is not empty), and
    each color space of /ColorSpace (an ICCBased one reads /N from its stream).
    """
    resources = page.resources
    resolver = page.document.resolver
    for key, value in resources.items():
        name = recover_pdf_name(key)
        if name == "Font":
            validate_fonts(resolver.as_dict(value) or {}, resolver)
        elif name == "ColorSpace":
            validate_color_spaces(resolver.as_dict(value) or {}, resolver)


def validate_fonts(fonts: PdfDict, resolver: Any) -> None:
    for font_value in fonts.values():
        font = resolver.as_dict(font_value)
        if font is None or recover_pdf_name(font.get("Subtype")) != "Type0":
            continue
        if font.get("DescendantFonts") is None:
            raise KeyError("DescendantFonts")
        if not resolver.array_at(font, "DescendantFonts"):
            raise AssertionError


def validate_color_spaces(color_spaces: PdfDict, resolver: Any) -> None:
    for color_space_value in color_spaces.values():
        color_space = resolver.resolve(color_space_value)
        if not isinstance(color_space, list) or len(color_space) < 2:
            continue
        if recover_pdf_name(color_space[0]) != "ICCBased":
            continue
        # pdfminer's stream_value(spec[1])["N"]: a missing /N names the stream's object,
        # and anything but a stream stands in for an empty stream with no object number.
        profile = resolver.resolve(color_space[1])
        dictionary = getattr(profile, "dictionary", None)
        if not isinstance(dictionary, dict):
            raise PDFKeyError("PDF stream object None does not have attribute 'N'")
        if dictionary.get("N") is None:
            reference = color_space[1]
            objid = reference.object_number if isinstance(reference, PdfReference) else None
            raise PDFKeyError(f"PDF stream object {objid} does not have attribute 'N'")
