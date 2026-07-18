# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from functools import lru_cache
from math import hypot
from typing import TYPE_CHECKING, Any, cast

from core_pdf.impl.engine.spec.s_07_content.capture import CapturedDrawing
from core_pdf.impl.engine.spec.s_07_content.inline_images import (
    decode_inline_image_data,
)
from core_pdf.impl.engine.spec.s_07_content.marked_content import MarkedContentEntry
from core_pdf.impl.engine.spec.s_07_objects.coercion import (
    normalize_pdf_name,
    parse_float,
    parse_int,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf.impl.objects import MISSING, PdfStream, PdfString
from core_pdf.impl.types import PdfDict

if TYPE_CHECKING:
    from core_layout.impl.layout.models import TextRun

    from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder


MATRIX_TOLERANCE = 0.1


@lru_cache(maxsize=128)
def detect_rotation_from_linear(
    A: float, B: float, C: float, D: float, tolerance: float = MATRIX_TOLERANCE
) -> int:
    scale_x = hypot(A, B)
    scale_y = hypot(C, D)
    if scale_x <= 0 or scale_y <= 0:
        return 0
    na, nb, nc, nd = A / scale_x, B / scale_x, C / scale_y, D / scale_y
    if (
        abs(na - 1.0) < tolerance
        and abs(nb) < tolerance
        and abs(nc) < tolerance
        and abs(nd - 1.0) < tolerance
    ):
        return 0
    if (
        abs(na) < tolerance
        and abs(nb - 1.0) < tolerance
        and abs(nc + 1.0) < tolerance
        and abs(nd) < tolerance
    ):
        return 90
    if (
        abs(na + 1.0) < tolerance
        and abs(nb) < tolerance
        and abs(nc) < tolerance
        and abs(nd + 1.0) < tolerance
    ):
        return 180
    if (
        abs(na) < tolerance
        and abs(nb + 1.0) < tolerance
        and abs(nc - 1.0) < tolerance
        and abs(nd) < tolerance
    ):
        return 270
    return 0


class OperatorMixin:
    """Operator methods mixed into the concrete TextState interpreter.

    These methods intentionally use the full mutable TextState surface. A
    protocol for every slot would be larger than the behavior it documents.
    """

    __slots__ = ()

    current_point: tuple[float, float] | None
    subpath_start: tuple[float, float] | None
    fill_pattern: PdfDict | None
    stroke_pattern: PdfDict | None
    render_intent: str | None
    current_decoder: FontDecoder | None
    pending_run: TextRun | None

    TEXT_OP = {
        "BT": "op_BT",
        "ET": "op_ET",
        "T*": "op_T_star",
        "Td": "op_Td",
        "TD": "op_TD",
        "Tj": "op_Tj",
        "TJ": "op_TJ",
        "Tm": "op_Tm",
        "Tf": "op_Tf",
        "TL": "op_TL",
        "Tc": "op_Tc",
        "Tw": "op_Tw",
        "Tz": "op_Tz",
        "Tr": "op_Tr",
        "Ts": "op_Ts",
        "'": "op_quote",
        '"': "op_double_quote",
        "Do": "op_Do",
        "BI": "op_BI",
        "BDC": "op_BDC",
        "BMC": "op_BMC",
        "EMC": "op_EMC",
        "q": "op_q",
        "Q": "op_Q",
        "cm": "op_cm",
        "g": "op_g",
        "rg": "op_rg",
        "k": "op_k",
        "G": "op_G",
        "RG": "op_RG",
        "K": "op_K",
        "CS": "op_CS",
        "cs": "op_cs",
        "SC": "op_SC",
        "SCN": "op_SCN",
        "sc": "op_sc",
        "scn": "op_scN",
        "sh": "op_sh",
        "i": "op_i",
        "ri": "op_ri",
        "MP": "op_MP",
        "DP": "op_DP",
        "BX": "op_BX",
        "EX": "op_EX",
        "d0": "op_d0",
        "d1": "op_d1",
        "w": "op_w",
        "J": "op_J",
        "j": "op_j",
        "M": "op_M",
        "d": "op_d",
        "gs": "op_gs",
        "m": "op_m",
        "l": "op_l",
        "re": "op_re",
        "h": "op_h",
        "c": "op_c",
        "v": "op_v",
        "y": "op_y",
        "W": "op_W",
        "W*": "op_W_star",
        "S": "op_paint_stroke",
        "s": "op_paint_stroke",
        "f": "op_paint_fill",
        "F": "op_paint_fill",
        "f*": "op_paint_fill",
        "B": "op_paint_fillstroke",
        "b": "op_paint_fillstroke",
        "B*": "op_paint_fillstroke",
        "b*": "op_paint_fillstroke",
        "n": "op_paint_clear",
    }
    TEXT_ONLY_NOOP_OPS = frozenset(
        {
            "m",
            "l",
            "h",
            "v",
            "y",
            "c",
            "re",
            "W",
            "W*",
            "S",
            "s",
            "f",
            "F",
            "f*",
            "B",
            "b",
            "B*",
            "b*",
            "n",
            "w",
            "J",
            "j",
            "M",
            "d",
            "i",
            "BX",
            "EX",
            "MP",
            "DP",
        }
    )
    TEXT_ONLY_OP = TEXT_OP.copy()
    op = ""
    for op in TEXT_ONLY_NOOP_OPS:
        TEXT_ONLY_OP[op] = "op_noop"
    del op

    def op_noop(self: Any, o, d):
        return

    def op_BT(self: Any, o, d):
        if self.pending_run:
            self.flush_run()
        self.tm_a = self.lm_a = 1.0
        self.tm_b = self.lm_b = 0.0
        self.tm_c = self.lm_c = 0.0
        self.tm_d = self.lm_d = 1.0
        self.tm_e = self.lm_e = 0.0
        self.tm_f = self.lm_f = 0.0
        ca, cb, cc, cd = self.ca, self.cb, self.cc, self.cd
        self.combined_A = ca
        self.combined_B = cb
        self.combined_C = cc
        self.combined_D = cd
        if ca == 1.0 and cb == 0.0 and cc == 0.0 and cd == 1.0:
            self.cached_rotation = 0
        else:
            self.cached_rotation = detect_rotation_from_linear(ca, cb, cc, cd)
        self.invisible_text_layer = False

    def op_ET(self: Any, o, d):
        pending = self.pending_run
        if pending:
            if self.capture_runs:
                self.runs.append(pending)
            self.pending_run = None

    def op_T_star(self: Any, o, d):
        if self.pending_run:
            self.runs.append(self.pending_run)
            self.pending_run = None
        ty = -self.leading
        self.tm_e = self.lm_e + ty * self.lm_c
        self.tm_f = self.lm_f + ty * self.lm_d
        self.lm_e = self.tm_e
        self.lm_f = self.tm_f

    def op_Td(self: Any, o, d):
        if len(o) < 2:
            return
        try:
            tx = self.as_float(o[0])
            ty = self.as_float(o[1])
        except (TypeError, ValueError):
            return
        if self.pending_run:
            self.runs.append(self.pending_run)
            self.pending_run = None
        self.tm_e = self.lm_e + tx * self.lm_a + ty * self.lm_c
        self.tm_f = self.lm_f + tx * self.lm_b + ty * self.lm_d
        self.lm_e = self.tm_e
        self.lm_f = self.tm_f

    def op_Td_values(self: Any, tx: float, ty: float) -> None:
        if self.pending_run:
            self.runs.append(self.pending_run)
            self.pending_run = None
        self.tm_e = self.lm_e + tx * self.lm_a + ty * self.lm_c
        self.tm_f = self.lm_f + tx * self.lm_b + ty * self.lm_d
        self.lm_e = self.tm_e
        self.lm_f = self.tm_f

    def op_TD(self: Any, o, d):
        if len(o) < 2:
            return
        try:
            tx = self.as_float(o[0])
            ty = self.as_float(o[1])
        except (TypeError, ValueError):
            return
        self.leading = -ty
        if self.pending_run:
            self.runs.append(self.pending_run)
            self.pending_run = None
        self.tm_e = self.lm_e + tx * self.lm_a + ty * self.lm_c
        self.tm_f = self.lm_f + tx * self.lm_b + ty * self.lm_d
        self.lm_e = self.tm_e
        self.lm_f = self.tm_f

    def op_TD_values(self: Any, tx: float, ty: float) -> None:
        self.leading = -ty
        if self.pending_run:
            self.runs.append(self.pending_run)
            self.pending_run = None
        self.tm_e = self.lm_e + tx * self.lm_a + ty * self.lm_c
        self.tm_f = self.lm_f + tx * self.lm_b + ty * self.lm_d
        self.lm_e = self.tm_e
        self.lm_f = self.tm_f

    def op_Tj(self: Any, o, d):
        if not o:
            return
        decoder = self.current_decoder if self.current_decoder is not None else self.get_decoder()
        operand = o[0]
        if type(operand) is PdfString:
            self.append_text(data=operand.data, decoder=decoder)
        else:
            self.append_text(operand, decoder=decoder)

    def op_TJ(self: Any, o, d):
        if not o:
            return
        self.append_tj_array(o[0])

    def op_Tm(self: Any, o, d):
        if len(o) >= 6:
            try:
                a = self.as_float(o[0])
                b = self.as_float(o[1])
                c = self.as_float(o[2])
                d_ = self.as_float(o[3])
                e = self.as_float(o[4])
                f = self.as_float(o[5])
            except (TypeError, ValueError):
                return
            self.flush_run()
            self.tm_a = self.lm_a = a
            self.tm_b = self.lm_b = b
            self.tm_c = self.lm_c = c
            self.tm_d = self.lm_d = d_
            self.tm_e = self.lm_e = e
            self.tm_f = self.lm_f = f
            self.update_combined()

    def op_Tm_values(
        self: Any, a: float, b: float, c: float, d_: float, e: float, f: float
    ) -> None:
        self.flush_run()
        self.tm_a = self.lm_a = a
        self.tm_b = self.lm_b = b
        self.tm_c = self.lm_c = c
        self.tm_d = self.lm_d = d_
        self.tm_e = self.lm_e = e
        self.tm_f = self.lm_f = f
        self.update_combined()

    def op_Tf(self: Any, o, d):
        if len(o) < 2:
            return
        font_operand = o[0]
        font_size_operand = o[1]
        self.op_Tf_values(font_operand, font_size_operand)

    def op_Tf_values(self: Any, font_operand: Any, font_size_operand: Any) -> None:
        decoder_matches_resources = self.current_decoder_resources_id == self.resources_id
        if (
            self.current_decoder is not None
            and decoder_matches_resources
            and font_operand is self.font_operand
        ):
            if font_size_operand is not self.font_size_operand:
                value_type = type(font_size_operand)
                if value_type is float or value_type is int:
                    font_size = font_size_operand
                else:
                    try:
                        font_size = self.as_float(font_size_operand)
                    except (TypeError, ValueError):
                        return
                if self.font_size != font_size:
                    self.font_size = font_size
                    self.update_text_scales()
                    self.update_font_metrics()
                self.font_size_operand = font_size_operand
            return
        cache_key = (self.resources_id, font_operand, font_size_operand)
        cached = self.font_setting_cache.get(cache_key)
        if cached is None:
            font_name = self.resolve_name(font_operand)
            if font_name is None:
                return
            try:
                font_size = self.as_float(font_size_operand)
            except (TypeError, ValueError):
                return
            self.font_setting_cache[cache_key] = (font_name, font_size)
        else:
            font_name, font_size = cached
        if (
            self.current_font == font_name
            and self.current_decoder is not None
            and decoder_matches_resources
        ):
            if self.font_size != font_size:
                self.font_size = font_size
                self.update_text_scales()
                self.update_font_metrics()
            self.font_operand = font_operand
            self.font_size_operand = font_size_operand
            return
        self.current_font = font_name
        self.font_size = font_size
        self.update_text_scales()
        self.font_operand = font_operand
        self.font_size_operand = font_size_operand
        self.current_decoder = None
        self.current_decoder = self.get_decoder(update_metrics=False)
        self.update_font_metrics()

    def op_TL(self: Any, o, d):
        if not o:
            return
        try:
            self.leading = self.as_float(o[0])
        except (TypeError, ValueError):
            return

    def op_Tc(self: Any, o, d):
        if not o:
            return
        try:
            self.char_space = self.as_float(o[0])
        except (TypeError, ValueError):
            return
        self.update_char_space_scale()

    def op_Tc_values(self: Any, char_space: float) -> None:
        self.char_space = char_space
        self.update_char_space_scale()

    def op_Tw(self: Any, o, d):
        if not o:
            return
        try:
            self.word_space = self.as_float(o[0])
        except (TypeError, ValueError):
            return
        self.update_word_space_scale()

    def op_Tw_values(self: Any, word_space: float) -> None:
        if self.word_space == word_space:
            return
        self.word_space = word_space
        self.update_word_space_scale()

    def op_Tr(self: Any, o, d):
        if not o:
            return
        try:
            self.render_mode = self.as_int(o[0])
        except (TypeError, ValueError):
            return

    def op_Tz(self: Any, o, d):
        if not o:
            return
        try:
            self.horizontal_scale = self.as_float(o[0])
        except (TypeError, ValueError):
            return
        self.update_horizontal_text_scale()

    def op_Ts(self: Any, o, d):
        if not o:
            return
        try:
            self.rise = self.as_float(o[0])
        except (TypeError, ValueError):
            return

    def op_quote(self: Any, o, d):
        if len(o) < 1:
            return
        if self.pending_run:
            self.runs.append(self.pending_run)
            self.pending_run = None
        ty = -self.leading
        self.tm_e = self.lm_e + ty * self.lm_c
        self.tm_f = self.lm_f + ty * self.lm_d
        self.lm_e = self.tm_e
        self.lm_f = self.tm_f
        self.pending_line_break = True
        decoder = self.current_decoder if self.current_decoder is not None else self.get_decoder()
        operand = o[0]
        if type(operand) is PdfString:
            self.append_text(data=operand.data, decoder=decoder)
        else:
            self.append_text(operand, decoder=decoder)

    def op_double_quote(self: Any, o, d):
        if len(o) < 3:
            return
        try:
            self.word_space = self.as_float(o[0])
            self.char_space = self.as_float(o[1])
        except (TypeError, ValueError):
            return
        if self.pending_run:
            self.runs.append(self.pending_run)
            self.pending_run = None
        ty = -self.leading
        self.tm_e = self.lm_e + ty * self.lm_c
        self.tm_f = self.lm_f + ty * self.lm_d
        self.lm_e = self.tm_e
        self.lm_f = self.tm_f
        self.pending_line_break = True
        decoder = self.current_decoder if self.current_decoder is not None else self.get_decoder()
        operand = o[2]
        if type(operand) is PdfString:
            self.append_text(data=operand.data, decoder=decoder)
        else:
            self.append_text(operand, decoder=decoder)

    def op_BI(self: Any, o, d):
        if not o:
            return
        image = o[0]
        if not hasattr(image, "dictionary"):
            return
        if self.capture_graphics and self.is_graphics_visible():
            dictionary = dict(image.dictionary)
            data = decode_inline_image_data(
                dictionary,
                getattr(image, "data", b""),
            )
            self.inline_images.append(
                {
                    "seqno": self.sequence,
                    "dictionary": dictionary,
                    "data": data,
                    "ctm": self.ctm,
                    "xobject_depth": self.xobject_depth,
                }
            )
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=None,
                    fill_opacity=None,
                    blend_mode=self.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    kind="inline-image",
                    dash_pattern=self.transformed_dash_pattern(),
                    items=[],
                )
            )

    def op_BDC(self: Any, o, d):
        tag = self.resolve_name(o[0]) if o else None
        layer = None
        if tag == "OC":
            layer = self.resolve_marked_content_layer(o[1]) if len(o) >= 2 else None
        actual_text = (
            self.resolve_marked_content_actual_text(o[1]) if tag == "Span" and len(o) >= 2 else None
        )
        self.marked_content_stack.append(MarkedContentEntry(layer=layer, actual_text=actual_text))

    def op_BMC(self: Any, o, d):
        self.marked_content_stack.append(MarkedContentEntry())

    def op_EMC(self: Any, o, d):
        if self.marked_content_stack:
            entry = self.marked_content_stack.pop()
            self.emit_actual_text_span(entry)

    def op_G(self: Any, o, d):
        if o:
            self.set_stroke_color(o[0])

    def op_RG(self: Any, o, d):
        if len(o) >= 3:
            self.set_stroke_color(o[0], o[1], o[2])

    def op_K(self: Any, o, d):
        if len(o) >= 4:
            self.set_stroke_color(o[0], o[1], o[2], o[3])

    def op_w(self: Any, o, d):
        if o:
            try:
                value = self.as_float(o[0])
            except (TypeError, ValueError):
                return
            self.line_width = max(0.0, value)

    def op_J(self: Any, o, d):
        if o:
            try:
                self.line_cap = self.as_int(o[0])
            except (TypeError, ValueError):
                return

    def op_j(self: Any, o, d):
        if o:
            try:
                self.line_join = self.as_int(o[0])
            except (TypeError, ValueError):
                return

    def op_M(self: Any, o, d):
        if o:
            try:
                value = self.as_float(o[0])
            except (TypeError, ValueError):
                return
            self.miter_limit = max(1.0, value)

    def op_d(self: Any, o, d):
        if o and len(o) >= 2:
            array_obj = o[0]
            try:
                phase = self.as_float(o[1])
                if isinstance(array_obj, (list, tuple)):
                    dash_array = [self.as_float(v) for v in array_obj]
                else:
                    dash_array = []
            except (TypeError, ValueError):
                return
            self.dash_pattern = (dash_array, phase)

    def op_m(self: Any, o, d):
        if len(o) >= 2:
            try:
                x = self.as_float(o[0])
                y = self.as_float(o[1])
            except (TypeError, ValueError):
                return
            if (
                self.capture_clipping
                or (self.capture_graphics or self.capture_glyphs)
                and self.is_graphics_visible()
            ):
                self.current_path.move_to(x, y)
            self.current_point = (x, y)
            self.subpath_start = self.current_point

    def op_l(self: Any, o, d):
        if len(o) >= 2 and self.current_point is not None:
            try:
                nx, ny = self.as_float(o[0]), self.as_float(o[1])
            except (TypeError, ValueError):
                return
            if (
                self.capture_clipping
                or (self.capture_graphics or self.capture_glyphs)
                and self.is_graphics_visible()
            ):
                self.current_path.line_to(nx, ny)
            self.current_point = (nx, ny)

    def op_re(self: Any, o, d):
        if len(o) >= 4:
            try:
                x, y = self.as_float(o[0]), self.as_float(o[1])
                w, h = self.as_float(o[2]), self.as_float(o[3])
            except (TypeError, ValueError):
                return
            if (
                self.capture_clipping
                or (self.capture_graphics or self.capture_glyphs)
                and self.is_graphics_visible()
            ):
                self.current_path.rect(x, y, w, h)
            self.current_point = (x, y)
            self.subpath_start = (x, y)

    def op_h(self: Any, o, d):
        if self.current_point is not None and self.subpath_start is not None:
            if (
                self.capture_clipping
                or (self.capture_graphics or self.capture_glyphs)
                and self.is_graphics_visible()
            ):
                self.current_path.close()
            self.current_point = self.subpath_start

    def op_c(self: Any, o, d):
        if len(o) >= 6:
            try:
                x1 = self.as_float(o[0])
                y1 = self.as_float(o[1])
                x2 = self.as_float(o[2])
                y2 = self.as_float(o[3])
                x3 = self.as_float(o[4])
                y3 = self.as_float(o[5])
            except (TypeError, ValueError):
                return
            self.append_cubic_curve(x1, y1, x2, y2, x3, y3)

    def op_v(self: Any, o, d):
        if len(o) >= 4 and self.current_point is not None:
            x0, y0 = self.current_point
            try:
                x2 = self.as_float(o[0])
                y2 = self.as_float(o[1])
                x3 = self.as_float(o[2])
                y3 = self.as_float(o[3])
            except (TypeError, ValueError):
                return
            self.append_cubic_curve(x0, y0, x2, y2, x3, y3)

    def op_y(self: Any, o, d):
        if len(o) >= 4 and self.current_point is not None:
            x0, y0 = self.current_point
            try:
                x1 = self.as_float(o[0])
                y1 = self.as_float(o[1])
                x3 = self.as_float(o[2])
                y3 = self.as_float(o[3])
            except (TypeError, ValueError):
                return
            self.append_cubic_curve(x0, y0, x1, y1, x3, y3)

    def op_paint_stroke(self: Any, o, d):
        if (
            self.capture_graphics
            and self.is_graphics_visible()
            and d == "s"
            and self.current_point is not None
            and self.subpath_start is not None
        ):
            self.current_path.close()
        self.flush_drawing("stroke")
        self.current_point = None
        self.subpath_start = None

    def op_paint_fill(self: Any, o, d):
        self.flush_drawing("fill", "evenodd" if d == "f*" else "nonzero")
        self.current_point = None
        self.subpath_start = None

    def op_paint_fillstroke(self: Any, o, d):
        if (
            self.capture_graphics
            and self.is_graphics_visible()
            and (d == "b" or d == "b*")
            and self.current_point is not None
            and self.subpath_start is not None
        ):
            self.current_path.close()
        self.flush_drawing("fillstroke", "evenodd" if d in {"B*", "b*"} else "nonzero")
        self.current_point = None
        self.subpath_start = None

    def op_paint_clear(self: Any, o, d):
        self.current_path.clear()
        self.current_point = None
        self.subpath_start = None

    def op_W(self: Any, o, d):
        path = self.current_path.transformed(self.ctm)
        if not path.has_segments():
            return
        clip_bbox = path.bbox()
        if clip_bbox is not None:
            if self.clip_bbox is None:
                self.clip_bbox = clip_bbox
            else:
                x0 = max(self.clip_bbox[0], clip_bbox[0])
                y0 = max(self.clip_bbox[1], clip_bbox[1])
                x1 = min(self.clip_bbox[2], clip_bbox[2])
                y1 = min(self.clip_bbox[3], clip_bbox[3])
                self.clip_bbox = (x0, y0, x1, y1)
        if self.capture_graphics and self.is_graphics_visible():
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=None,
                    fill_opacity=None,
                    blend_mode=self.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    line_width=0.0,
                    line_cap=self.line_cap,
                    line_join=self.line_join,
                    dash_pattern=self.transformed_dash_pattern(),
                    fill_rule="evenodd" if d == "W*" else "nonzero",
                    kind="clip",
                    path=path,
                )
            )

    def op_W_star(self: Any, o, d):
        self.op_W(o, d)

    def normalize_colors(self: Any, *components: Any) -> tuple[float, ...] | None:
        cache_key = components
        cached = self.color_normalization_cache.get(cache_key, MISSING)
        if cached is not MISSING:
            return cached
        values: list[float] = []
        for component in components:
            try:
                values.append(max(0.0, min(1.0, self.as_float(component))))
            except ValueError:
                self.color_normalization_cache[cache_key] = None
                return None
        if not values:
            self.color_normalization_cache[cache_key] = None
            return None
        normalized = tuple(values)
        self.color_normalization_cache[cache_key] = normalized
        return normalized

    def set_stroke_color(self: Any, *components: Any) -> None:
        normalized = self.normalize_colors(*components)
        if normalized is not None:
            self.stroke_color = normalized
            self.stroke_pattern = None

    def set_fill_color(self: Any, *components: Any) -> None:
        normalized = self.normalize_colors(*components)
        if normalized is not None:
            self.fill_color = normalized
            self.fill_pattern = None

    def normalize_color_operands(self: Any, o) -> tuple[float, ...] | None:
        count = len(o)
        if count == 1:
            c0 = o[0]
            t0 = type(c0)
            if t0 is float or t0 is int:
                if c0 < 0.0:
                    return (0.0,)
                if c0 > 1.0:
                    return (1.0,)
                return (float(c0),)
            return self.normalize_colors(c0)
        if count == 3:
            c0 = o[0]
            c1 = o[1]
            c2 = o[2]
            t0 = type(c0)
            t1 = type(c1)
            t2 = type(c2)
            if (
                (t0 is float or t0 is int)
                and (t1 is float or t1 is int)
                and (t2 is float or t2 is int)
            ):
                return (
                    max(0.0, min(1.0, float(c0))),
                    max(0.0, min(1.0, float(c1))),
                    max(0.0, min(1.0, float(c2))),
                )
            return self.normalize_colors(c0, c1, c2)
        if count == 4:
            c0 = o[0]
            c1 = o[1]
            c2 = o[2]
            c3 = o[3]
            t0 = type(c0)
            t1 = type(c1)
            t2 = type(c2)
            t3 = type(c3)
            if (
                (t0 is float or t0 is int)
                and (t1 is float or t1 is int)
                and (t2 is float or t2 is int)
                and (t3 is float or t3 is int)
            ):
                return (
                    max(0.0, min(1.0, float(c0))),
                    max(0.0, min(1.0, float(c1))),
                    max(0.0, min(1.0, float(c2))),
                    max(0.0, min(1.0, float(c3))),
                )
            return self.normalize_colors(c0, c1, c2, c3)
        return self.normalize_colors(*o)

    def resolve_color_space(
        self: Any, name_obj: Any, *, default_fallback: bool = False
    ) -> str | None:
        try:
            cache_key = (self.resources_id, name_obj, default_fallback)
            cached = self.color_space_cache.get(cache_key, MISSING)
            if cached is not MISSING:
                return cached
        except TypeError:
            cache_key = None

        name = self.resolve_name(name_obj)
        if name is None:
            color_space = "DeviceGray" if default_fallback else None
            if cache_key is not None:
                self.color_space_cache[cache_key] = color_space
            return color_space

        color_space = self.lookup_page_resource("ColorSpace", name)
        if color_space is None:
            resolved = name if default_fallback else None
            if cache_key is not None:
                self.color_space_cache[cache_key] = resolved
            return resolved

        color_space_name = normalize_pdf_name(color_space)
        if color_space_name is not None:
            if cache_key is not None:
                self.color_space_cache[cache_key] = color_space_name
            return color_space_name
        if isinstance(color_space, (list, tuple)) and color_space:
            base = color_space[0]
            base_name = normalize_pdf_name(base)
            if base_name is not None:
                if cache_key is not None:
                    self.color_space_cache[cache_key] = base_name
                return base_name

        if isinstance(name, str) and not name.startswith("/"):
            if cache_key is not None:
                self.color_space_cache[cache_key] = name
            return name

        resolved = name if default_fallback else None
        if cache_key is not None:
            self.color_space_cache[cache_key] = resolved
        return resolved

    def resolve_pattern_color(self: Any, operands: tuple[Any, ...]) -> PdfDict | None:
        if not operands:
            return None
        pattern_name = self.resolve_name(operands[-1])
        if not pattern_name:
            return None
        pattern = self.lookup_page_resource("Pattern", pattern_name)
        if isinstance(pattern, PdfStream):
            pattern_dict = self.resolve_dict(pattern.dictionary)
        else:
            pattern_dict = self.resolve_dict(pattern) if pattern is not None else None
        if not isinstance(pattern_dict, dict):
            return None
        pattern_type = parse_int(lookup_dict_key(pattern_dict, "PatternType"), None)
        if pattern_type == 2:
            shading = lookup_dict_key(pattern_dict, "Shading")
            shading = self.resolve(shading)
            shading_dict = self.resolve_dict(shading) if shading is not None else None
            if not isinstance(shading_dict, dict):
                return None
            return cast(PdfDict, {"kind": "shading", "dictionary": dict(shading_dict)})
        if pattern_type != 1 or not isinstance(pattern, PdfStream):
            return None
        paint_type = parse_int(lookup_dict_key(pattern_dict, "PaintType"), 1)
        if paint_type not in {1, 2}:
            return None
        base_color = None
        if paint_type == 2:
            base_color = self.normalize_color_operands(operands[:-1])
            if base_color is None:
                return None
        bbox = self.document.resolver.resolve_box(lookup_dict_key(pattern_dict, "BBox"))
        if bbox is None:
            return None
        x_step = self.document.resolver.resolve_float(
            lookup_dict_key(pattern_dict, "XStep"), default=None
        )
        y_step = self.document.resolver.resolve_float(
            lookup_dict_key(pattern_dict, "YStep"), default=None
        )
        if x_step is None or y_step is None or x_step == 0.0 or y_step == 0.0:
            return None
        try:
            matrix = Matrix.from_operand(lookup_dict_key(pattern_dict, "Matrix"))
        except ValueError:
            matrix = IDENTITY_MATRIX
        resources = self.resolve_dict(lookup_dict_key(pattern_dict, "Resources")) or {}
        nested_state = type(self)(
            self.document,
            self.page,
            hidden_layers=self.hidden_layers,
            capture_runs=True,
            capture_glyphs=True,
            capture_graphics=True,
            decoder_cache=self.decoder_cache,
        )
        try:
            nested_state.consume_stream(pattern, resources, matrix, 0)
        except Exception:
            return None
        return cast(
            PdfDict,
            {
                "kind": "tiling",
                "bbox": tuple(bbox),
                "x_step": float(x_step),
                "y_step": float(y_step),
                "drawings": [
                    {
                        "kind": drawing.kind,
                        "fill": (
                            base_color if drawing.kind in {"fill", "fillstroke"} else drawing.fill
                        ),
                        "fill_pattern": drawing.fill_pattern,
                        "fill_opacity": drawing.fill_opacity,
                        "stroke_color": (
                            base_color
                            if drawing.kind in {"stroke", "fillstroke"}
                            else drawing.stroke_color
                        ),
                        "stroke_pattern": drawing.stroke_pattern,
                        "stroke_opacity": drawing.stroke_opacity,
                        "line_width": drawing.line_width,
                        "line_cap": drawing.line_cap,
                        "line_join": drawing.line_join,
                        "dash_pattern": drawing.dash_pattern,
                        "fill_rule": drawing.fill_rule,
                        "blend_mode": drawing.blend_mode,
                        "soft_mask_alpha": drawing.soft_mask_alpha,
                        "raw_data": drawing.raw_data,
                        "dictionary": drawing.dictionary,
                        "items": list(drawing.items),
                        "path": drawing.path,
                        "rect": drawing.rect,
                    }
                    for drawing in nested_state.drawings
                ],
                "runs": [
                    {
                        "text": run.text,
                        "bbox": (run.x0, run.y0, run.x1, run.y1),
                        "fill_color": run.fill_color,
                        "visible": run.visible,
                    }
                    for run in nested_state.runs
                ],
                "glyphs": [
                    {
                        "text": glyph.text,
                        "bbox": (
                            glyph.ink_rect.x0,
                            glyph.ink_rect.y0,
                            glyph.ink_rect.x1,
                            glyph.ink_rect.y1,
                        ),
                        "advance_bbox": (
                            glyph.advance_rect.x0,
                            glyph.advance_rect.y0,
                            glyph.advance_rect.x1,
                            glyph.advance_rect.y1,
                        ),
                        "fill_color": glyph.fill,
                        "visible": glyph.visible,
                        "code": glyph.cid,
                        "gid": glyph.gid,
                        "font_name": glyph.font_name,
                        "unicode_source": glyph.unicode_source,
                        "alternates": glyph.alternates,
                        "bitmap": glyph.bitmap,
                        "bitmap_width": glyph.bitmap_width,
                        "bitmap_height": glyph.bitmap_height,
                    }
                    for glyph in nested_state.glyphs
                    if glyph.bitmap
                ],
            },
        )

    def op_CS(self: Any, o, d):
        if o:
            name_obj = o[0]
            try:
                cached = self.color_space_cache.get((self.resources_id, name_obj, True), MISSING)
            except TypeError:
                cached = MISSING
            if cached is not MISSING:
                if cached is not None and self.stroke_color_space != cached:
                    self.stroke_color_space = cached
                return
            color_space = self.resolve_color_space(name_obj, default_fallback=True)
            if color_space is not None:
                self.stroke_color_space = color_space

    def op_cs(self: Any, o, d):
        if o:
            name_obj = o[0]
            try:
                cached = self.color_space_cache.get((self.resources_id, name_obj, True), MISSING)
            except TypeError:
                cached = MISSING
            if cached is not MISSING:
                if cached is not None and self.fill_color_space != cached:
                    self.fill_color_space = cached
                return
            color_space = self.resolve_color_space(name_obj, default_fallback=True)
            if color_space is not None:
                self.fill_color_space = color_space

    def op_SC(self: Any, o, d):
        normalized = self.normalize_color_operands(o)
        if normalized is not None:
            self.stroke_color = normalized
            self.stroke_pattern = None

    def op_SCN(self: Any, o, d):
        if self.stroke_color_space == "Pattern":
            self.stroke_pattern = self.resolve_pattern_color(o)
            if len(o) > 1:
                normalized = self.normalize_color_operands(o[:-1])
                if normalized is not None:
                    self.stroke_color = normalized
            return
        normalized = self.normalize_color_operands(o)
        if normalized is not None:
            self.stroke_color = normalized
            self.stroke_pattern = None

    def op_sc(self: Any, o, d):
        normalized = self.normalize_color_operands(o)
        if normalized is not None:
            self.fill_color = normalized
            self.fill_pattern = None

    def op_scN(self: Any, o, d):
        if self.fill_color_space == "Pattern":
            self.fill_pattern = self.resolve_pattern_color(o)
            if len(o) > 1:
                normalized = self.normalize_color_operands(o[:-1])
                if normalized is not None:
                    self.fill_color = normalized
            return
        normalized = self.normalize_color_operands(o)
        if normalized is not None:
            self.fill_color = normalized
            self.fill_pattern = None

    def op_i(self: Any, o, d):
        if o:
            try:
                value = self.as_float(o[0])
            except ValueError:
                return
            self.flatness = max(0, min(100, int(value)))

    def op_ri(self: Any, o, d):
        if not o:
            return
        value = self.resolve_name_like_value(o[0])
        if isinstance(value, str):
            self.render_intent = value

    def op_MP(self: Any, o, d):
        self.marked_content_stack.append(MarkedContentEntry())

    def op_DP(self: Any, o, d):
        tag = self.resolve_name(o[0]) if o else None
        layer = None
        if tag == "OC" and len(o) >= 2:
            layer = self.resolve_marked_content_layer(o[1])
        actual_text = (
            self.resolve_marked_content_actual_text(o[1]) if tag == "Span" and len(o) >= 2 else None
        )
        self.marked_content_stack.append(MarkedContentEntry(layer=layer, actual_text=actual_text))

    def resolve_marked_content_actual_text(self: Any, value: Any) -> str | None:
        props = self.resolve_marked_content_properties(value)
        if not isinstance(props, dict):
            return None
        return self.resolve_str(lookup_dict_key(props, "ActualText"))

    def resolve_marked_content_properties(self: Any, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        resolved = self.resolve(value)
        if isinstance(resolved, dict):
            return resolved
        name = self.resolve_name(value)
        if not name:
            return None
        props = self.lookup_page_resource("Properties", name)
        return props if isinstance(props, dict) else None

    def resolve_marked_content_layer(self: Any, value: Any) -> str | None:
        if value is None:
            return None

        resolved = self.resolve(value)
        if isinstance(resolved, dict):
            oc = lookup_dict_key(resolved, "OC")
            if oc is not None:
                return self.resolve_name(oc) or self.resolve_str(oc)

        return self.resolve_name(value) or self.resolve_str(value)

    def op_BX(self: Any, o, d):
        self.compatibility_depth += 1

    def op_EX(self: Any, o, d):
        self.compatibility_depth = max(0, self.compatibility_depth - 1)

    def op_d0(self: Any, o, d):
        return

    def op_d1(self: Any, o, d):
        return

    def op_sh(self: Any, o, d):
        if not o or not self.capture_graphics or not self.is_graphics_visible():
            return
        shading_ref = self.resolve_name(o[0])
        if not shading_ref:
            return
        shading = self.lookup_page_resource("Shading", shading_ref)
        if not isinstance(shading, dict):
            return
        self.drawings.append(
            CapturedDrawing(
                seqno=self.sequence,
                fill=self.fill_color,
                fill_opacity=self.fill_opacity,
                stroke_color=self.stroke_color,
                stroke_opacity=self.stroke_opacity,
                line_width=self.line_width,
                line_cap=self.line_cap,
                line_join=self.line_join,
                dash_pattern=self.transformed_dash_pattern(),
                blend_mode=self.blend_mode,
                soft_mask_alpha=self.group_alpha,
                kind="shading",
                items=[],
                dictionary=dict(shading),
            )
        )

    @staticmethod
    def as_float(value: Any) -> float:
        value_type = type(value)
        if value_type is float:
            return value
        if value_type is int:
            return float(value)
        if value_type is bool:
            raise ValueError("invalid numeric operand")
        parsed = parse_float(value, None)
        if parsed is None:
            raise ValueError("invalid numeric operand")
        return parsed

    @staticmethod
    def as_int(value: Any) -> int:
        if type(value) is bool:
            raise ValueError("invalid numeric operand")
        parsed = parse_int(value, None)
        if parsed is None:
            raise ValueError("invalid numeric operand")
        return parsed

    def resolve_extgstate(self: Any, name: str) -> dict[str, Any] | None:
        cache_key = (self.resources_id, name)
        cached = self.extgstate_cache.get(cache_key, MISSING)
        if cached is not MISSING:
            return cached
        resolved = self.lookup_page_resource("ExtGState", name)
        if not isinstance(resolved, dict):
            self.extgstate_cache[cache_key] = None
            return None
        self.extgstate_cache[cache_key] = resolved
        return resolved

    def op_q(self: Any, o, d):
        if self.capture_graphics:
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=None,
                    fill_opacity=None,
                    kind="state-push",
                )
            )
        self.stack.append(
            (
                self.ca,
                self.cb,
                self.cc,
                self.cd,
                self.ce,
                self.cf,
                self.fill_color,
                self.fill_pattern,
                self.fill_opacity,
                self.stroke_color,
                self.stroke_pattern,
                self.stroke_opacity,
                self.fill_color_space,
                self.stroke_color_space,
                self.compatibility_depth,
                self.blend_mode,
                self.group_alpha,
                self.flatness,
                self.render_intent,
                self.clip_bbox,
                self.line_width,
                self.line_cap,
                self.line_join,
                self.miter_limit,
                self.dash_pattern,
                self.font_size,
                self.font_operand,
                self.font_size_operand,
                self.horizontal_scale,
                self.char_space,
                self.word_space,
                self.rise,
                self.leading,
                self.render_mode,
                self.current_font,
                self.current_decoder,
            )
        )

    def op_Q(self: Any, o, d):
        if self.capture_graphics:
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=None,
                    fill_opacity=None,
                    kind="state-pop",
                )
            )
        if self.stack:
            (
                self.ca,
                self.cb,
                self.cc,
                self.cd,
                self.ce,
                self.cf,
                self.fill_color,
                self.fill_pattern,
                self.fill_opacity,
                self.stroke_color,
                self.stroke_pattern,
                self.stroke_opacity,
                self.fill_color_space,
                self.stroke_color_space,
                self.compatibility_depth,
                self.blend_mode,
                self.group_alpha,
                self.flatness,
                self.render_intent,
                self.clip_bbox,
                self.line_width,
                self.line_cap,
                self.line_join,
                self.miter_limit,
                self.dash_pattern,
                self.font_size,
                self.font_operand,
                self.font_size_operand,
                self.horizontal_scale,
                self.char_space,
                self.word_space,
                self.rise,
                self.leading,
                self.render_mode,
                self.current_font,
                self.current_decoder,
            ) = self.stack.pop()
            self.update_combined()

    def op_cm(self: Any, o, d):
        if not o or len(o) < 6:
            return
        if len(o) > 6:
            o = o[:6]
        try:
            m_a, m_b, m_c, m_d, m_e, m_f = (
                self.as_float(o[0]),
                self.as_float(o[1]),
                self.as_float(o[2]),
                self.as_float(o[3]),
                self.as_float(o[4]),
                self.as_float(o[5]),
            )
        except (TypeError, ValueError):
            return

        self.op_cm_values(m_a, m_b, m_c, m_d, m_e, m_f)

    def op_cm_values(
        self: Any,
        m_a: float,
        m_b: float,
        m_c: float,
        m_d: float,
        m_e: float,
        m_f: float,
    ) -> None:
        ca, cb, cc, cd, ce, cf = self.ca, self.cb, self.cc, self.cd, self.ce, self.cf
        self.ca = m_a * ca + m_b * cc
        self.cb = m_a * cb + m_b * cd
        self.cc = m_c * ca + m_d * cc
        self.cd = m_c * cb + m_d * cd
        self.ce = m_e * ca + m_f * cc + ce
        self.cf = m_e * cb + m_f * cd + cf
        self.update_combined()

    def op_g(self: Any, o, d):
        if o:
            self.set_fill_color(o[0])

    def op_rg(self: Any, o, d):
        if len(o) >= 3:
            self.set_fill_color(o[0], o[1], o[2])

    def op_k(self: Any, o, d):
        if len(o) >= 4:
            self.set_fill_color(o[0], o[1], o[2], o[3])

    def op_gs(self: Any, o, d):
        if not o:
            return
        name = self.resolve_name(o[0])
        if not name:
            return
        extgstate = self.resolve_extgstate(name)
        if not extgstate:
            return
        try:
            fill_opacity = lookup_dict_key(extgstate, "ca")
            if fill_opacity is not None:
                self.fill_opacity = max(0.0, min(1.0, self.as_float(fill_opacity)))
            stroke_opacity = lookup_dict_key(extgstate, "CA")
            if stroke_opacity is not None:
                self.stroke_opacity = max(0.0, min(1.0, self.as_float(stroke_opacity)))
            blend_mode = lookup_dict_key(extgstate, "BM")
            if blend_mode is not None:
                if isinstance(blend_mode, (list, tuple)):
                    blend_mode = blend_mode[0] if blend_mode else None
                if blend_mode is not None:
                    self.blend_mode = self.resolve_name_like_value(blend_mode)
        except (TypeError, ValueError):
            return


__all__ = ("OperatorMixin", "detect_rotation_from_linear")
