# SPDX-License-Identifier: AGPL-3.0-only
"""Small stateful collaborators used by the content-stream operation target."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core_pdf.impl.engine.spec.s_07_content.capture import CapturedDrawing
from core_pdf.impl.engine.spec.s_07_content.marked_content import MarkedContentEntry
from core_pdf.impl.objects import PdfString


class GraphicsComponent:
    """Own graphics-state stack and CTM mutations for a content stream host."""

    def __init__(self, host: Any) -> None:
        self.host = host

    def save(self) -> None:
        state = self.host
        state.clip_scope_stack.append(False)
        state.stack.append(
            (
                state.ca,
                state.cb,
                state.cc,
                state.cd,
                state.ce,
                state.cf,
                state.fill_color,
                state.fill_pattern,
                state.fill_opacity,
                state.stroke_color,
                state.stroke_pattern,
                state.stroke_opacity,
                state.fill_color_space,
                state.stroke_color_space,
                state.compatibility_depth,
                state.blend_mode,
                state.group_alpha,
                state.flatness,
                state.render_intent,
                state.clip_bbox,
                state.line_width,
                state.line_cap,
                state.line_join,
                state.miter_limit,
                state.dash_pattern,
                state.font_size,
                state.font_operand,
                state.font_size_operand,
                state.horizontal_scale,
                state.char_space,
                state.word_space,
                state.rise,
                state.leading,
                state.render_mode,
                state.current_font,
                state.current_decoder,
            )
        )

    def restore(self) -> None:
        state = self.host
        clip_scope_emitted = state.clip_scope_stack.pop() if state.clip_scope_stack else False
        if state.capture_graphics and clip_scope_emitted:
            state.drawings.append(
                CapturedDrawing(
                    seqno=state.sequence,
                    fill=None,
                    fill_opacity=None,
                    kind="state-pop",
                )
            )
        if not state.stack:
            return
        (
            state.ca,
            state.cb,
            state.cc,
            state.cd,
            state.ce,
            state.cf,
            state.fill_color,
            state.fill_pattern,
            state.fill_opacity,
            state.stroke_color,
            state.stroke_pattern,
            state.stroke_opacity,
            state.fill_color_space,
            state.stroke_color_space,
            state.compatibility_depth,
            state.blend_mode,
            state.group_alpha,
            state.flatness,
            state.render_intent,
            state.clip_bbox,
            state.line_width,
            state.line_cap,
            state.line_join,
            state.miter_limit,
            state.dash_pattern,
            state.font_size,
            state.font_operand,
            state.font_size_operand,
            state.horizontal_scale,
            state.char_space,
            state.word_space,
            state.rise,
            state.leading,
            state.render_mode,
            state.current_font,
            state.current_decoder,
        ) = state.stack.pop()
        state.update_combined()
        # Text-state values are included in our graphics-state snapshot for
        # compatibility with malformed producers that change them inside q/Q.
        # Restore their derived scales and metrics together with the raw
        # values; otherwise glyph advances continue using the inner state's
        # font size or horizontal scale.
        state.update_text_scales()
        state.update_font_metrics()

    def concatenate(self, values: tuple[float, float, float, float, float, float]) -> None:
        state = self.host
        m_a, m_b, m_c, m_d, m_e, m_f = values
        ca, cb, cc, cd, ce, cf = state.ca, state.cb, state.cc, state.cd, state.ce, state.cf
        state.ca = m_a * ca + m_b * cc
        state.cb = m_a * cb + m_b * cd
        state.cc = m_c * ca + m_d * cc
        state.cd = m_c * cb + m_d * cd
        state.ce = m_e * ca + m_f * cc + ce
        state.cf = m_e * cb + m_f * cd + cf
        state.update_combined()

    def set_stroke_gray(self, operands: Any) -> None:
        if operands:
            self.host.set_stroke_color(operands[0])

    def set_stroke_rgb(self, operands: Any) -> None:
        if len(operands) >= 3:
            self.host.set_stroke_color(operands[0], operands[1], operands[2])

    def set_stroke_cmyk(self, operands: Any) -> None:
        if len(operands) >= 4:
            self.host.set_stroke_color(operands[0], operands[1], operands[2], operands[3])

    def set_line_width(self, operands: Any) -> None:
        if operands:
            try:
                self.host.line_width = max(0.0, self.host.as_float(operands[0]))
            except (TypeError, ValueError):
                return

    def set_line_cap(self, operands: Any) -> None:
        if operands:
            try:
                self.host.line_cap = self.host.as_int(operands[0])
            except (TypeError, ValueError):
                return

    def set_line_join(self, operands: Any) -> None:
        if operands:
            try:
                self.host.line_join = self.host.as_int(operands[0])
            except (TypeError, ValueError):
                return

    def set_miter_limit(self, operands: Any) -> None:
        if operands:
            try:
                self.host.miter_limit = max(1.0, self.host.as_float(operands[0]))
            except (TypeError, ValueError):
                return

    def set_dash_pattern(self, operands: Any) -> None:
        if not operands or len(operands) < 2:
            return
        try:
            phase = self.host.as_float(operands[1])
            array_obj = operands[0]
            dash_array = (
                [self.host.as_float(value) for value in array_obj]
                if isinstance(array_obj, (list, tuple))
                else []
            )
        except (TypeError, ValueError):
            return
        self.host.dash_pattern = (dash_array, phase)


class TextComponent:
    """Own the small, high-frequency text-state transitions."""

    def __init__(self, host: Any) -> None:
        self.host = host

    def begin(self) -> None:
        state = self.host
        if state.pending_run:
            state.flush_run()
        state.tm_a = state.lm_a = 1.0
        state.tm_b = state.lm_b = 0.0
        state.tm_c = state.lm_c = 0.0
        state.tm_d = state.lm_d = 1.0
        state.tm_e = state.lm_e = 0.0
        state.tm_f = state.lm_f = 0.0
        state.compat_tj_cursor_x = 0.0
        state.compat_tj_cursor_y = 0.0
        state.combined_A = state.ca
        state.combined_B = state.cb
        state.combined_C = state.cc
        state.combined_D = state.cd
        if state.ca == 1.0 and state.cb == 0.0 and state.cc == 0.0 and state.cd == 1.0:
            state.cached_rotation = 0
        else:
            state.cached_rotation = state.detect_rotation(state.ca, state.cb, state.cc, state.cd)
        state.invisible_text_layer = False

    def end(self) -> None:
        state = self.host
        pending = state.pending_run
        if pending:
            if state.capture_runs:
                state.runs.append(pending)
            state.pending_run = None

    def move(self, tx: float, ty: float) -> None:
        state = self.host
        if state.pending_run:
            state.runs.append(state.pending_run)
            state.pending_run = None
        # Preserve the specification's affine operation order. Exact layout
        # grouping can hinge on the final ULP at a character-margin boundary.
        state.tm_e = tx * state.lm_a + ty * state.lm_c + state.lm_e
        state.tm_f = tx * state.lm_b + ty * state.lm_d + state.lm_f
        state.lm_e = state.tm_e
        state.lm_f = state.tm_f
        state.compat_tj_cursor_x = 0.0
        state.compat_tj_cursor_y = 0.0

    def set_matrix(self, a: float, b: float, c: float, d: float, e: float, f: float) -> None:
        state = self.host
        state.flush_run()
        state.tm_a = state.lm_a = a
        state.tm_b = state.lm_b = b
        state.tm_c = state.lm_c = c
        state.tm_d = state.lm_d = d
        state.tm_e = state.lm_e = e
        state.tm_f = state.lm_f = f
        state.compat_tj_cursor_x = 0.0
        state.compat_tj_cursor_y = 0.0
        state.update_combined()

    def set_leading_and_move(self, tx: float, ty: float) -> None:
        state = self.host
        state.leading = -ty
        self.move(tx, ty)

    def set_char_space(self, value: float) -> None:
        state = self.host
        state.char_space = value
        state.update_char_space_scale()

    def set_word_space(self, value: float) -> None:
        state = self.host
        if state.word_space == value:
            return
        state.word_space = value
        state.update_word_space_scale()

    def move_operands(self, operands: Any, *, set_leading: bool = False) -> None:
        if len(operands) < 2:
            return
        try:
            tx = self.host.as_float(operands[0])
            ty = self.host.as_float(operands[1])
        except (TypeError, ValueError):
            return
        if set_leading:
            self.host.leading = -ty
        self.move(tx, ty)

    def show(self, operand: Any) -> None:
        state = self.host
        decoder = (
            state.current_decoder if state.current_decoder is not None else state.get_decoder()
        )
        if type(operand) is PdfString:
            state.append_text(
                data=operand.data,
                decoder=decoder,
                string_syntax="literal" if operand.is_literal else "hex",
                compatibility_data=operand.compatibility_data,
            )
        else:
            state.append_text(operand, decoder=decoder)

    def show_array(self, operands: Any) -> None:
        if operands:
            self.host.append_tj_array(operands[0])

    def set_leading_operand(self, operands: Any) -> None:
        if not operands:
            return
        try:
            self.host.leading = self.host.as_float(operands[0])
        except (TypeError, ValueError):
            return

    def set_render_mode_operand(self, operands: Any) -> None:
        if not operands:
            return
        try:
            self.host.render_mode = self.host.as_int(operands[0])
        except (TypeError, ValueError):
            return

    def set_horizontal_scale_operand(self, operands: Any) -> None:
        if not operands:
            return
        try:
            self.host.horizontal_scale = self.host.as_float(operands[0])
        except (TypeError, ValueError):
            return
        self.host.update_horizontal_text_scale()

    def set_rise_operand(self, operands: Any) -> None:
        if not operands:
            return
        try:
            self.host.rise = self.host.as_float(operands[0])
        except (TypeError, ValueError):
            return

    def quote(self, operands: Any) -> None:
        if not operands:
            return
        self.move(0.0, -self.host.leading)
        self.host.pending_line_break = True
        self.show(operands[0])

    def double_quote(self, operands: Any) -> None:
        if len(operands) < 3:
            return
        try:
            word_space = self.host.as_float(operands[0])
            char_space = self.host.as_float(operands[1])
        except (TypeError, ValueError):
            return
        self.host.word_space = word_space
        self.host.char_space = char_space
        self.host.update_char_space_scale()
        self.host.update_word_space_scale()
        if not getattr(self.host.document, "legacy_pdfminer_text_operators", False):
            self.move(0.0, -self.host.leading)
            self.host.pending_line_break = True
        self.show(operands[2])

    def set_spacing_operand(self, operands: Any, setter: Callable[[float], None]) -> None:
        if not operands:
            return
        try:
            value = self.host.as_float(operands[0])
        except (TypeError, ValueError):
            return
        setter(value)

    def set_char_space_operand(self, operands: Any) -> None:
        self.set_spacing_operand(operands, self.set_char_space)

    def set_word_space_operand(self, operands: Any) -> None:
        self.set_spacing_operand(operands, self.set_word_space)


class ContentComponent:
    """Own marked-content stack transitions while the host retains resources."""

    def __init__(self, host: Any) -> None:
        self.host = host

    def begin_with_properties(self, operands: Any) -> None:
        """`BDC` — open a marked-content scope carrying a property list."""
        state = self.host
        tag = state.document.resolver.resolve_name(operands[0]) if operands else None
        layer = (
            state.resolve_marked_content_layer(operands[1])
            if tag == "OC" and len(operands) >= 2
            else None
        )
        actual_text = (
            state.resolve_marked_content_actual_text(operands[1])
            if tag == "Span" and len(operands) >= 2
            else None
        )
        mcid = state.resolve_marked_content_mcid(operands[1]) if len(operands) >= 2 else None
        state.marked_content_stack.append(
            MarkedContentEntry(layer=layer, actual_text=actual_text, mcid=mcid)
        )

    def begin(self) -> None:
        """`BMC` — open a marked-content scope with no property list."""
        self.host.marked_content_stack.append(MarkedContentEntry())

    def end(self) -> None:
        """`EMC` — close the innermost scope opened by `BMC` or `BDC`."""
        state = self.host
        if state.marked_content_stack:
            state.emit_actual_text_span(state.marked_content_stack.pop())

    def mark_point(self) -> None:
        """`MP` — a marked-content point.

        Points are not scopes: only `BMC` and `BDC` open one, and only `EMC`
        closes one.  Pushing here would leave an entry that the next `EMC`
        pops in place of its own, stranding the real scope on the stack.
        """

    def mark_point_with_properties(self, internal_operands: Any) -> None:
        """`DP` — a marked-content point carrying a property list.

        Opens no scope, so its `/OC` layer, `/ActualText`, and `/MCID` apply to
        the point alone and must not be inherited by the content that follows.
        """
