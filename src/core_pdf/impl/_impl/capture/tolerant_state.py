# SPDX-License-Identifier: AGPL-3.0-only
"""Reader repairs and substitutions around strict PDF content transitions."""

from __future__ import annotations

import typing
from typing import Any, cast

from core_pdf.impl._impl.capture.recovery import CaptureRecovery
from core_pdf.impl._impl.document.recovery.resources import (
    resolve_resource_dict as recover_resources,
)
from core_pdf.impl._impl.graphics.color_spec import color_spec_from_value
from core_pdf.impl._impl.model.geometry import transform_bbox
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.operations import (
    ContentOperands,
)
from core_pdf_spec.s_07_content.patterns import PatternPaint, ShadingPattern, TilingPattern
from core_pdf_spec.s_07_content.state import TextState as SpecTextState
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
)
from core_pdf_spec.s_08_graphics.color_spec import ImageColorSpec
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph as DecodedGlyph
from core_pdf_spec.s_09_fonts.service import FontService as FontDecoder
from core_pdf_spec.types import PdfReference, PdfString


class RecoveringTextState(SpecTextState):
    """Reader repairs layered over the shared PDF state transitions."""

    recovery: CaptureRecovery

    def get_decoder(self, *, update_metrics: bool = True) -> "FontDecoder":
        if self.current_decoder is not None:
            return self.current_decoder

        try:
            font_obj_ref = (
                self.lookup_page_resource("Font", self.current_font) if self.current_font else None
            )
        except PdfParseError as error:
            self.handle_operand_error(error, "font-resource")
            font_obj_ref = None
        if font_obj_ref is None:
            return self.font_provider({}, typing.cast(dict[str, Any], self.resources))

        try:
            font_obj = self.document.resolver.resolve(font_obj_ref)
        except PdfParseError as error:
            self.handle_operand_error(error, "font-resolution")
            font_obj = None
        if isinstance(font_obj, PdfStream):
            font_obj = font_obj.dictionary
        if not isinstance(font_obj, dict):
            decoder = self.font_provider({}, typing.cast(dict[str, Any], self.resources))
            self.current_decoder = decoder
            self.current_decoder_resources_id = self.resources_id
            if update_metrics:
                self.update_font_metrics()
            return decoder

        font_dict = typing.cast(PdfDict, font_obj)
        resolved_font = self.document.resolver.resolve_font_dict(font_dict)
        decoder = self.font_provider(
            typing.cast(dict[str, Any], resolved_font), typing.cast(dict[str, Any], self.resources)
        )
        self.current_decoder = decoder
        self.current_decoder_resources_id = self.resources_id
        if update_metrics:
            self.update_font_metrics()
        return decoder

    def append_xobject(self, name_obj: Any, depth: int) -> None:
        name = self.document.resolver.resolve_name(name_obj)
        if not name:
            return
        raw_xobj = self.lookup_page_resource("XObject", name)
        stream_key = (
            ("ref", raw_xobj.object_number, raw_xobj.generation_number)
            if isinstance(raw_xobj, PdfReference)
            else None
        )
        xobj = self.document.resolver.resolve(raw_xobj)
        if not isinstance(xobj, PdfStream):
            return
        xobj_dict = xobj.dictionary
        subtype = self.document.resolver.resolve_name(xobj_dict.get("Subtype"))
        if self.document.resolver.resolve_name(xobj_dict.get("Type")) == "ObjStm":
            return
        if subtype == "Image":
            self.sink.paint_image(self, xobj)
            return
        if subtype != "Form":
            return
        group_alpha = None
        group = xobj_dict.get("Group")
        if group is not None:
            group_dict = self.document.resolver.resolve_dict(group)
            if (
                isinstance(group_dict, dict)
                and self.document.resolver.resolve_name(group_dict.get("S")) == "Transparency"
            ):
                # PDF 32000-1 Table 147: a transparency group dictionary holds
                # S/CS/I/K and nothing else. The constant alpha and blend mode
                # that composite the finished group into its backdrop come from
                # the graphics state in effect at the `Do` (11.6.6), so reading
                # a /ca off the group dictionary found nothing and dropped the
                # group entirely -- the contents then painted straight onto the
                # page at full opacity in Normal mode, losing the blend.
                #
                # An explicitly isolated group has its own transparent backdrop,
                # even at full opacity: a child's blend must not see the page.
                blend = self.blend_mode
                isolated = self.document.resolver.resolve(group_dict.get("I")) is True
                if isolated or self.fill_opacity < 1.0 or (blend is not None and blend != "Normal"):
                    group_alpha = max(0.0, min(1.0, self.fill_opacity))
        resources = self.resolve_resources(xobj_dict.get("Resources")) or self.resources
        xobj_matrix = xobj_dict.get("Matrix")
        nested_ctm = self.matrix_operand(xobj_matrix, "form").multiply(self.ctm)
        raw_form_bbox = xobj_dict.get("BBox")
        form_bbox = self.document.resolver.resolve_box(raw_form_bbox)
        transformed_form_bbox = (
            transform_bbox(form_bbox, nested_ctm) if form_bbox is not None else None
        )
        self.stream_executor.queue(
            xobj,
            resources,
            nested_ctm,
            depth + 1,
            clip_bbox=transformed_form_bbox,
            form_bbox_operand=raw_form_bbox,
            group_alpha=group_alpha,
            stream_key=stream_key,
        )

    def decode_operand(
        self, operand: object, decoder: FontDecoder
    ) -> tuple[str, bytes, tuple[DecodedGlyph, ...]]:
        text: str | None
        if type(operand) is PdfString:
            data, text = operand.data, None
        elif type(operand) is bytes:
            data, text = operand, None
        elif type(operand) is str:
            data, text = operand.encode("latin-1", "replace"), operand
        else:
            text = self.document.resolver.resolve_str(operand)
            if text is None:
                return "", b"", ()
            data = text.encode("latin-1", "replace")

        glyphs = decoder.decode_glyphs(data if isinstance(data, bytes) else bytes(data))
        if text is None:
            text = "".join([glyph.unicode for glyph in glyphs])
        return text, data, glyphs

    def append_tj_array(self, array: Any) -> None:
        if not isinstance(array, (list, tuple)):
            return
        super().append_tj_array(array)

    def tj_array_extra_bytes(self, item: object) -> bytes:
        """Retain Latin-1 text and skip other unsupported reader entries."""
        return item.encode("latin-1") if type(item) is str else b""

    def op_Tj(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        # Operators consume their operands from the top of the operand stack.
        # A well-formed Tj has exactly one string, but damaged streams sometimes
        # leave older operands before it.  Those older values are not part of
        # the text-showing operation.
        self.show_text_operand(operands[-1])

    def resolve_color_space(self, name_obj: Any) -> tuple[str, ImageColorSpec | None]:
        """Resolve a cs/CS resource once for both its name and conversion spec."""
        name = self.document.resolver.resolve_name(name_obj)
        if name is None:
            return "DeviceGray", None
        value = self.document.resolver.deep_resolve(self.lookup_page_resource("ColorSpace", name))
        if value is None:
            # An inline device space (`/DeviceRGB cs`) names no resource.
            value = name
        base = value[0] if isinstance(value, (list, tuple)) and value else value
        color_space = normalize_pdf_name(base) or name
        try:
            spec = color_spec_from_value(value)
        except (ValueError, TypeError) as error:
            self.handle_operand_error(error, "color-space")
            spec = None
        return color_space, spec

    def resolve_pattern_color(self, operands: tuple[Any, ...]) -> PatternPaint | None:
        if not operands:
            return None
        pattern_name = self.document.resolver.resolve_name(operands[-1])
        if not pattern_name:
            return None
        pattern = self.document.resolver.resolve(self.lookup_page_resource("Pattern", pattern_name))
        pattern_dict: PdfDict | None
        if isinstance(pattern, PdfStream):
            pattern_dict = cast(PdfDict, pattern.dictionary)
        else:
            pattern_dict = (
                self.document.resolver.resolve_dict(pattern) if pattern is not None else None
            )
        if not isinstance(pattern_dict, dict):
            return None
        pattern_type = self.document.resolver.resolve_int(pattern_dict.get("PatternType"))
        if pattern_type == 2:
            shading: object = pattern_dict.get("Shading")
            shading = self.document.resolver.resolve(shading)
            shading_dict = (
                self.document.resolver.resolve_dict(shading) if shading is not None else None
            )
            if not isinstance(shading_dict, dict):
                return None
            return ShadingPattern(dict(shading_dict))
        if pattern_type != 1 or not isinstance(pattern, PdfStream):
            return None
        paint_type = self.document.resolver.resolve_int(pattern_dict.get("PaintType"), 1)
        if paint_type not in {1, 2}:
            return None
        base_color = None
        if paint_type == 2:
            base_color = self.normalize_color_operands(operands[:-1])
            if base_color is None:
                return None
        bbox = self.document.resolver.resolve_box(pattern_dict.get("BBox"))
        if bbox is None:
            return None
        x_step = self.document.resolver.resolve_float(pattern_dict.get("XStep"), default=None)
        y_step = self.document.resolver.resolve_float(pattern_dict.get("YStep"), default=None)
        if x_step is None or y_step is None or x_step == 0.0 or y_step == 0.0:
            return None
        matrix = self.matrix_operand(
            self.document.resolver.deep_resolve(pattern_dict.get("Matrix")), "pattern"
        )
        resources = self.resolve_resources(pattern_dict.get("Resources")) or {}
        return TilingPattern(
            bbox=bbox,
            x_step=float(x_step),
            y_step=float(y_step),
            stream=pattern,
            resources=resources,
            matrix=matrix,
            paint_type=paint_type,
            base_color=base_color,
        )

    def as_floats(self, operands: ContentOperands, count: int) -> tuple[float, ...] | None:
        """The leading `count` numeric operands, with errors delegated to the consumer."""
        if len(operands) < count:
            self.handle_operand_error(PdfParseError("missing numeric operand"), "numeric-operands")
            return None
        try:
            return tuple([self.as_float(operands[i]) for i in range(count)])
        except (TypeError, ValueError) as error:
            self.handle_operand_error(error, "numeric-operands")
            return None

    def as_int_operand(self, operands: ContentOperands) -> int | None:
        """The first integer operand, with errors delegated to the consumer."""
        if not operands:
            self.handle_operand_error(PdfParseError("missing numeric operand"), "integer-operand")
            return None
        try:
            return self.as_int(operands[0])
        except (TypeError, ValueError) as error:
            self.handle_operand_error(error, "integer-operand")
            return None

    def op_Q(self, operands: ContentOperands, depth: int) -> None:
        if len(self.stack) <= self.graphics_stack_floor:
            return
        self.restore_graphics_state(self.pop_graphics_save())

    def op_EX(self, operands: ContentOperands, depth: int) -> None:
        self.compatibility_depth = max(0, self.compatibility_depth - 1)

    def op_l(self, operands: ContentOperands, depth: int) -> None:
        if self.current_point is None or (values := self.as_floats(operands, 2)) is None:
            return
        x, y = values
        self.current_path.line_to(x, y)
        self.current_point = (x, y)

    def op_v(self, operands: ContentOperands, depth: int) -> None:
        if self.current_point is None or (values := self.as_floats(operands, 4)) is None:
            return
        x0, y0 = self.current_point
        x2, y2, x3, y3 = values
        self.append_cubic_curve(x0, y0, x2, y2, x3, y3)

    def op_y(self, operands: ContentOperands, depth: int) -> None:
        if self.current_point is None or (values := self.as_floats(operands, 4)) is None:
            return
        x1, y1, x3, y3 = values
        # `y` doubles the endpoint as the second control point, unlike `v`,
        # which uses the current point as the first one.
        self.append_cubic_curve(x1, y1, x3, y3, x3, y3)

    def append_cubic_curve(
        self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float
    ) -> None:
        if self.current_point is None:
            self.current_point = (x3, y3)
            return
        x0, y0 = self.current_point
        self.current_path.cubic_to((x0, y0, x1, y1, x2, y2, x3, y3), self.ctm, float(self.flatness))
        self.current_point = (x3, y3)

    def op_d(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) < 2:
            return
        try:
            phase = self.as_float(operands[1])
            array_obj = operands[0]
            dash_array = (
                [self.as_float(value) for value in array_obj]
                if isinstance(array_obj, (list, tuple))
                else []
            )
        except (TypeError, ValueError) as error:
            self.handle_operand_error(error, "dash-pattern")
            return
        self.dash_pattern = (dash_array, phase)

    def op_w(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is not None:
            self.line_width = max(0.0, values[0])

    def op_M(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is not None:
            self.miter_limit = max(1.0, values[0])

    def op_Tf(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) < 2:
            return
        font_operand = operands[0]
        font_size_operand = operands[1]
        decoder_matches_resources = self.current_decoder_resources_id == self.resources_id
        if (
            self.current_decoder is not None
            and decoder_matches_resources
            and font_operand is self.font_operand
        ):
            if font_size_operand is not self.font_size_operand:
                try:
                    font_size = self.as_float(font_size_operand)
                except (TypeError, ValueError) as error:
                    self.handle_operand_error(error, "font-size")
                    return
                if self.font_size != font_size:
                    self.font_size = font_size
                    self.update_text_scales()
                    self.update_font_metrics()
                self.font_size_operand = font_size_operand
            return
        font_name = self.document.resolver.resolve_name(font_operand)
        if font_name is None:
            return
        try:
            font_size = self.as_float(font_size_operand)
        except (TypeError, ValueError) as error:
            self.handle_operand_error(error, "font-size")
            return
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

    def op_gs(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        name = self.document.resolver.resolve_name(operands[0])
        if not name:
            return
        extgstate = self.resolve_extgstate(name)
        if not extgstate:
            return
        try:
            fill_opacity = extgstate.get("ca")
            if fill_opacity is not None:
                self.fill_opacity = max(0.0, min(1.0, self.as_float(fill_opacity)))
            stroke_opacity = extgstate.get("CA")
            if stroke_opacity is not None:
                self.stroke_opacity = max(0.0, min(1.0, self.as_float(stroke_opacity)))
            blend_mode = extgstate.get("BM")
            if blend_mode is not None:
                if isinstance(blend_mode, (list, tuple)):
                    blend_mode = blend_mode[0] if blend_mode else None
                if blend_mode is not None:
                    self.blend_mode = self.named_value(blend_mode)
        except (TypeError, ValueError) as error:
            self.handle_operand_error(error, "extended-graphics-state")
            return

    def op_sh(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        name = self.document.resolver.resolve_name(operands[0])
        if not name:
            return
        shading = self.document.resolver.resolve_dict(self.lookup_page_resource("Shading", name))
        if isinstance(shading, dict):
            self.sink.paint_shading(self, shading)

    def op_EMC(self, operands: ContentOperands, depth: int) -> None:
        if self.marked_content_stack:
            self.sink.end_marked_content(self, self.marked_content_stack.pop())
            self.sink.text_boundary(self, "marked")

    def normalize_colors(self, *components: Any) -> tuple[float, ...] | None:
        values: list[float] = []
        for component in components:
            try:
                values.append(max(0.0, min(1.0, self.as_float(component))))
            except ValueError as error:
                self.handle_operand_error(error, "color-components")
                return None
        if not values:
            return None
        return tuple(values)

    def resolve_resources(self, value: object) -> PdfDict | None:
        return recover_resources(value, self.document.resolver)

    def handle_operand_error(self, error: Exception, context: str) -> None:
        pass

    def matrix_operand(self, value: object, context: str) -> Matrix:
        if value is None:
            return IDENTITY_MATRIX
        try:
            return Matrix.from_operand(value)
        except ValueError:
            if context == "form" and isinstance(value, (list, tuple)) and len(value) > 6:
                return Matrix.from_operand(value[:6])
            if context == "pattern":
                return IDENTITY_MATRIX
            raise
