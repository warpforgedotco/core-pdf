# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from contextlib import suppress
from copy import copy
from dataclasses import replace
from math import hypot
from typing import TYPE_CHECKING

from core_pdf.impl.caches import MISSING
from core_pdf.impl.capture_host import CaptureHost
from core_pdf.impl.capture_program import CapturedProgram
from core_pdf.impl.capture_records import (
    CapturedSoftMask,
    PatternPaint,
    ShadingPattern,
    TilingPattern,
)
from core_pdf.impl.graphics_color import color_operands_to_srgb
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.model import GraphicsState
from core_pdf_spec.s_07_content.model import ShadingPattern as PdfShadingPattern
from core_pdf_spec.s_07_content.model import TilingPattern as PdfTilingPattern
from core_pdf_spec.s_08_graphics.color import color_space_paints
from core_pdf_spec.s_08_graphics.color_rendering import ColorRendering, override_color_rendering
from core_pdf_spec.s_11_transparency.soft_masks import SoftMask as PdfSoftMask

if TYPE_CHECKING:
    pass


GRAPHICS_STATE_FIELDS = GraphicsState.__fields__


MASK_OVERRIDDEN_FIELDS = frozenset(
    {"ctm", "soft_mask", "fill_opacity", "stroke_opacity", "blend_mode"}
)


MASK_KEYED_FIELDS = tuple(
    name for name in GRAPHICS_STATE_FIELDS if name not in MASK_OVERRIDDEN_FIELDS
)


def state_key(value: object) -> object:
    if isinstance(value, tuple):
        return tuple(state_key(part) for part in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value), value)
    return ("identity", id(value))


class PaintResolutionMixin(CaptureHost):
    __slots__ = ()

    def resolve_soft_mask(self, value: object) -> PdfSoftMask | None:
        mask = super().resolve_soft_mask(value)
        if mask is not None:
            scopes = self.caches.capture_mask_resources
            if scopes.get(mask, default=MISSING) is MISSING:
                scopes.put(mask, self.resources)
        return mask

    def initial_pattern(self, *, stroke: bool) -> bool:
        space = self.graphics.stroke_space if stroke else self.graphics.fill_space
        pattern = self.graphics.stroke_pattern if stroke else self.graphics.fill_pattern
        return space.kind == "Pattern" and pattern is None

    def text_paint_mode(self, *, check_colorants: bool = True) -> int:
        mode = self.graphics.render_mode
        if mode not in range(8):
            return mode
        fills = mode in {0, 2, 4, 6} and not self.initial_pattern(stroke=False)
        strokes = mode in {1, 2, 5, 6} and not self.initial_pattern(stroke=True)
        if check_colorants:
            fills = fills and color_space_paints(self.graphics.fill_space)
            strokes = strokes and color_space_paints(self.graphics.stroke_space)
        paint = 2 if fills and strokes else 0 if fills else 1 if strokes else 3
        return paint + (4 if mode >= 4 else 0)

    def capture_color(self, *, stroke: bool) -> tuple[float, ...] | None:
        graphics = self.graphics
        if stroke:
            color = graphics.stroke_color
            spec = graphics.stroke_space
        else:
            color = graphics.fill_color
            spec = graphics.fill_space
        if color is None or spec is None or not color_space_paints(spec):
            return color
        intent = graphics.render_intent
        black_point = graphics.black_point_compensation
        key = (id(spec), color, intent, black_point)
        caches = self.caches
        previous = caches.capture_colors.get(key)
        if previous is not None:
            return previous[1]
        converted = color_operands_to_srgb(spec, list(color), rendering=graphics.color_rendering)
        result = converted if converted is not None else color
        caches.capture_colors.put(key, (spec, result))
        return result

    def capture_shading_dictionary(self, dictionary: dict) -> dict:
        shadings = self.caches.capture_shadings
        cached = shadings.get(dictionary)
        if cached is not None:
            return cached
        captured = {
            key: self.resolver.deep_resolve(value)
            if str(key) in {"ColorSpace", "Function", "Coords", "Domain", "Extend", "BBox"}
            else value
            for key, value in dictionary.items()
        }
        return shadings.put(dictionary, captured)

    def capture_pattern(self, pattern: object) -> PatternPaint | None:
        if pattern is None:
            return None
        rendering = self.graphics.color_rendering
        initial_alpha_is_shape = (
            pattern.alpha_is_shape if isinstance(pattern, PdfTilingPattern) else False
        )
        initial_text_knockout = (
            pattern.text_knockout if isinstance(pattern, PdfTilingPattern) else True
        )
        key = (rendering, initial_alpha_is_shape, initial_text_knockout)
        cached = self.capture_patterns.get(pattern, *key, default=MISSING)
        if cached is not MISSING:
            return cached
        result: PatternPaint | None = None
        if isinstance(pattern, PdfShadingPattern):
            if pattern.extgstate is not None:
                values = {
                    str(key): self.resolver.resolve(value)
                    for key, value in pattern.extgstate.items()
                    if str(key) in {"RI", "UseBlackPtComp"}
                }
                with suppress(ValueError):
                    rendering = override_color_rendering(values, rendering)
            result = ShadingPattern(
                self.capture_shading_dictionary(pattern.dictionary), color_rendering=rendering
            )
        elif isinstance(pattern, PdfTilingPattern):
            nested = self.nested_capture_state()
            try:
                result = self.capture_tiling_pattern(
                    nested, pattern, rendering, initial_alpha_is_shape, initial_text_knockout
                )
            finally:
                nested.release()
        return self.capture_patterns.put(pattern, result, *key)

    def capture_tiling_pattern(
        self,
        nested: CaptureHost,
        pattern: PdfTilingPattern,
        rendering: ColorRendering,
        initial_alpha_is_shape: bool,
        initial_text_knockout: bool,
    ) -> TilingPattern | None:
        nested.graphics.render_intent = self.graphics.render_intent
        nested.graphics.black_point_compensation = self.graphics.black_point_compensation
        nested.graphics.alpha_is_shape = initial_alpha_is_shape
        nested.graphics.text_knockout = initial_text_knockout
        try:
            nested.stream_executor.consume(pattern.stream, pattern.resources, pattern.matrix, 0)
        except Exception:
            return None
        if pattern.paint_type == 2:
            base_color = pattern.base_color
            if base_color is not None and pattern.base_color_spec is not None:
                converted = color_operands_to_srgb(
                    pattern.base_color_spec, base_color, rendering=rendering
                )
                if converted is not None:
                    base_color = converted
            for drawing in nested.drawings:
                if drawing.kind in {"fill", "fillstroke"}:
                    drawing.fill = base_color
                if drawing.kind in {"stroke", "fillstroke"}:
                    drawing.stroke_color = base_color
            for glyph in nested.glyphs:
                glyph.style = replace(glyph.style, fill=base_color, stroke_color=base_color)
        return TilingPattern(
            pattern.bbox,
            pattern.x_step,
            pattern.y_step,
            CapturedProgram(
                glyphs=tuple(glyph for glyph in nested.glyphs if glyph.has_paint),
                drawings=tuple(nested.drawings),
                inline_images=tuple(nested.inline_images),
                text_boundaries=tuple(nested.text_boundaries),
                options=nested.options,
            ),
        )

    def capture_graphics_soft_mask(self) -> CapturedSoftMask | None:
        mask = self.graphics.soft_mask
        if mask is None or not self.options.render_details:
            return None
        key = tuple(state_key(getattr(self.graphics, name)) for name in MASK_KEYED_FIELDS)
        caches = self.caches
        cached = caches.capture_soft_masks.get(mask, key, default=MISSING)
        if cached is not MISSING:
            return cached
        graphics = copy(self.graphics)
        graphics.ctm = mask.ctm
        graphics.soft_mask = None
        graphics.fill_opacity = graphics.stroke_opacity = 1.0
        graphics.blend_mode = None
        caches.capture_soft_masks.put(mask, None, key)
        group_key = id(mask.group)
        if mask.subtype != "Alpha" or group_key in caches.capture_active_mask_groups:
            return None
        if len(caches.capture_active_mask_groups) >= 10:
            return None
        caches.capture_active_mask_groups.add(group_key)
        nested: CaptureHost | None = None
        try:
            nested = self.nested_capture_state()
            nested.graphics = copy(graphics)
            scope = caches.capture_mask_resources.get(mask, default=MISSING)
            nested.resources = self.resources if scope is MISSING else scope
            frame = nested.append_form_xobject(mask.group, 0)
            if frame is None:
                return None
            nested.stream_executor.consume_frame(frame)
            nested.run_accumulator.flush()
            if not nested.text_boundaries:
                return None
            return caches.capture_soft_masks.put(
                mask, CapturedSoftMask(nested.captured_program(), mask.transfer), key
            )
        except PdfParseError, TypeError, ValueError, ArithmeticError:
            return None
        finally:
            caches.capture_active_mask_groups.remove(group_key)
            if nested is not None:
                nested.release()

    def graphics_scale(self) -> float:
        ctm = self.graphics.ctm
        cached = self.scale_cache
        if cached is not None and cached[0] is ctm:
            return cached[1]
        x_scale = hypot(ctm.a, ctm.b)
        y_scale = hypot(ctm.c, ctm.d)
        if x_scale == 0 and y_scale == 0:
            scale = 1.0
        elif x_scale == 0:
            scale = y_scale
        elif y_scale == 0:
            scale = x_scale
        else:
            scale = (x_scale + y_scale) * 0.5
        self.scale_cache = (ctm, scale)
        return scale

    def transformed_line_width(self) -> float:
        line_width = max(0.0, self.graphics.line_width)
        if line_width == 0:
            return 0.0
        return line_width * self.graphics_scale()

    def transformed_dash_pattern(self) -> tuple[list[float], float] | None:
        dash_pattern = self.graphics.dash_pattern
        if not dash_pattern:
            return None
        dash_array, phase = dash_pattern
        scale = self.graphics_scale()
        return [max(0.0, float(value) * scale) for value in dash_array], float(phase) * scale


__all__ = (
    "GRAPHICS_STATE_FIELDS",
    "MASK_KEYED_FIELDS",
    "MASK_OVERRIDDEN_FIELDS",
    "PaintResolutionMixin",
    "state_key",
)
