# SPDX-License-Identifier: AGPL-3.0-only

"""TextState methods: the colors and patterns paint takes.

Each function takes the state as self; TextState lists them as its methods.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from core_pdf.impl.base.types import MISSING, MissingObject
from core_pdf.impl.capture.records import PatternPaint, ShadingPattern
from core_pdf.impl.graphics.color import color_operands_to_cmyk, color_operands_to_srgb
from core_pdf.impl.graphics.color_spec import GRAY_RGB_KINDS
from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_07_content import model
from core_pdf.impl.spec.s_08_graphics.color import color_space_paints
from core_pdf.impl.spec.s_08_graphics.color_rendering import override_color_rendering
from core_pdf.impl.spec.s_08_graphics.default_spaces import remap_color_space
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX

if TYPE_CHECKING:
    from core_pdf.impl.capture.recording import TextState


def initial_pattern(self: TextState, *, stroke: bool) -> bool:
    space = self.graphics.stroke_space if stroke else self.graphics.fill_space
    pattern = self.graphics.stroke_pattern if stroke else self.graphics.fill_pattern
    return space.kind == "Pattern" and pattern is None


def capture_color(self: TextState, *, stroke: bool) -> tuple[float, ...] | None:
    graphics = self.graphics
    if stroke:
        color = graphics.stroke_color
        spec = graphics.stroke_space
    else:
        color = graphics.fill_color
        spec = graphics.fill_space
    if color is None or spec is None:
        return color
    intent = graphics.render_intent
    black_point = graphics.black_point_compensation
    # Consecutive paints usually reuse the very same color and space objects.
    last = self.last_stroke_color if stroke else self.last_fill_color
    if (
        last is not None
        and last[0] is spec
        and last[1] is color
        and last[2] == intent
        and last[3] == black_point
    ):
        return last[4]
    if not color_space_paints(spec):
        return color
    cmyk = self.capture_cmyk
    key = (id(spec), color, intent, black_point, cmyk)
    caches = self.caches
    previous = caches.capture_colors.get(key)
    if previous is not None:
        result = previous[1]
    else:
        converted: tuple[float, ...] | None = None
        if cmyk and spec.kind not in GRAY_RGB_KINDS:
            # A CMYK blending space takes a colour that comes down to CMYK as that CMYK.
            converted = color_operands_to_cmyk(spec, color)
        if converted is None:
            converted = color_operands_to_srgb(
                spec, list(color), rendering=graphics.color_rendering
            )
        result = converted if converted is not None else color
        caches.capture_colors.put(key, (spec, result))
    memo = (spec, color, intent, black_point, result)
    if stroke:
        self.last_stroke_color = memo
    else:
        self.last_fill_color = memo
    return result


# The entries of a shading dictionary the renderer reads, resolved when it is captured.
SHADING_RESOLVED_KEYS = frozenset(
    {
        "ShadingType",
        "ColorSpace",
        "Function",
        "Coords",
        "Domain",
        "Extend",
        "BBox",
        "Matrix",
        "Background",
        "Decode",
        "BitsPerCoordinate",
        "BitsPerComponent",
        "BitsPerFlag",
        "VerticesPerRow",
    }
)


@cites(
    "ISO 32000-2",
    "8.6.5.6",
    introduced="1.1",
    rule="a shading's device ColorSpace remaps through the current default spaces",
)
def capture_shading_dictionary(self: TextState, dictionary: dict) -> dict:
    """A shading dictionary with the entries the renderer reads resolved, and its ColorSpace
    remapped through the default spaces of the current resource dictionary."""
    shadings = self.caches.capture_shadings
    defaults = self.current_default_spaces()
    pins = () if defaults is None else (defaults,)
    cached = shadings.get(dictionary, pins=pins)
    if cached is not None:
        return cached
    captured = {
        key: self.resolver.deep_resolve(value) if str(key) in SHADING_RESOLVED_KEYS else value
        for key, value in dictionary.items()
    }
    if defaults is not None:
        for key, value in captured.items():
            if str(key) == "ColorSpace":
                captured[key] = remap_color_space(value, defaults.values)
                break
    return shadings.put(dictionary, captured, pins=pins)


def capture_pattern(self: TextState, pattern: object) -> PatternPaint | None:
    if pattern is None:
        return None
    rendering = self.graphics.color_rendering
    initial_alpha_is_shape = (
        pattern.alpha_is_shape if isinstance(pattern, model.TilingPattern) else False
    )
    initial_text_knockout = (
        pattern.text_knockout if isinstance(pattern, model.TilingPattern) else True
    )
    # A shading pattern's colour space remaps through the default spaces where it paints.
    defaults = self.current_default_spaces() if isinstance(pattern, model.ShadingPattern) else None
    cache_key = (
        id(pattern),
        rendering,
        initial_alpha_is_shape,
        initial_text_knockout,
        id(defaults),
    )
    cached = self.capture_patterns.get_key(pattern, cache_key, MISSING)
    if not isinstance(cached, MissingObject):
        return cached
    result: PatternPaint | None = None
    if isinstance(pattern, model.ShadingPattern):
        if pattern.extgstate is not None:
            values = {
                str(key): self.resolver.resolve(value)
                for key, value in pattern.extgstate.items()
                if str(key) in {"RI", "UseBlackPtComp"}
            }
            with suppress(ValueError):
                rendering = override_color_rendering(values, rendering)
        matrix = pattern.matrix
        result = ShadingPattern(
            self.capture_shading_dictionary(pattern.dictionary),
            color_rendering=rendering,
            matrix=None if matrix == IDENTITY_MATRIX else matrix,
        )
    elif isinstance(pattern, model.TilingPattern):
        nested = self.nested_capture_state()
        try:
            result = self.capture_tiling_pattern(
                nested, pattern, rendering, initial_alpha_is_shape, initial_text_knockout
            )
        finally:
            nested.release()
    return self.capture_patterns.put_key(
        pattern, cache_key, result, () if defaults is None else (defaults,)
    )
