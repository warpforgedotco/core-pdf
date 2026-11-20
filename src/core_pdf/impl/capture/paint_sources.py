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
from core_pdf.impl.graphics.overprint import (
    SPOT,
    overprint_marks,
    process_color,
    raw_overprint_marks,
)
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
    ancestors = self.paint_ancestors
    if ancestors is not None and spec.kind in ancestors.spaces:
        # A device colour set before the group it paints in is that group's CIE-based
        # colour all the same (11.7.2); one set inside it was remapped as it was set.
        spec, color = ancestors.remap(spec, color)
    if not color_space_paints(spec):
        return color
    cmyk = self.capture_cmyk
    profile = self.capture_cmyk_profile
    key = (id(spec), color, intent, black_point, cmyk, profile)
    caches = self.caches
    previous = caches.capture_colors.get(key)
    if previous is not None:
        result = previous[1]
    else:
        converted: tuple[float, ...] | None = None
        if cmyk and spec.kind not in GRAY_RGB_KINDS:
            # A CMYK blending space takes a colour that comes down to CMYK as that CMYK, of
            # its own profile.
            converted = color_operands_to_cmyk(
                spec, color, profile, rendering=graphics.color_rendering
            )
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


def capture_overprint(self: TextState, *, stroke: bool) -> int | None:
    """The process components a paint with the current stroking or nonstroking colour
    marks while that overprint parameter is on (graphics.overprint.overprint_marks), or None
    for all four, as when it is off."""
    graphics = self.graphics
    if not (graphics.stroke_overprint if stroke else graphics.fill_overprint):
        return None
    marks = overprint_marks(
        graphics.stroke_space if stroke else graphics.fill_space,
        graphics.stroke_color if stroke else graphics.fill_color,
        graphics.overprint_mode,
        current_color=True,
    )
    if marks is not None:
        self.uses_overprint = True
    return marks


def overprint_color(
    self: TextState, captured: tuple[float, ...] | None, marks: int | None, *, stroke: bool
) -> tuple[float, ...] | None:
    """The colour an overprinting paint with marks takes: in a CMYK blending space, a
    Separation or DeviceN colour naming process colorants alone paints their tints directly
    (graphics.overprint.process_color), not its alternate's conversion; any other paint keeps
    captured."""
    if marks is None or marks & SPOT or not self.capture_cmyk:
        return captured
    graphics = self.graphics
    space = graphics.stroke_space if stroke else graphics.fill_space
    color = graphics.stroke_color if stroke else graphics.fill_color
    if space is None or color is None:
        return captured
    direct = process_color(space, color)
    return captured if direct is None else direct


def capture_space_overprint(self: TextState, space: object) -> int | None:
    """capture_overprint for an image or a shading painted in space, a resolved ColorSpace
    entry, which the nonstroking overprint parameter governs."""
    if not self.graphics.fill_overprint:
        return None
    marks = raw_overprint_marks(space)
    if marks is not None:
        self.uses_overprint = True
    return marks


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
        # A tiling cell's colours are captured for the blending space it paints into, and
        # the groups it paints in (11.7.2); a shading pattern's do not depend on them.
        (self.capture_cmyk, self.capture_cmyk_profile, id(self.ancestor_spaces))
        if isinstance(pattern, model.TilingPattern)
        else None,
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
        nested.set_ancestor_spaces(self.ancestor_spaces)
        nested.set_capture_cmyk(self.capture_cmyk, self.capture_cmyk_profile)
        try:
            result = self.capture_tiling_pattern(
                nested, pattern, rendering, initial_alpha_is_shape, initial_text_knockout
            )
        finally:
            nested.release()
    pins = tuple(spaces for spaces in (defaults, self.ancestor_spaces) if spaces is not None)
    return self.capture_patterns.put_key(pattern, cache_key, result, pins)
