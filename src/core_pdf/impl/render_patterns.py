# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from core_pdf.impl.capture_records import (
    CapturedPath,
    TilingPattern,
)
from core_pdf.impl.geometry import rect_tuple
from core_pdf.impl.graphics_device_profiles import cmyk_floats_to_srgb
from core_pdf.impl.render_blend import (
    clamp01,
    color_component,
)
from core_pdf.impl.render_commands import append_captured_program
from core_pdf.impl.render_display import DisplayList
from core_pdf.impl.render_model import DisplayItem, ImagePaintItem, PathPaintItem
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering

if TYPE_CHECKING:
    from core_pdf.impl.render_target import RasterTarget

TilingCellCache = dict[tuple[int, bool], tuple[TilingPattern, DisplayList, CapturedPath]]


def tiling_cell(target: RasterTarget, pattern: TilingPattern) -> tuple[DisplayList, CapturedPath]:
    preserve_object_boundaries = target.group_source_shape is not None
    key = (id(pattern), preserve_object_boundaries)
    cached = target.tiling_cell_cache.get(key)
    if cached is not None and cached[0] is pattern:
        return cached[1], cached[2]
    cell_x0, cell_y0, cell_x1, cell_y1 = pattern.bbox
    display = DisplayList(
        target.width,
        target.height,
        preserve_object_boundaries=preserve_object_boundaries,
    )
    cell_clip = CapturedPath()
    cell_clip.rect(cell_x0, cell_y0, cell_x1 - cell_x0, cell_y1 - cell_y0)
    append_captured_program(display, pattern.program, include_text=True)
    target.tiling_cell_cache[key] = (pattern, display, cell_clip)
    return display, cell_clip


# Display items that change what later items paint into, but paint nothing.
NON_PAINTING_KINDS = frozenset(
    {"scope-begin", "scope-end", "state-push", "state-pop", "clip", "group-begin", "group-end"}
)


def cell_paints_nothing(
    items: Iterable[DisplayItem], cell_clip: CapturedPath, scale: float
) -> bool:
    """Whether a tiling cell's items all lie outside the cell clip they are painted under.

    A tile paints its items translated, under the cell clip translated by the
    same amount, so whether anything lands inside the clip does not depend on
    which tile it is. Conservative: an item whose extent is not known -- a
    shading, an unknown kind, a missing box -- answers no. Strokes are widened
    by ten line widths for miter joins, and every box by two device pixels,
    so a partly covered pixel at either edge cannot slip through.
    """
    clip_box = cell_clip.bbox()
    if clip_box is None:
        return False
    margin = 2.0 / scale if scale > 0 else 2.0
    left, bottom, right, top = clip_box
    left -= margin
    bottom -= margin
    right += margin
    top += margin
    for item in items:
        if isinstance(item, PathPaintItem):
            path = item.path
            box = item.bbox
            if box is None and type(path) is CapturedPath:
                box = path.bbox()
            spread = 10.0 * abs(float(item.line_width or 0.0))
        elif isinstance(item, ImagePaintItem):
            box = item.bbox
            spread = 0.0
        elif item.kind in NON_PAINTING_KINDS:
            continue
        elif item.kind == "glyph":
            box = item.data.get("bbox")
            spread = 0.0
        else:
            return False
        extent = rect_tuple(box)
        if extent is None:
            return False
        pad = spread + margin
        if (
            extent[0] - pad <= right
            and extent[2] + pad >= left
            and extent[1] - pad <= top
            and extent[3] + pad >= bottom
        ):
            return False
    return True


def shading_color_rgba(
    color_model: str,
    components: list[float] | tuple[float, ...],
    opacity: Any,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> tuple[int, int, int, int]:
    alpha = color_component(opacity, 255) if type(opacity) in {int, float} else 255
    name = color_model or "DeviceRGB"
    if name.endswith("DeviceGray") or len(components) == 1:
        gray = color_component(components[0] if components else 0.0)
        return gray, gray, gray, alpha
    if name.endswith("DeviceCMYK") and len(components) >= 4:
        c, m, y, k = (clamp01(v) for v in components[:4])
        red, green, blue = cmyk_floats_to_srgb(c, m, y, k, rendering=rendering)
        return red, green, blue, alpha
    rgb = [color_component(c) for c in components[:3]]
    while len(rgb) < 3:
        rgb.append(rgb[-1] if rgb else 0)
    return rgb[0], rgb[1], rgb[2], alpha


def tiling_pattern_uses_normal_blends(
    pattern: TilingPattern, active: set[int] | None = None
) -> bool:
    if active is None:
        active = set()
    identity = id(pattern)
    if identity in active:
        return False
    active.add(identity)
    try:
        program = pattern.program
        modes = [drawing.blend_mode for drawing in program.drawings]
        modes.extend(glyph.blend_mode for glyph in program.glyphs)
        modes.extend(image.blend_mode for image in program.inline_images)
        if any(mode is not None and mode.casefold() != "normal" for mode in modes):
            return False
        for drawing in program.drawings:
            for nested in (drawing.fill_pattern, drawing.stroke_pattern):
                if isinstance(nested, TilingPattern) and not tiling_pattern_uses_normal_blends(
                    nested, active
                ):
                    return False
        return True
    finally:
        active.remove(identity)
