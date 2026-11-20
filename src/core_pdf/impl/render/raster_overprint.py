# SPDX-License-Identifier: AGPL-3.0-only

"""RasterTarget methods: overprinting paint in a DeviceCMYK blending space (ISO 32000-2
8.6.7, 11.7.4).

A paint captured while its overprint parameter was on carries the process components it
marks (graphics.overprint). Where it lands in a DeviceCMYK group -- each of whose two passes
holds three additive components, the complements of cyan, magenta and yellow or of black
(raster_spaces) -- a pass whose components it marks all paints it as any other. In a pass
with a component it does not mark, it paints alone into a transparent layer of its own,
which gives its colour and alpha at each pixel, soft mask, opacity and clip included; the
layer then composites onto the group as 11.7.4.3 describes: the special overprinting blend
mode B(cb, cs) takes cs for a marked component and, for one it does not mark, cb, or for
a paint whose space names a spot colorant the backdrop's tint with the paint's own (its
alternate colour's) added, clamped at full; and when the current blend mode is not Normal,
the result is blended with it, as if the paint were a non-isolated, non-knockout group of
its own.

In an RGB or gray space, the process components are not CMYK, and every paint marks them
all (Table 146), so overprinting changes nothing and the paint takes the usual path.
"""

from __future__ import annotations

from copy import replace
from typing import TYPE_CHECKING, Any, cast

lazy import numpy

from core_pdf.impl.base.arrays import UInt8Array, uint8_image_view
from core_pdf.impl.graphics.overprint import BLACK, CYAN, MAGENTA, SPOT, YELLOW
from core_pdf.impl.render.blend import BLEND_OP_MODES, BlendOp, blend_context
from core_pdf.impl.render.model import (
    DisplayItem,
    GlyphBitmapItem,
    ImagePaintItem,
    PaintItemBase,
    PathPaintItem,
    PathPaintKind,
    RasterGroup,
    ShadingItem,
    SpaceMode,
)
from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_11_transparency.blend import blend_colors
from core_pdf.impl.spec.standards import SemanticContext

if TYPE_CHECKING:
    from core_pdf.impl.render.target import RasterTarget

# Which of a pass's three components a paint marks, and whether its spot colorants add to
# the others.
type Channels = tuple[bool, bool, bool, bool]


class OverprintScratch:
    """The layer an overprinting paint is painted into, kept for the next one at the same
    group depth, and the window its last paint touched, which the next clears."""

    __slots__ = ("buffer", "view", "source_shape", "dirty")

    def __init__(self, height: int, width: int) -> None:
        self.buffer = bytearray(height * width * 4)
        self.view = uint8_image_view(self.buffer, (height, width, 4))
        self.source_shape: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None = None
        self.dirty: RasterGroup | None = None


def paint_marks(item: PaintItemBase) -> int | None:
    """The process components a paint item marks while overprinting, or None for all four:
    its fill's, or a stroke's."""
    if type(item) is PathPaintItem:
        if item.paint_kind is PathPaintKind.STROKE:
            return item.stroke_overprint
        return item.fill_overprint
    if type(item) is ImagePaintItem:
        return item.fill_overprint
    if type(item) is GlyphBitmapItem or type(item) is ShadingItem:
        return item.payload.get("fill_overprint")
    return None


def overprint_channels(self: RasterTarget, item: PaintItemBase) -> Channels | None:
    """Which components of the current pass item marks, when it overprints one of them;
    None when it marks them all and paints as usual. A fill and stroke paint apart first
    (raster_dispatch.paint_item)."""
    return marks_channels(self.space_mode, paint_marks(item))


def marks_channels(mode: SpaceMode, marks: int | None) -> Channels | None:
    """The Channels of a paint with marks in a pass of mode, or None when it marks all of
    the pass's components."""
    if marks is None:
        return None
    spot = bool(marks & SPOT)
    if mode is SpaceMode.CMY:
        cyan, magenta, yellow = bool(marks & CYAN), bool(marks & MAGENTA), bool(marks & YELLOW)
        if cyan and magenta and yellow:
            return None
        return cyan, magenta, yellow, spot
    if marks & BLACK:
        return None
    return False, False, False, spot


def fill_stroke_channels(
    self: RasterTarget, item: PathPaintItem
) -> tuple[Channels | None, Channels | None]:
    """overprint_channels of a fill and stroke's fill and of its stroke."""
    return (
        marks_channels(self.space_mode, item.fill_overprint),
        marks_channels(self.space_mode, item.stroke_overprint),
    )


def paint_fill_stroke_apart(
    self: RasterTarget, item: PathPaintItem, fill: Channels | None, stroke: Channels | None
) -> None:
    """Paint a fill and stroke whose fill or stroke overprints in the current pass as its
    fill and then its stroke, each overprinting as its channels say."""
    for kind, channels in ((PathPaintKind.FILL, fill), (PathPaintKind.STROKE, stroke)):
        part = replace(item, paint_kind=kind)
        if channels is None:
            self.paint_display_item(part)
        else:
            self.paint_overprinting(part, channels)


def push_overprint_layer(self: RasterTarget, channels: Channels, blend_mode: str | None) -> None:
    """Push the transparent layer one overprinting paint is painted into: isolated, so the
    paint lands unblended, and tracking shape when its parent does."""
    parent = self.buffer_stack[-1]
    depth = len(self.buffer_stack)
    scratch = self.overprint_scratch.get(depth)
    if scratch is None:
        scratch = self.overprint_scratch[depth] = OverprintScratch(self.height, self.width)
    elif scratch.dirty is not None:
        window = scratch.dirty.paint_window.slices()
        if window is not None:
            scratch.view[window] = 0
            if scratch.source_shape is not None:
                scratch.source_shape[window] = 0.0
    shape = None
    if parent.source_shape is not None:
        if scratch.source_shape is None:
            scratch.source_shape = numpy.zeros((self.height, self.width), dtype=numpy.float32)
        shape = scratch.source_shape
    layer = RasterGroup(
        scratch.buffer, None, blend_mode, view=scratch.view, source_shape=shape, overprint=channels
    )
    scratch.dirty = layer
    self.buffer_stack.append(layer)
    self.sync_group_mirrors()


def paint_overprinting(self: RasterTarget, item: PaintItemBase, channels: Channels) -> None:
    """Paint item, which overprints components of the current pass, through its layer."""
    self.push_overprint_layer(channels, item.blend_mode)
    try:
        self.paint_display_item(cast("DisplayItem", item))
    finally:
        self.composite_group(self.pop_group())


@cites(
    "ISO 32000-2",
    "11.7.4.3",
    introduced="1.4",
    rule="an overprinting paint blends by a mode taking the source for the components it "
    "marks and the backdrop for the rest, then by the current blend mode",
)
def composite_overprint_layer(
    destination: UInt8Array,
    layer: UInt8Array,
    channels: Channels,
    mode: BlendOp | None,
    *,
    semantic_context: SemanticContext,
) -> UInt8Array:
    """Composite layer, one overprinting paint's colour and alpha, onto destination, and
    return the alpha it composited with.

    For each component of the pass the special blend mode gives B(cb, cs) = cs where the
    paint marks it, and otherwise the backdrop's tint plus the paint's, clamped:
    max(cb + cs - 1, 0) on the additive components, which is cb where the paint's tint is 0.
    A blend mode other than Normal then blends the backdrop with that, as the implicit
    group 11.7.4.3 puts the paint in composites with it. The result composites as any
    blend does (11.3.6): cs (1 - ab) + ab B, weighted by the paint's alpha.
    """
    alpha = layer[..., 3]
    visible = alpha > 0
    if not numpy.any(visible):
        return alpha.copy()
    source = layer[visible][:, :3].astype(numpy.float64) / 255.0
    backdrop = destination[visible].astype(numpy.float64)
    below = backdrop[:, :3] / 255.0
    below_alpha = backdrop[:, 3] / 255.0
    source_alpha = alpha[visible].astype(numpy.float64) / 255.0
    unmarked = numpy.maximum(below + source - 1.0, 0.0) if channels[3] else below
    special = numpy.where(numpy.array(channels[:3]), source, unmarked)
    if mode is BlendOp.BACKDROP:
        blended = below
    elif mode is None or mode in {BlendOp.NORMAL, BlendOp.SOURCE, BlendOp.UNSUPPORTED}:
        blended = special
    else:
        blended = numpy.column_stack(
            blend_colors(
                (below[:, 0], below[:, 1], below[:, 2]),
                (special[:, 0], special[:, 1], special[:, 2]),
                BLEND_OP_MODES[mode],
                context=blend_context(semantic_context),
            )
        )
    below_weight = below_alpha[:, None]
    mixed = source * (1.0 - below_weight) + blended * below_weight
    result_alpha = source_alpha + below_alpha * (1.0 - source_alpha)
    color = (
        mixed * (source_alpha * 255.0)[:, None]
        + backdrop[:, :3] * (below_alpha * (1.0 - source_alpha))[:, None]
    ) / result_alpha[:, None]
    composited = numpy.empty((len(source), 4), dtype=numpy.float64)
    composited[:, :3] = color
    composited[:, 3] = result_alpha * 255.0
    numpy.rint(composited, out=composited)
    numpy.clip(composited, 0.0, 255.0, out=composited)
    destination[visible] = composited.astype(numpy.uint8)
    return alpha.copy()
