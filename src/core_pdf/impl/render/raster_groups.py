# SPDX-License-Identifier: AGPL-3.0-only

"""RasterTarget methods: transparency groups and scopes: pushing, compositing and popping them.

Each function takes the target as self; RasterTarget lists them as its methods.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

lazy import numpy

from core_pdf.impl.base.arrays import UInt8Array, uint8_image_view
from core_pdf.impl.capture.records import CapturedPath
from core_pdf.impl.render.blend import BlendOp, composite_blended_group_numpy
from core_pdf.impl.render.model import PixelWindow, RasterGroup, SoftMaskPlane
from core_pdf.impl.render.raster_overprint import composite_overprint_layer
from core_pdf.impl.render.raster_spots import KnockoutElement
from core_pdf.impl.render.raster_support import (
    ElementaryScratch,
    PixelBox,
    composite_masked_group,
    composite_nonisolated_group,
)
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import is_pdf_number
lazy from core_pdf_cythonized import (
    composite_elementary_knockout,
    composite_knockout_group,
    composite_normal_group,
)

if TYPE_CHECKING:
    from core_pdf.impl.render.target import RasterTarget


def group_window(self: RasterTarget, group: RasterGroup) -> tuple[slice, slice] | None:
    return group.paint_window.slices()


def composite_group(self: RasterTarget, group: RasterGroup) -> None:
    parent = self.buffer_stack[-1]
    window = self.group_window(group)
    if window is None:
        return
    rows, columns = window
    if parent.painted_boxes is not None:
        self.record_knockout_paint((columns.start, rows.start, columns.stop, rows.stop))
    # A group that changes spot plates (raster_spots) composites them once its colour has.
    before = (
        parent.view[rows, columns, 3].copy()
        if self.involves_plates(parent, group, rows, columns)
        else None
    )
    shape = group.source_shape[rows, columns] if group.source_shape is not None else None
    if shape is not None and group.alpha_is_shape:
        shape = shape * group.source_scale
        if group.mask_alpha is not None:
            shape = shape * group.mask_alpha[rows, columns]
    if (
        parent.knockout
        and parent.backdrop is not None
        and group.backdrop is not None
        and group.mask_alpha is None
        and group.source_alpha is not None
        and group.source_scale == 1.0
        and self.resolved_blend(group.blend_mode, casefold=True) in {None, BlendOp.NORMAL}
    ):
        assert parent.source_alpha is not None
        assert shape is not None
        initial_view = self.pixel_view(parent.backdrop)[rows, columns]
        composite_elementary_knockout(
            parent.view[rows, columns],
            initial_view,
            group.view[rows, columns],
            group.source_alpha[rows, columns],
            parent.source_alpha[rows, columns],
            shape,
            parent.source_shape[rows, columns] if parent.source_shape is not None else None,
        )
        self.extend_paint_window(rows, columns)
        if before is not None:
            # The element is the group's paint where its alpha rounds to a byte above 0, and
            # the initial backdrop elsewhere, as the kernel takes it.
            quantized = numpy.rint(
                numpy.clip(group.source_alpha[rows, columns], 0.0, 1.0) * 255.0
            ).astype(numpy.uint8)
            element_alpha = numpy.where(
                quantized > 0, group.view[rows, columns, 3], initial_view[..., 3]
            )
            self.composite_plates(
                parent,
                group,
                rows,
                columns,
                before,
                quantized,
                KnockoutElement(initial_view[..., 3], element_alpha, shape),
            )
        return
    if parent.knockout:
        assert parent.source_alpha is not None
        assert shape is not None
        initial = (
            self.pixel_view(parent.backdrop)[rows, columns]
            if parent.backdrop is not None
            else numpy.zeros((*shape.shape, 4), dtype=numpy.uint8)
        )
        element = initial.copy()
        effective_alpha = self.composite_group_into(group, element, rows, columns)
        composite_knockout_group(
            parent.view[rows, columns],
            initial,
            element,
            parent.source_alpha[rows, columns],
            effective_alpha,
            shape,
        )
        if before is not None:
            self.composite_plates(
                parent,
                group,
                rows,
                columns,
                before,
                effective_alpha,
                KnockoutElement(initial[..., 3], element[..., 3], shape),
            )
    else:
        effective_alpha = self.composite_group_into(
            group, self.pixel_array[rows, columns], rows, columns
        )
        self.record_source_alpha(rows, columns, effective_alpha)
        if before is not None:
            self.composite_plates(parent, group, rows, columns, before, effective_alpha, None)
    if shape is not None and parent.source_shape is not None:
        parent_shape = parent.source_shape[rows, columns]
        parent_shape += (1.0 - parent_shape) * shape
    self.extend_paint_window(rows, columns)


def composite_group_into(
    self: RasterTarget, group: RasterGroup, destination: UInt8Array, rows: slice, columns: slice
) -> UInt8Array:
    child = group.view[rows, columns]
    group_alpha = group.composite_alpha
    normalized_blend_mode = self.resolved_blend(group.blend_mode, casefold=True)
    if group.overprint is not None:
        channels = group.overprint
        if channels[4] and normalized_blend_mode in {None, BlendOp.NORMAL} and child.size >= 64:
            # A paint naming a spot without overprinting takes its own value for the components
            # it marks and none for the rest (11.7.4.3, Table 146): Normal compositing of its
            # layer with those components white. Its colour is kept for its plate.
            colour = child[..., :3].copy()
            for index, marked in enumerate(channels[:3]):
                if not marked:
                    child[..., index] = 255
            try:
                plane = composite_normal_group(destination, child, 1.0, 1.0, True)
            finally:
                child[..., :3] = colour
            assert plane is not None
            return plane
        return composite_overprint_layer(
            destination,
            child,
            group.overprint,
            normalized_blend_mode,
            semantic_context=self.semantic_context,
        )
    source_scale = group.source_scale
    if group.mask_alpha is not None:
        return composite_masked_group(
            destination,
            child,
            group.source_alpha[rows, columns]
            if group.backdrop is not None and group.source_alpha is not None
            else None,
            source_scale,
            normalized_blend_mode,
            group.mask_alpha,
            (rows, columns),
            semantic_context=self.semantic_context,
        )
    if group.backdrop is not None:
        assert group.source_alpha is not None
        return composite_nonisolated_group(
            destination,
            child,
            group.source_alpha[rows, columns],
            source_scale,
            normalized_blend_mode,
            semantic_context=self.semantic_context,
        )
    if (normalized_blend_mode is None or normalized_blend_mode is BlendOp.NORMAL) and len(
        group.pixels
    ) >= 4_096:
        plane = composite_normal_group(destination, child, source_scale, 1.0, True)
        assert plane is not None
        return plane
    effective_alpha = numpy.clip(
        numpy.rint(child[..., 3].astype(numpy.float64) * source_scale), 0.0, 255.0
    ).astype(numpy.uint8)
    composite_blended_group_numpy(
        destination,
        child,
        float(group_alpha) if is_pdf_number(group_alpha) else None,
        None,
        normalized_blend_mode,
        semantic_context=self.semantic_context,
    )
    return effective_alpha


def push_scope(self: RasterTarget, clip_path: CapturedPath | None = None) -> None:
    self.scope_stack.append(
        (
            self.clip.depth,
            self.clip_stack,
            self.clip_floor,
            len(self.buffer_stack),
            self.group_floor,
        )
    )
    self.clip_stack = []
    try:
        if clip_path is not None:
            self.clip.push(clip_path, "nonzero")
        self.clip_floor = self.clip.depth
        self.group_floor = len(self.buffer_stack)
    except BaseException:
        self.pop_scope()
        raise


def pop_scope(self: RasterTarget) -> None:
    if not self.scope_stack:
        return
    clip_depth, clip_stack, clip_floor, buffer_depth, group_floor = self.scope_stack.pop()
    try:
        while len(self.buffer_stack) > buffer_depth:
            self.composite_group(self.pop_group())
    finally:
        while len(self.buffer_stack) > buffer_depth:
            self.pop_group()
        self.clip.restore(clip_depth)
        self.clip_stack = clip_stack
        self.clip_floor = clip_floor
        self.group_floor = group_floor


def push_scratch_group(
    self: RasterTarget,
    group_alpha: float | None,
    blend_mode: str | None,
    *,
    track_shape: bool,
    mask_alpha: SoftMaskPlane | None,
    alpha_is_shape: bool,
    region: PixelBox | None = None,
    sync_box: PixelBox | None = None,
) -> PixelBox | None:
    """Push a non-isolated group on the scratch planes of its depth, its pixels the
    backdrop's. A knockout group (region given) is a copy of the backdrop in region only.

    sync_box, for a group that paints only inside it, may limit the copy of the backdrop
    to that box. Return the box the group's paint must stay inside for its pixels to be
    those of a copy of the whole backdrop, or None when its copy is whole: paint_item
    paints an item that goes outside it again, with no sync_box.
    """
    parent = self.buffer_stack[-1]
    backdrop = parent.backdrop if parent.knockout else self.pixels
    knockout = region is not None
    if backdrop is None:
        self.push_group(
            bytearray(len(self.pixels)),
            group_alpha,
            blend_mode,
            isolated=False,
            knockout=knockout,
            track_shape=track_shape,
            mask_alpha=mask_alpha,
            alpha_is_shape=alpha_is_shape,
        )
        return None
    depth = len(self.buffer_stack)
    scratch = self.elementary_scratch.get(depth)
    if scratch is None:
        scratch = self.elementary_scratch[depth] = ElementaryScratch(
            len(self.pixels), self.height, self.width
        )
    buffer = scratch.buffer
    source_alpha = scratch.source_alpha
    source_shape = scratch.source_shape
    synced_box = scratch.synced_box
    limit = None
    if region is not None:
        copy_box(self, scratch, backdrop, region)
        scratch.synced_parent = None
    elif (
        parent.knockout
        and scratch.synced_parent is parent
        and (synced_box is None or (sync_box is not None and box_within(sync_box, synced_box)))
    ):
        # Synced but for the last group's paint, which stayed inside synced_box.
        dirty = scratch.dirty
        if dirty is not None:
            rows, columns = dirty
            scratch.view[rows, columns] = self.pixel_view(backdrop)[rows, columns]
            source_alpha[rows, columns] = 0.0
            source_shape[rows, columns] = 0.0
        limit = synced_box
    elif sync_box is not None:
        copy_box(self, scratch, backdrop, sync_box)
        scratch.synced_parent = parent if parent.knockout else None
        scratch.synced_box = limit = sync_box
    else:
        buffer[:] = backdrop
        source_alpha.fill(0.0)
        source_shape.fill(0.0)
        scratch.synced_parent = parent if parent.knockout else None
        scratch.synced_box = None
    scratch.dirty = None
    self.buffer_stack.append(
        RasterGroup(
            buffer,
            group_alpha,
            blend_mode,
            view=scratch.view,
            backdrop=backdrop,
            source_alpha=source_alpha,
            source_shape=(
                source_shape if knockout or track_shape or parent.source_shape is not None else None
            ),
            knockout=knockout,
            alpha_is_shape=alpha_is_shape,
            mask_alpha=mask_alpha,
            painted_boxes=[] if knockout else None,
            spots=self.child_plates(parent, False),
        )
    )
    self.sync_group_mirrors()
    return limit


def copy_box(
    self: RasterTarget, scratch: ElementaryScratch, backdrop: bytearray, box: PixelBox
) -> None:
    """Copy backdrop into scratch inside box, its source planes cleared there."""
    x0, y0, x1, y1 = box
    if x1 > x0 and y1 > y0:
        scratch.view[y0:y1, x0:x1] = self.pixel_view(backdrop)[y0:y1, x0:x1]
        scratch.source_alpha[y0:y1, x0:x1] = 0.0
        scratch.source_shape[y0:y1, x0:x1] = 0.0


def box_within(inner: PixelBox, outer: PixelBox) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def push_group(
    self: RasterTarget,
    buffer: bytearray,
    group_alpha: float | None,
    blend_mode: str | None,
    *,
    isolated: bool = True,
    knockout: bool = False,
    alpha_is_shape: bool = False,
    track_shape: bool = False,
    mask_alpha: SoftMaskPlane | None = None,
    keeps_plates: bool = False,
) -> None:
    """Push a group on buffer. keeps_plates: an isolated group starts from its parent's spot
    plates all the same, as one an overprinting paint paints through does (raster_spots)."""
    parent = self.buffer_stack[-1]
    backdrop = None if isolated else parent.backdrop if parent.knockout else self.pixels
    source_alpha = None
    if backdrop is not None:
        buffer[:] = backdrop
    if backdrop is not None or knockout:
        source_alpha = numpy.zeros((self.height, self.width), dtype=numpy.float32)
    source_shape = (
        numpy.zeros((self.height, self.width), dtype=numpy.float32)
        if knockout or track_shape or parent.source_shape is not None
        else None
    )
    self.buffer_stack.append(
        RasterGroup(
            buffer,
            group_alpha,
            blend_mode,
            view=uint8_image_view(buffer, (self.height, self.width, 4)),
            backdrop=backdrop,
            source_alpha=source_alpha,
            source_shape=source_shape,
            knockout=knockout,
            alpha_is_shape=alpha_is_shape,
            mask_alpha=mask_alpha,
            painted_boxes=[] if knockout else None,
            spots=self.child_plates(parent, isolated and not keeps_plates),
        )
    )
    self.sync_group_mirrors()


def pop_group(self: RasterTarget) -> RasterGroup:
    child = self.buffer_stack.pop()
    scratch = self.elementary_scratch.get(len(self.buffer_stack))
    if scratch is not None and child.pixels is scratch.buffer:
        scratch.dirty = child.paint_window.slices()
    self.sync_group_mirrors()
    return child


def sync_group_mirrors(self: RasterTarget) -> None:
    group = self.buffer_stack[-1]
    self.pixels = group.pixels
    self.pixel_array = group.view
    self.group_source_alpha = group.source_alpha
    self.group_source_shape = group.source_shape
    self.paint_window = group.paint_window if len(self.buffer_stack) > 1 else None


@contextmanager
def detached_buffer(
    self: RasterTarget, buffer: bytearray, window: PixelWindow | None = None
) -> Generator[None]:
    self.pixels = buffer
    self.pixel_array = self.pixel_view(buffer)
    self.group_source_alpha = None
    self.group_source_shape = None
    self.paint_window = window
    try:
        yield
    finally:
        self.sync_group_mirrors()
