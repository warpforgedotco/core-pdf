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
from core_pdf.impl.render.blend import BlendOp, blend_op, composite_blended_group_numpy
from core_pdf.impl.render.model import PixelWindow, RasterGroup, SoftMaskPlane
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
        and (
            group.blend_mode is None or blend_op(group.blend_mode, casefold=True) is BlendOp.NORMAL
        )
    ):
        assert parent.source_alpha is not None
        assert shape is not None
        composite_elementary_knockout(
            parent.view[rows, columns],
            self.pixel_view(parent.backdrop)[rows, columns],
            group.view[rows, columns],
            group.source_alpha[rows, columns],
            parent.source_alpha[rows, columns],
            shape,
            parent.source_shape[rows, columns] if parent.source_shape is not None else None,
        )
        self.extend_paint_window(rows, columns)
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
    else:
        effective_alpha = self.composite_group_into(
            group, self.pixel_array[rows, columns], rows, columns
        )
        self.record_source_alpha(rows, columns, effective_alpha)
    if shape is not None and parent.source_shape is not None:
        parent_shape = parent.source_shape[rows, columns]
        parent_shape += (1.0 - parent_shape) * shape
    self.extend_paint_window(rows, columns)


def composite_group_into(
    self: RasterTarget,
    group: RasterGroup,
    destination: UInt8Array,
    rows: slice,
    columns: slice,
) -> UInt8Array:
    child = group.view[rows, columns]
    group_alpha = group.composite_alpha
    group_blend_mode = group.blend_mode
    normalized_blend_mode = blend_op(group_blend_mode, casefold=True)
    source_scale = group.source_scale
    if group.mask_alpha is not None:
        return composite_masked_group(
            destination,
            child,
            group.source_alpha[rows, columns]
            if group.backdrop is not None and group.source_alpha is not None
            else None,
            source_scale,
            group_blend_mode,
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
            group_blend_mode,
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
        group_blend_mode,
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
) -> None:
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
        return
    depth = len(self.buffer_stack)
    scratch = self.elementary_scratch.get(depth)
    if scratch is None:
        scratch = self.elementary_scratch[depth] = ElementaryScratch(
            len(self.pixels), self.height, self.width
        )
    buffer = scratch.buffer
    source_alpha = scratch.source_alpha
    source_shape = scratch.source_shape
    if region is not None:
        x0, y0, x1, y1 = region
        if x1 > x0 and y1 > y0:
            scratch.view[y0:y1, x0:x1] = self.pixel_view(backdrop)[y0:y1, x0:x1]
            source_alpha[y0:y1, x0:x1] = 0.0
            source_shape[y0:y1, x0:x1] = 0.0
        scratch.synced_parent = None
    elif parent.knockout and scratch.synced_parent is parent:
        dirty = scratch.dirty
        if dirty is not None:
            rows, columns = dirty
            scratch.view[rows, columns] = self.pixel_view(backdrop)[rows, columns]
            source_alpha[rows, columns] = 0.0
            source_shape[rows, columns] = 0.0
        scratch.synced_parent = parent
    else:
        buffer[:] = backdrop
        source_alpha.fill(0.0)
        source_shape.fill(0.0)
        scratch.synced_parent = parent if parent.knockout else None
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
        )
    )
    self.sync_group_mirrors()


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
) -> None:
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
