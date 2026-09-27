# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy

from core_pdf.impl.array_views import uint8_image_view
from core_pdf.impl.capture_records import CapturedPath
from core_pdf.impl.render_compositing import RasterCompositing
from core_pdf.impl.render_model import PixelWindow, RasterGroup, SoftMaskPlane
from core_pdf.impl.render_raster_state import ElementaryScratch, PixelBox


class RasterGroups(RasterCompositing):
    __slots__ = ()

    def push_scope(self, clip_path: CapturedPath | None = None) -> None:
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

    def pop_scope(self) -> None:
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
        self,
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
                    source_shape
                    if knockout or track_shape or parent.source_shape is not None
                    else None
                ),
                knockout=knockout,
                alpha_is_shape=alpha_is_shape,
                mask_alpha=mask_alpha,
                painted_boxes=[] if knockout else None,
            )
        )
        self.sync_group_mirrors()

    def push_group(
        self,
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

    def pop_group(self) -> RasterGroup:
        child = self.buffer_stack.pop()
        scratch = self.elementary_scratch.get(len(self.buffer_stack))
        if scratch is not None and child.pixels is scratch.buffer:
            scratch.dirty = child.paint_window.slices()
        self.sync_group_mirrors()
        return child

    def sync_group_mirrors(self) -> None:
        group = self.buffer_stack[-1]
        self.pixels = group.pixels
        self.pixel_array = group.view
        self.group_source_alpha = group.source_alpha
        self.group_source_shape = group.source_shape
        self.paint_window = group.paint_window if len(self.buffer_stack) > 1 else None

    @contextmanager
    def detached_buffer(
        self, buffer: bytearray, window: PixelWindow | None = None
    ) -> Iterator[None]:
        self.pixels = buffer
        self.pixel_array = self.pixel_view(buffer)
        self.group_source_alpha = None
        self.group_source_shape = None
        self.paint_window = window
        try:
            yield
        finally:
            self.sync_group_mirrors()
