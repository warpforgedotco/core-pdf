# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core_pdf.impl.capture_host import CaptureHost
from core_pdf.impl.capture_records import (
    CapturedDrawing,
    CapturedPath,
    CapturedSubpath,
    CapturedTextBoundary,
    marker_drawing,
)
from core_pdf.impl.geometry import intersect_bbox, transform_bbox
from core_pdf.impl.types import Rectangle
from core_pdf_spec.s_07_content.streams import ContentStreamFrame

if TYPE_CHECKING:
    pass


@dataclass(slots=True)
class CaptureGraphicsSave:
    clip_bbox: Rectangle | None
    group_alpha: float | None
    clip_scope_emitted: bool = False


class ScopeCaptureMixin(CaptureHost):
    __slots__ = ()

    def save_graphics(self, state: object) -> None:
        self.capture_graphics_stack.append(CaptureGraphicsSave(self.clip_bbox, self.group_alpha))

    def restore_graphics(self, state: object) -> None:
        saved = self.capture_graphics_stack.pop()
        self.clip_bbox = saved.clip_bbox
        self.group_alpha = saved.group_alpha
        if saved.clip_scope_emitted:
            self.drawings.append(marker_drawing("state-pop", self.sequence))
            self.sequence += 1

    def enter_stream(self, state: object, frame: ContentStreamFrame) -> None:
        self.capture_text_frames[id(frame)] = (self.capture_text_open, False)
        self.capture_text_open = False
        self.capture_frames[id(frame)] = (
            self.layout_form_bbox,
            self.layout_form_id,
            self.pending_line_break,
        )
        if frame.form_bbox is not None:
            x0, y0, x1, y1 = frame.form_bbox
            clip_path = CapturedPath()
            if x1 > x0 and y1 > y0:
                clip_path.subpaths.append(
                    CapturedSubpath([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True)
                )
                clip_path = clip_path.transformed(frame.ctm)
            self.drawings.append(
                CapturedDrawing(
                    kind="scope-begin",
                    seqno=self.sequence,
                    path=clip_path,
                    fill=None,
                    fill_opacity=None,
                    line_width=0.0,
                )
            )
            self.sequence += 1
        if frame.group_alpha is not None:
            self.drawings.append(
                marker_drawing(
                    "group-begin",
                    self.sequence,
                    fill_opacity=frame.group_alpha,
                    blend_mode=self.graphics.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    group_isolated=frame.group_isolated,
                    group_knockout=frame.group_knockout,
                    alpha_is_shape=self.graphics.alpha_is_shape,
                    graphics_soft_mask=self.capture_graphics_soft_mask(),
                )
            )
            self.sequence += 1
            self.group_alpha = None
        self.text_boundaries.append(CapturedTextBoundary(self.sequence, "stream-begin"))
        previous_text_open, _ = self.capture_text_frames[id(frame)]
        self.capture_text_frames[id(frame)] = (previous_text_open, True)
        layout_bbox = None
        raw_bbox = frame.form_bbox_operand
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            x, y, w, h = (
                self.resolver.resolve_float(value, default=None) for value in raw_bbox[:4]
            )
            if x is not None and y is not None and w is not None and h is not None:
                layout_bbox = transform_bbox((x, y, x + w, y + h), frame.ctm)
        self.layout_form_bbox = layout_bbox
        if frame.is_form:
            self.layout_form_id = (*(self.layout_form_id or ()), (frame.source_key, layout_bbox))
        else:
            self.layout_form_id = None
        if frame.clip_bbox is not None:
            self.clip_bbox = intersect_bbox(self.clip_bbox, frame.clip_bbox)
        self.pending_line_break = False
        self.stream_order += 1

    def exit_stream(self, state: object, frame: ContentStreamFrame) -> None:
        if id(frame) in self.capture_text_frames:
            previous_text_open, started = self.capture_text_frames.pop(id(frame))
            if started and self.capture_text_open:
                self.text_boundaries.append(CapturedTextBoundary(self.sequence, "end"))
            if started:
                self.text_boundaries.append(CapturedTextBoundary(self.sequence, "stream-end"))
            self.capture_text_open = previous_text_open
        old = self.capture_frames.pop(id(frame), None)
        if old is not None:
            self.layout_form_bbox, self.layout_form_id, self.pending_line_break = old
        if self.capture_marked_entries:
            active_entries = {id(entry) for entry in self.marked_content_stack}
            self.capture_marked_entries = {
                key: value
                for key, value in self.capture_marked_entries.items()
                if key in active_entries
            }
        if frame.group_alpha is not None:
            self.drawings.append(
                marker_drawing(
                    "group-end",
                    self.sequence,
                    fill_opacity=frame.group_alpha,
                    blend_mode=self.graphics.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    group_isolated=frame.group_isolated,
                    group_knockout=frame.group_knockout,
                    alpha_is_shape=self.graphics.alpha_is_shape,
                )
            )
            self.sequence += 1
        if old is not None and frame.form_bbox is not None:
            self.drawings.append(marker_drawing("scope-end", self.sequence))
            self.sequence += 1

    def emit_clip_scope_push(self) -> None:
        if not self.capture_graphics_stack or self.capture_graphics_stack[-1].clip_scope_emitted:
            return
        self.capture_graphics_stack[-1].clip_scope_emitted = True
        self.drawings.append(marker_drawing("state-push", self.sequence))
        self.sequence += 1

    def is_clipped_away(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        for clip in (self.clip_bbox, self.page_clip):
            if clip is None:
                continue
            if x1 <= clip[0] or x0 >= clip[2] or y1 <= clip[1] or y0 >= clip[3]:
                return True
        return False


__all__ = (
    "CaptureGraphicsSave",
    "ScopeCaptureMixin",
)
