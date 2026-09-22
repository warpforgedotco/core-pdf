# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

from core_pdf.impl.capture.program import CapturedProgram
from core_pdf.impl.capture.records import CapturedSoftMask
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.model import GraphicsState

if TYPE_CHECKING:
    from core_pdf.impl.capture.recording import RecordingMethods


GRAPHICS_STATE_FIELDS = GraphicsState.__fields__


def internal_state_key(value: object) -> object:
    if isinstance(value, tuple):
        return tuple(internal_state_key(part) for part in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value), value)
    return ("identity", id(value))


def capture_graphics_soft_mask(state: RecordingMethods) -> CapturedSoftMask | None:
    mask = state.graphics.soft_mask
    if mask is None:
        return None
    graphics = copy(state.graphics)
    graphics.ctm = mask.ctm
    graphics.soft_mask = None
    graphics.fill_opacity = graphics.stroke_opacity = 1.0
    graphics.blend_mode = None
    key = (
        id(mask),
        tuple(internal_state_key(getattr(graphics, name)) for name in GRAPHICS_STATE_FIELDS),
    )
    cached = state.capture_soft_masks.get(key)
    if cached is not None:
        return cached[2]
    state.capture_soft_masks[key] = (mask, graphics, None)
    group_key = id(mask.group)
    if mask.subtype != "Alpha" or group_key in state.capture_active_mask_groups:
        return None
    if len(state.capture_active_mask_groups) >= 10:
        return None
    state.capture_active_mask_groups.add(group_key)
    try:
        nested = state.nested_capture_state()
        nested.graphics = copy(graphics)
        scope = state.capture_mask_resources.get(id(mask))
        nested.resources = scope[1] if scope is not None else state.resources
        frame = nested.append_form_xobject(mask.group, 0)
        if frame is None:
            return None
        nested.stream_executor.consume_frame(frame)
        nested.run_accumulator.flush()
        if not nested.text_boundaries:
            return None
        result = CapturedSoftMask(
            CapturedProgram(
                runs=tuple(nested.runs),
                glyphs=tuple(nested.glyphs),
                drawings=tuple(nested.drawings),
                inline_images=tuple(nested.inline_images),
                lines=tuple(nested.lines),
                text_boundaries=tuple(nested.text_boundaries),
            ),
            mask.transfer,
        )
        state.capture_soft_masks[key] = (mask, graphics, result)
        return result
    except PdfParseError, TypeError, ValueError, ArithmeticError:
        return None
    finally:
        state.capture_active_mask_groups.remove(group_key)
