# SPDX-License-Identifier: AGPL-3.0-only
"""Capture mask artwork independently of page extraction and paint ordering."""

from __future__ import annotations

from copy import copy
from dataclasses import fields
from typing import TYPE_CHECKING

from core_pdf.impl._impl.capture.program import CapturedProgram
from core_pdf.impl._impl.capture.records import CapturedSoftMask
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.model import GraphicsState

if TYPE_CHECKING:
    from core_pdf.impl._impl.capture.recording import RecordingMethods


GRAPHICS_STATE_FIELDS = tuple(item.name for item in fields(GraphicsState))


def internal_state_key(value: object) -> object:
    if isinstance(value, tuple):
        return tuple(internal_state_key(part) for part in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value), value)
    return ("identity", id(value))


def capture_graphics_soft_mask(state: RecordingMethods) -> CapturedSoftMask | None:
    """Freeze a selected Alpha mask without adding its marks to page products.

    CTM/resources come from installation; other inherited graphics parameters
    come from this paint, following ordinary Form invocation. Failed, recursive,
    or unsupported masks retain the reader's unmasked fallback.
    """
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
    # Keep the snapshot alive with the cache key: identities used in that key
    # cannot be recycled while their captured result remains reusable.
    state.capture_soft_masks[key] = (mask, graphics, None)
    group_key = id(mask.group)
    if mask.subtype != "Alpha" or group_key in state.capture_active_mask_groups:
        return None
    # A mask can install another mask whose G installs the original again.
    # These capture states share a guard even though their stream stacks differ.
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
            # A valid empty Form still emits stream scope boundaries. Their
            # absence means entry failed before its content could be evaluated.
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
