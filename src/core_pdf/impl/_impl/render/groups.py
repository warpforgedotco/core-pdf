# SPDX-License-Identifier: AGPL-3.0-only
"""Composite backdrop-bearing RGB groups into the selected byte raster."""

from __future__ import annotations

from typing import Any

import numpy

from core_pdf.impl._impl.render.blend import internal_blend_visible_pixels, internal_clamp01
from core_pdf.impl._impl.runtime.array_views import UInt8Array
from core_pdf_spec.s_11_transparency.groups import (
    composite_knockout_element,
    remove_group_backdrop,
)
from core_pdf_spec.standards import SemanticContext


def internal_composite_nonisolated_group(
    destination: UInt8Array,
    rendered: UInt8Array,
    source_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]],
    opacity: float,
    blend_mode: str | None,
    *,
    semantic_context: SemanticContext,
    mask_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None = None,
) -> UInt8Array:
    """Remove the initial backdrop before applying the group's outer state once.

    The destination is the suspended parent's buffer, still equal to the initial
    backdrop. Return the effective source alpha for any enclosing non-isolated
    group. Empty areas must preserve even a transparent backdrop's hidden RGB.
    """
    opacity = internal_clamp01(opacity)
    scaled_alpha = source_alpha.astype(numpy.float64) * opacity * 255.0
    if mask_alpha is not None:
        scaled_alpha *= mask_alpha
    effective_alpha = numpy.rint(scaled_alpha).astype(numpy.uint8)
    visible = effective_alpha > 0
    if not numpy.any(visible):
        return effective_alpha
    mode = blend_mode.casefold() if isinstance(blend_mode, str) else None
    if opacity == 1.0 and mode in {None, "normal"}:
        # The rendered result already includes this exact backdrop. At full
        # opacity/Normal, replacing covered pixels avoids a lossy round trip.
        if mask_alpha is None:
            destination[visible] = rendered[visible]
            return effective_alpha
        unchanged_alpha = visible & (mask_alpha == 1.0)
        destination[unchanged_alpha] = rendered[unchanged_alpha]
        visible &= ~unchanged_alpha
        if not numpy.any(visible):
            return effective_alpha
    backdrop = destination[visible].astype(numpy.float64)
    result = rendered[visible].astype(numpy.float64) / 255.0
    colors, _ = remove_group_backdrop(
        result[..., :3],
        result[..., 3],
        backdrop[..., :3] / 255.0,
        backdrop[..., 3] / 255.0,
        source_alpha[visible],
        validate=False,
    )
    # Byte rounding during painting can put the reconstructed color just
    # outside gamut; clipping belongs to this selected raster output policy.
    colors = numpy.clip(colors, 0.0, 1.0)
    internal_blend_visible_pixels(
        destination,
        visible,
        colors[..., 0],
        colors[..., 1],
        colors[..., 2],
        effective_alpha[visible].astype(numpy.float64) / 255.0,
        mode,
        semantic_context=semantic_context,
    )
    return effective_alpha


def internal_composite_masked_group(
    destination: UInt8Array,
    rendered: UInt8Array,
    source_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None,
    opacity: float,
    blend_mode: str | None,
    mask_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]],
    *,
    semantic_context: SemanticContext,
) -> UInt8Array:
    """Apply a graphics soft mask once to a group's source alpha, not its backdrop."""
    if source_alpha is not None:
        return internal_composite_nonisolated_group(
            destination,
            rendered,
            source_alpha,
            opacity,
            blend_mode,
            semantic_context=semantic_context,
            mask_alpha=mask_alpha,
        )
    # An isolated group has no initial backdrop to remove. Its stored alpha
    # already describes just its own marks, including nested masked objects.
    effective_alpha = numpy.clip(
        numpy.rint(rendered[..., 3].astype(numpy.float64) * opacity * mask_alpha), 0, 255
    ).astype(numpy.uint8)
    visible = effective_alpha > 0
    if not numpy.any(visible):
        return effective_alpha
    colors = rendered[visible, :3].astype(numpy.float64) / 255.0
    internal_blend_visible_pixels(
        destination,
        visible,
        colors[:, 0],
        colors[:, 1],
        colors[:, 2],
        effective_alpha[visible].astype(numpy.float64) / 255.0,
        blend_mode.casefold() if isinstance(blend_mode, str) else None,
        semantic_context=semantic_context,
    )
    return effective_alpha


def internal_composite_knockout_group(
    destination: UInt8Array,
    backdrop: UInt8Array,
    element: UInt8Array,
    group_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]],
    element_alpha: UInt8Array,
    shape: numpy.ndarray[Any, Any],
) -> None:
    """Replace group contributions by element shape using the strict group equation."""
    # Raster alpha and coverage are independently rounded. Reconcile the small
    # discrepancy at the output boundary before passing unit samples to spec.
    effective_alpha = element_alpha.astype(numpy.float64) / 255.0
    shape = numpy.maximum(numpy.clip(shape, 0.0, 1.0), effective_alpha)
    visible = shape > 0.0
    if not numpy.any(visible):
        return
    previous = destination[visible].astype(numpy.float64) / 255.0
    initial = backdrop[visible].astype(numpy.float64) / 255.0
    painted = element[visible].astype(numpy.float64) / 255.0
    colors, complete, accumulated = composite_knockout_element(
        previous[..., :3],
        previous[..., 3],
        backdrop_components=initial[..., :3],
        backdrop_alpha=initial[..., 3],
        element_components=painted[..., :3],
        element_alpha=painted[..., 3],
        shape=shape[visible],
        group_alpha=group_alpha[visible],
        element_group_alpha=effective_alpha[visible],
        validate=False,
    )
    destination[visible] = numpy.clip(
        numpy.rint(numpy.column_stack((colors, complete)) * 255.0), 0, 255
    ).astype(numpy.uint8)
    group_alpha[visible] = accumulated
