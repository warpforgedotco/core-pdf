# SPDX-License-Identifier: AGPL-3.0-only
"""Composite backdrop-bearing RGB groups into the selected byte raster."""

from __future__ import annotations

from typing import Any

import numpy

from core_pdf.impl._impl.render.blend import internal_blend_channels_f64
from core_pdf.impl._impl.runtime.array_views import UInt8Array
from core_pdf_spec.s_11_transparency.groups import remove_group_backdrop
from core_pdf_spec.standards import SemanticContext


def internal_composite_nonisolated_group(
    destination: UInt8Array,
    rendered: UInt8Array,
    source_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]],
    opacity: float,
    blend_mode: str | None,
    *,
    semantic_context: SemanticContext,
) -> UInt8Array:
    """Remove the initial backdrop before applying the group's outer state once.

    The destination is the suspended parent's buffer, still equal to the initial
    backdrop. Return the effective source alpha for any enclosing non-isolated
    group. Empty areas must preserve even a transparent backdrop's hidden RGB.
    """
    opacity = max(0.0, min(1.0, opacity))
    effective_alpha = numpy.rint(source_alpha.astype(numpy.float64) * opacity * 255.0).astype(
        numpy.uint8
    )
    visible = effective_alpha > 0
    if not numpy.any(visible):
        return effective_alpha
    mode = blend_mode.casefold() if isinstance(blend_mode, str) else None
    if opacity == 1.0 and mode in {None, "normal"}:
        # The rendered result already includes this exact backdrop. At full
        # opacity/Normal, replacing covered pixels avoids a lossy round trip.
        destination[visible] = rendered[visible]
        return effective_alpha
    backdrop = destination[visible].astype(numpy.float64)
    result = rendered[visible].astype(numpy.float64) / 255.0
    colors, _ = remove_group_backdrop(
        result[..., :3],
        result[..., 3],
        backdrop[..., :3] / 255.0,
        backdrop[..., 3] / 255.0,
        source_alpha[visible],
    )
    # Byte rounding during painting can put the reconstructed color just
    # outside gamut; clipping belongs to this selected raster output policy.
    colors = numpy.clip(colors, 0.0, 1.0)
    channels = internal_blend_channels_f64(
        colors[..., 0],
        colors[..., 1],
        colors[..., 2],
        effective_alpha[visible].astype(numpy.float64) / 255.0,
        backdrop[..., 0],
        backdrop[..., 1],
        backdrop[..., 2],
        backdrop[..., 3],
        mode,
        semantic_context=semantic_context,
    )
    for channel, values in enumerate(channels):
        destination[..., channel][visible] = numpy.clip(values, 0.0, 255.0).astype(numpy.uint8)
    return effective_alpha
