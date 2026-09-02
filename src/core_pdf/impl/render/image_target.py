# SPDX-License-Identifier: AGPL-3.0-only
"""Stateful decoded-image painting facade for raster targets."""

from core_pdf.impl.render.image_affine_target import internal_ImageAffineTargetMixin
from core_pdf.impl.render.image_axis_target import internal_ImageAxisTargetMixin


class internal_ImageTargetMixin(
    internal_ImageAffineTargetMixin,
    internal_ImageAxisTargetMixin,
):
    """Combine affine and axis-aligned decoded-image painting operations."""

    __slots__ = ()
