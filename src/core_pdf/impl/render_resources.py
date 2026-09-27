# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_pdf.impl.caches import ByteBudgetCache, IdentityCache
from core_pdf.impl.capture_records import CapturedPath, CapturedSoftMask
from core_pdf.impl.graphics_images import PreparedImage
from core_pdf.impl.graphics_shading import PreparedShading, ShadingEvaluatorCache
from core_pdf.impl.render_display import DisplayList
from core_pdf.impl.render_model import SoftMaskPlane
from core_pdf_spec.s_08_graphics.image_spec import ImageSource

SoftMaskKey = tuple[int, int, tuple[float, float]]
SOFT_MASK_CACHE_BYTES = 512 << 20
PREPARED_IMAGE_CACHE_BYTES = 256 << 20
SHADING_CACHE_LIMIT = 4096
TILING_CELL_CACHE_LIMIT = 4096

type SoftMaskCache = ByteBudgetCache[SoftMaskKey, tuple[CapturedSoftMask, SoftMaskPlane | None]]
type PreparedImageCache = ByteBudgetCache[int, tuple[ImageSource, PreparedImage | None]]
type PreparedShadingCache = IdentityCache[PreparedShading | None]
type TilingCellCache = IdentityCache[tuple[DisplayList, CapturedPath]]


class RenderResources:
    __slots__ = (
        "soft_masks",
        "images",
        "tiling_cells",
        "shadings",
        "shading_evaluators",
        "active_soft_masks",
    )

    def __init__(self) -> None:
        self.soft_masks: SoftMaskCache = ByteBudgetCache(SOFT_MASK_CACHE_BYTES)
        self.images: PreparedImageCache = ByteBudgetCache(PREPARED_IMAGE_CACHE_BYTES)
        self.tiling_cells: TilingCellCache = IdentityCache(TILING_CELL_CACHE_LIMIT)
        self.shadings: PreparedShadingCache = IdentityCache(SHADING_CACHE_LIMIT)
        self.shading_evaluators: ShadingEvaluatorCache = IdentityCache(SHADING_CACHE_LIMIT)
        self.active_soft_masks: set[SoftMaskKey] = set()
