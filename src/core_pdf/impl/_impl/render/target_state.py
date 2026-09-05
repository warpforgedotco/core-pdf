# SPDX-License-Identifier: AGPL-3.0-only
"""State and operations required by the raster target's painting mixins.

Imported only while type checking. The concrete target owns the buffers and clip
state; this interface checks cross-mixin calls without a runtime adapter.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

import numpy

from core_pdf.impl._impl.render.clipping import internal_ClipState
from core_pdf.impl._impl.render.model import DisplayItem, ImagePaintItem, PathPaintItem
from core_pdf.impl._impl.runtime.array_views import ByteBuffer, UInt8Array
from core_pdf.impl.spec.s_07_content.capture import CapturedPath, TilingPattern
from core_pdf.impl.spec.s_08_graphics.image_decode import PreparedImage
from core_pdf.impl.spec.s_08_graphics.shading import PreparedShading


class internal_RasterState(Protocol):
    pixels: bytearray
    buffer_stack: list[tuple[bytearray, float | None, str | None]]
    clip: internal_ClipState
    width: int
    height: int
    scale: float
    crop_x0: float
    crop_y0: float
    crop_y1: float
    page_pixels: UInt8Array
    page_buffer: bytearray

    def blend_normal_pixel(self, idx: int, sr: int, sg: int, sb: int, sa: int) -> None: ...

    def blend_normal_solid_span(
        self, row: int, start: int, end: int, rgba: tuple[int, int, int, int]
    ) -> None: ...

    def blend_px(
        self,
        idx: int,
        rgba: tuple[int, int, int, int],
        target_alpha_scale: float | None,
        mode: str | None,
    ) -> None: ...

    def blit_affine_image(
        self,
        quad: tuple[tuple[float, float], ...],
        converted: ByteBuffer,
        width_px: int,
        height_px: int,
        comps: int,
        constant_alpha: float | None,
        blend_mode: str | None,
    ) -> bool: ...

    def blit_image_mask(
        self, item: ImagePaintItem, prepared: PreparedImage, blend_mode: str | None
    ) -> None: ...

    def blit_image_rows_blended(
        self,
        converted: ByteBuffer,
        comps: int,
        source_alpha: UInt8Array | None,
        constant_alpha: float,
        has_constant_alpha: bool,
        soft_mask: UInt8Array | None,
        x_unit_map: numpy.ndarray[Any, Any],
        y_unit_map: numpy.ndarray[Any, Any],
        src_x_map: numpy.ndarray[Any, Any],
        src_y_map: numpy.ndarray[Any, Any],
        ix0: int,
        iy0: int,
        ix1: int,
        iy1: int,
        x_span: int,
        y_span: int,
        width_px: int,
    ) -> None: ...

    def blit_image_rows_opaque(
        self,
        converted: ByteBuffer,
        comps: int,
        src_x_map: numpy.ndarray[Any, Any],
        src_y_map: numpy.ndarray[Any, Any],
        ix0: int,
        iy0: int,
        ix1: int,
        iy1: int,
        width_px: int,
        height_px: int,
    ) -> None: ...

    def blit_opaque_sampled_tiles(
        self,
        source_pixels: numpy.ndarray[Any, Any],
        target_region: numpy.ndarray[Any, Any],
        source_y: numpy.ndarray[Any, Any],
        source_x: numpy.ndarray[Any, Any],
        valid_rows: numpy.ndarray[Any, Any],
        valid_columns: numpy.ndarray[Any, Any],
        comps: int,
        *,
        transposed: bool = False,
    ) -> None: ...

    def can_blend_normal_fast(self, blend_mode: str | None) -> bool: ...

    def fast_fill_path(
        self,
        edges: list[tuple[float, float, float, float]],
        bbox: tuple[float, float, float, float],
    ) -> bool: ...

    def fill_cap(
        self,
        px: float,
        py: float,
        line_width: float,
        rgba: tuple[int, int, int, int],
        line_cap: int,
        blend_mode: str | None = None,
    ) -> None: ...

    def fill_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
    ) -> None: ...

    def fill_join(
        self,
        px: float,
        py: float,
        line_width: float,
        rgba: tuple[int, int, int, int],
        line_join: int = 0,
        blend_mode: str | None = None,
    ) -> None: ...

    def fill_line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        line_width: float,
        rgba: tuple[int, int, int, int],
        dash_pattern: tuple[list[float], float] | None = None,
        blend_mode: str | None = None,
        line_cap: int = 0,
    ) -> None: ...

    def fill_path_scanlines(
        self,
        edge_segments: list[tuple[float, float, float, float, float, float]],
        pixel_box: tuple[int, int, int, int],
        rgba: tuple[int, int, int, int],
        blend_mode: str | None,
        fill_rule: str,
    ) -> None: ...

    def fill_rect(
        self,
        box: tuple[float, float, float, float] | None,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
    ) -> None: ...

    def internal_resolved_blend(
        self, blend_mode: str | None
    ) -> tuple[float | None, str | None]: ...

    def paint_items(
        self,
        items: Iterable[DisplayItem],
        *,
        translation: tuple[float, float] | None = None,
        parent_blend_mode: str | None = None,
        clip_path: CapturedPath | None = None,
    ) -> None: ...

    def paint_shading(self, data: dict[str, Any], blend_mode: str | None) -> None: ...

    def paint_tiling_pattern(
        self, pattern: TilingPattern, target_data: PathPaintItem, blend_mode: str | None
    ) -> bool: ...

    def pixel_view(self, buffer: bytearray | bytes) -> UInt8Array: ...

    def shading_box(
        self, data: dict[str, Any], shading: PreparedShading
    ) -> tuple[float, float, float, float]: ...
