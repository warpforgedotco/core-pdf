# SPDX-License-Identifier: AGPL-3.0-only
"""Mutable raster target, clipping, and render metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from core_pdf.impl.render.blend import internal_BlendTargetMixin
from core_pdf.impl.render.clipping import internal_ClipState
from core_pdf.impl.render.images import internal_ImageTargetMixin
from core_pdf.impl.render.paths import internal_PathTargetMixin
from core_pdf.impl.render.patterns import internal_PatternTargetMixin
from core_pdf.impl.runtime.array_views import uint8_image_view

if TYPE_CHECKING:
    from core_pdf.impl.render.page import RenderedPage


class internal_RasterTarget(
    internal_BlendTargetMixin,
    internal_ImageTargetMixin,
    internal_PathTargetMixin,
    internal_PatternTargetMixin,
):
    """The RGBA byte buffer being painted, plus the transparency-group stack.

    Lifted out of ``RenderedPage.rasterize``. ``pixels`` is *rebound*, not just
    mutated: a ``group-begin`` pushes a fresh buffer that subsequent painting
    goes to, and ``group-end`` pops it and composites it back down. That is why
    this is an object with explicit push/pop rather than a plain buffer.

    Every method hoists ``self.pixels`` into a local before touching it — these
    run per pixel and per span, where a repeated attribute load is not free.
    """

    __slots__ = (
        "pixels",
        "buffer_stack",
        "pixel_views",
        "clip",
        "width",
        "height",
        "scale",
        "crop_x0",
        "crop_y1",
        "page_pixels",
        "page_buffer",
        "raster_x_coordinate_cache",
        "raster_y_coordinate_cache",
        "raster_x_sample_cache",
        "raster_y_sample_cache",
        # Bound clip methods cached at construction. Reading them back is a plain
        # attribute load; going through `self.clip.<name>` would allocate a fresh
        # bound method on every call, and fill_rect alone makes ~1.8M of them.
        "page_box_to_pixels",
        "current_clip",
        "clip_paths_are_axis_aligned_rects",
        "clip_row_visible_spans",
        "pixel_in_clip",
        "page",
        "raster_metrics",
        "page_x_coordinate_cache",
        "page_y_coordinate_cache",
        "crop_y0",
        "color_cache",
        "mark_clip_metadata_dirty",
        "path_bbox",
        "clip_path_stack",
    )

    def __init__(
        self,
        pixels: bytearray,
        group_alpha: float | None,
        *,
        clip: internal_ClipState,
        page: "RenderedPage",
        raster_metrics: internal_RasterMetrics,
        width: int,
        height: int,
        scale: float,
        crop_x0: float,
        crop_y0: float,
        crop_y1: float,
        page_view: numpy.ndarray[Any, Any],
    ) -> None:
        self.pixels = pixels
        self.buffer_stack: list[tuple[bytearray, float | None, str | None]] = [
            (pixels, group_alpha, None)
        ]
        self.pixel_views: dict[int, numpy.ndarray[Any, Any]] = {id(pixels): page_view}
        self.clip = clip
        self.width = width
        self.height = height
        self.scale = scale
        self.crop_x0 = crop_x0
        self.crop_y1 = crop_y1
        self.page_pixels = page_view
        self.page_buffer = pixels
        self.raster_x_coordinate_cache: dict[tuple[int, int], numpy.ndarray[Any, Any]] = {}
        self.raster_y_coordinate_cache: dict[tuple[int, int], numpy.ndarray[Any, Any]] = {}
        self.raster_x_sample_cache: dict[int, tuple[float, ...]] = {}
        self.raster_y_sample_cache: dict[int, tuple[float, ...]] = {}
        self.page_box_to_pixels = clip.page_box_to_pixels
        self.current_clip = clip.current_clip
        self.clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
        self.clip_row_visible_spans = clip.clip_row_visible_spans
        self.pixel_in_clip = clip.pixel_in_clip
        self.page = page
        self.raster_metrics = raster_metrics
        self.page_x_coordinate_cache: dict[tuple[int, int], numpy.ndarray[Any, Any]] = {}
        self.page_y_coordinate_cache: dict[tuple[int, int], numpy.ndarray[Any, Any]] = {}
        self.crop_y0 = crop_y0
        self.color_cache: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        self.mark_clip_metadata_dirty = clip.mark_clip_metadata_dirty
        self.path_bbox = clip.path_bbox
        self.clip_path_stack = clip.clip_path_stack

    def push_group(
        self, buffer: bytearray, group_alpha: float | None, blend_mode: str | None
    ) -> None:
        self.buffer_stack.append((buffer, group_alpha, blend_mode))
        self.pixels = buffer

    def pop_group(self) -> tuple[bytearray, float | None, str | None]:
        child = self.buffer_stack.pop()
        self.pixels = self.buffer_stack[-1][0]
        return child

    def pixel_view(self, buffer: bytearray | bytes) -> numpy.ndarray[Any, Any]:
        """Return a reusable array view for an active RGBA byte buffer."""
        pixel_views = self.pixel_views
        key = id(buffer)
        view = pixel_views.get(key)
        if view is None:
            view = uint8_image_view(buffer, (self.height, self.width, 4))
            pixel_views[key] = view
        return view


class internal_RasterMetrics:
    """Image-decode and tiled-blit tallies collected while a page rasterizes.

    These are not debug counters: ``tests/benchmarks`` asserts on every field
    (an image is decoded exactly once, tiled affine blitting stays under a 1 MiB
    scratch budget). Keep them wired up through any refactor.
    """

    __slots__ = (
        "image_count",
        "image_decode_seconds",
        "image_blit_seconds",
        "tiled_affine_blit_count",
        "tiled_affine_peak_scratch_bytes",
    )

    def __init__(self) -> None:
        self.image_count = 0
        self.image_decode_seconds = 0.0
        self.image_blit_seconds = 0.0
        self.tiled_affine_blit_count = 0
        self.tiled_affine_peak_scratch_bytes = 0

    def as_metadata(self) -> dict[str, float | int]:
        return {
            "image_count": self.image_count,
            "decode_seconds": self.image_decode_seconds,
            "blit_seconds": self.image_blit_seconds,
            "tiled_affine_blit_count": self.tiled_affine_blit_count,
            "tiled_affine_peak_scratch_bytes": self.tiled_affine_peak_scratch_bytes,
        }
