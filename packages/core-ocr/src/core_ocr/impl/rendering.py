# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

from core_ocr.impl.types import (
    LEPTONICA_PIX_COLOR_BYTES_PER_PIXEL,
    LEPTONICA_PIX_MAX_BYTES,
    OcrImage,
    OcrIteratorLayout,
    leptonica_pix_size_is_supported,
    ocr_float_value,
)

OCR_RENDER_DPI_CANDIDATES = (300, 400)
OCR_RENDER_MAX_DPI = 400
OCR_RENDER_TILE_MAX_SIDE_PIXELS = 8192
OCR_RENDER_TILE_OVERLAP_PIXELS = 256
OCR_DENSE_VECTOR_RENDER_TILE_MIN_TOKENS = 160
OCR_SCHEMATIC_VECTOR_RENDER_TILE_MIN_TOKENS = 80
OCR_DENSE_VECTOR_RENDER_TILE_MIN_PIXELS = 6_000_000
OCR_DENSE_VECTOR_RENDER_TILE_MAX_SIDE_PIXELS = 2048
OCR_RENDER_TILE_MAX_WORKERS = 4
OCR_TIMEOUT_DISABLED_VALUES = {"", "0", "none", "off", "false", "no"}


class OcrRenderablePage(Protocol):
    extraction_cache: Any

    @property
    def media_box(self) -> tuple[float, float, float, float] | None: ...

    def get_page_profile(self) -> object: ...


@dataclass(frozen=True)
class OcrRenderTile:
    crop: tuple[float, float, float, float]
    core: tuple[float, float, float, float]
    x_offset: int
    y_offset: int
    crop_width: int
    crop_height: int
    page_width: int
    page_height: int
    core_left: int
    core_top: int
    core_right: int
    core_bottom: int
    index: int


@dataclass(frozen=True)
class TileOcrLayoutResult:
    layout: OcrIteratorLayout


@dataclass(frozen=True)
class RenderedTileOcrResult:
    tile: OcrRenderTile
    result: TileOcrLayoutResult


def ocr_timeout_seconds() -> float | None:
    value = os.environ.get("CORE_PDF_OCR_TIMEOUT")
    if value is None or value.strip().casefold() in OCR_TIMEOUT_DISABLED_VALUES:
        return None
    try:
        timeout = float(value)
    except ValueError:
        return None
    if timeout <= 0:
        return None
    return timeout


def ocr_render_dpi_candidates() -> tuple[int, ...]:
    value = os.environ.get("CORE_PDF_OCR_RENDER_DPIS")
    if value is None or not value.strip():
        return OCR_RENDER_DPI_CANDIDATES
    candidates: list[int] = []
    for part in re.split(r"[\s,;:]+", value.strip()):
        if not part:
            continue
        try:
            dpi = int(part)
        except ValueError:
            continue
        if dpi <= 0 or dpi > OCR_RENDER_MAX_DPI:
            continue
        candidates.append(dpi)
    if not candidates:
        return OCR_RENDER_DPI_CANDIDATES
    return tuple(sorted(set(candidates)))


def ocr_render_dpi_candidates_for_vector_text(vector_text: str) -> tuple[int, ...]:
    del vector_text
    return ocr_render_dpi_candidates()


def ocr_render_dpi_candidates_for_page(
    page: OcrRenderablePage,
) -> tuple[int, ...]:
    try:
        profile = page.get_page_profile()
    except Exception:
        return ocr_render_dpi_candidates()
    strategy = profile.recommended_strategy
    if strategy == "native_text":
        # Prefer 300dpi first so strong candidates can short-circuit before we
        # spend time on the 250dpi fallback. Keep the fallback for pages where
        # the higher-resolution pass is still weak.
        return (300, 250)
    if strategy == "text_table":
        return (250, 300)
    if strategy == "vector_or_table":
        return (250, 300)
    if strategy == "image_or_ocr":
        dimensions = ocr_page_dimensions_points(page)
        if dimensions is not None and dimensions[1] > dimensions[0]:
            return (300,)
        return (400, 300)
    return ocr_render_dpi_candidates()


def rendered_page_for_ocr_render(
    page: OcrRenderablePage,
    *,
    dpi: int | None = None,
    source: str = "ocr_render",
    include_text: bool | None = None,
) -> Any:
    from core_ocr.impl.services import get_candidate_services

    if include_text is None:
        cache = page.extraction_cache
        include_text = not bool(
            isinstance(cache, dict) and cache.get("ocr_render_exclude_native_text")
        )
    cache = page.extraction_cache
    cache_key = (
        "ocr_rendered_page_with_annotations_and_layers"
        if include_text
        else "ocr_rendered_page_without_native_text"
    )
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    rendered = get_candidate_services().render_page_for_ocr(page, include_text=include_text)
    if cache is not None:
        cache[cache_key] = rendered
    return rendered


def render_page_for_ocr_at_dpi(
    page: OcrRenderablePage,
    *,
    dpi: int,
    source: str,
    include_text: bool | None = None,
) -> OcrImage | None:
    if include_text is None:
        cache = page.extraction_cache
        include_text = not bool(
            isinstance(cache, dict) and cache.get("ocr_render_exclude_native_text")
        )
    cache = page.extraction_cache
    cache_key = ("ocr_raster_image", dpi, include_text)
    if cache is not None and cache_key in cache:
        cached = cache.get(cache_key)
        if isinstance(cached, OcrImage):
            return replace(cached, source=source)
        return None
    rendered = rendered_page_for_ocr_render(page, dpi=dpi, source=source)
    width_points = max(1.0, float(rendered.width))
    height_points = max(1.0, float(rendered.height))
    scale = dpi / 72.0
    width = max(1, int(round(width_points * scale)))
    height = max(1, int(round(height_points * scale)))
    if not leptonica_pix_size_is_supported(width, height):
        if cache is not None:
            cache[cache_key] = None
        return None
    data = rendered.rasterize(background=(255, 255, 255, 255), scale=scale)
    if rendered.rotate % 180:
        width, height = height, width
    expected_size = width * height * 4
    if len(data) != expected_size:
        if cache is not None:
            cache[cache_key] = None
        return None
    image = OcrImage(
        data=data,
        width=width,
        height=height,
        bytes_per_pixel=4,
        bytes_per_line=width * 4,
        source=source,
        cache_key=cache_key,
        resolution=dpi,
        page_bbox=(0.0, 0.0, width_points, height_points),
    )
    if cache is not None:
        cache[cache_key] = image
    return image


def derived_ocr_image_at_dpi(
    image: OcrImage,
    *,
    dpi: int,
    source: str,
) -> OcrImage | None:
    current_resolution = image.resolution or dpi
    if current_resolution <= 0 or dpi <= 0:
        return None
    if dpi >= current_resolution:
        return None
    current_width = image.target_width or image.width
    current_height = image.target_height or image.height
    if current_width <= 0 or current_height <= 0:
        return None
    scale = dpi / current_resolution
    target_width = max(1, int(round(current_width * scale)))
    target_height = max(1, int(round(current_height * scale)))
    if not leptonica_pix_size_is_supported(target_width, target_height):
        return None
    return replace(
        image,
        source=source,
        target_width=target_width,
        target_height=target_height,
        resolution=dpi,
    )


def ocr_page_dimensions_points(page: OcrRenderablePage) -> tuple[float, float] | None:
    box = _rect_box_tuple(getattr(page, "media_box", None))
    if box is None:
        return None
    x0, y0, x1, y1 = box
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    if width <= 0.0 or height <= 0.0:
        return None
    return (width, height)


def safe_ocr_render_dpi(
    width_points: float,
    height_points: float,
    requested_dpi: int,
) -> int | None:
    if requested_dpi <= 0 or width_points <= 0.0 or height_points <= 0.0:
        return None
    if ocr_render_pixel_size_is_supported(width_points, height_points, requested_dpi):
        return requested_dpi
    page_area_points = width_points * height_points
    max_scale = math.sqrt(
        (LEPTONICA_PIX_MAX_BYTES - 1) / (page_area_points * LEPTONICA_PIX_COLOR_BYTES_PER_PIXEL)
    )
    dpi = min(requested_dpi, int(math.floor(max_scale * 72.0)))
    while dpi > 0:
        if ocr_render_pixel_size_is_supported(width_points, height_points, dpi):
            return dpi
        dpi -= 1
    return None


def ocr_render_pixel_size_is_supported(
    width_points: float,
    height_points: float,
    dpi: int,
) -> bool:
    width, height = ocr_render_pixel_dimensions(width_points, height_points, dpi)
    return leptonica_pix_size_is_supported(width, height)


def ocr_render_pixel_dimensions(
    width_points: float,
    height_points: float,
    dpi: int,
) -> tuple[int, int]:
    scale = dpi / 72.0
    width = max(1, int(round(width_points * scale)))
    height = max(1, int(round(height_points * scale)))
    return (width, height)


def rotated_ocr_pixel_dimensions(
    width: int,
    height: int,
    rotate: int,
) -> tuple[int, int]:
    if rotate % 180:
        return (height, width)
    return (width, height)


def ocr_render_tiles(
    width_points: float,
    height_points: float,
    dpi: int,
    *,
    max_side_pixels: int | None = None,
    overlap_pixels: int | None = None,
) -> list[OcrRenderTile]:
    if width_points <= 0.0 or height_points <= 0.0 or dpi <= 0:
        return []
    scale = dpi / 72.0
    tile_max_side_pixels = max(1, max_side_pixels or OCR_RENDER_TILE_MAX_SIDE_PIXELS)
    tile_overlap_pixels = max(
        0,
        min(
            OCR_RENDER_TILE_OVERLAP_PIXELS if overlap_pixels is None else overlap_pixels,
            tile_max_side_pixels // 4,
        ),
    )
    tile_side_points = tile_max_side_pixels / scale
    overlap_points = tile_overlap_pixels / scale
    core_side_points = max(tile_side_points - overlap_points * 2.0, 1.0 / scale)
    page_width, page_height = ocr_render_pixel_dimensions(
        width_points,
        height_points,
        dpi,
    )
    tiles: list[OcrRenderTile] = []
    y0 = 0.0
    index = 0
    while y0 < height_points - 0.01:
        y1 = min(height_points, y0 + core_side_points)
        x0 = 0.0
        while x0 < width_points - 0.01:
            x1 = min(width_points, x0 + core_side_points)
            crop_x0 = max(0.0, x0 - overlap_points)
            crop_y0 = max(0.0, y0 - overlap_points)
            crop_x1 = min(width_points, x1 + overlap_points)
            crop_y1 = min(height_points, y1 + overlap_points)
            tile_width, tile_height = ocr_render_pixel_dimensions(
                crop_x1 - crop_x0,
                crop_y1 - crop_y0,
                dpi,
            )
            if leptonica_pix_size_is_supported(tile_width, tile_height):
                tiles.append(
                    OcrRenderTile(
                        crop=(crop_x0, crop_y0, crop_x1, crop_y1),
                        core=(x0, y0, x1, y1),
                        x_offset=max(0, int(round(crop_x0 * scale))),
                        y_offset=max(0, int(round((height_points - crop_y1) * scale))),
                        crop_width=tile_width,
                        crop_height=tile_height,
                        page_width=page_width,
                        page_height=page_height,
                        core_left=max(0, int(round(x0 * scale))),
                        core_top=max(0, int(round((height_points - y1) * scale))),
                        core_right=max(0, int(round(x1 * scale))),
                        core_bottom=max(0, int(round((height_points - y0) * scale))),
                        index=index,
                    )
                )
                index += 1
            x0 = x1
        y0 = y1
    return tiles


def ocr_render_tile_worker_count(tile_count: int) -> int:
    if tile_count <= 1:
        return 1
    value = os.environ.get("CORE_PDF_OCR_TILE_WORKERS")
    if value is not None and value.strip():
        try:
            configured = int(value)
        except ValueError:
            configured = 1
        return max(1, min(tile_count, configured))
    return max(1, min(tile_count, os.cpu_count() or 1, OCR_RENDER_TILE_MAX_WORKERS))


def _rect_box_tuple(value: object) -> tuple[float, float, float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        box = cast(Sequence[object], value)
        try:
            return (
                ocr_float_value(box[0]),
                ocr_float_value(box[1]),
                ocr_float_value(box[2]),
                ocr_float_value(box[3]),
            )
        except (TypeError, ValueError):
            return None
    x0 = getattr(value, "x0", None)
    y0 = getattr(value, "y0", None)
    x1 = getattr(value, "x1", None)
    y1 = getattr(value, "y1", None)
    if x0 is None or y0 is None or x1 is None or y1 is None:
        return None
    try:
        return (
            ocr_float_value(x0),
            ocr_float_value(y0),
            ocr_float_value(x1),
            ocr_float_value(y1),
        )
    except (TypeError, ValueError):
        return None
