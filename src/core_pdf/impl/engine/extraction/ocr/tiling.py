# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Protocol

from core_ocr.impl import candidates as ocr_candidates
from core_ocr.impl import text_analysis as ocr_text_analysis
from core_ocr.impl.types import (
    OcrImage,
    OcrIteratorLayout,
    OcrRow,
    OcrTextResult,
    leptonica_pix_size_is_supported,
    ocr_int_value,
)

from core_pdf.impl.engine.extraction.ocr import (
    candidate_generation as ocr_candidate_generation,
)
from core_pdf.impl.engine.extraction.ocr import execution as ocr_execution
from core_pdf.impl.engine.extraction.ocr import iterator_layout as ocr_iterator_layout
from core_pdf.impl.engine.extraction.ocr import rendering as ocr_rendering
from core_pdf.impl.engine.rendering.models import RenderedPage

OCR_TILE_PAGE_SEGMENTATION_MODE = 6
OCR_TILE_SPARSE_PAGE_SEGMENTATION_MODE = 11


class LayoutRepairFunction(Protocol):
    def __call__(
        self,
        layout: OcrIteratorLayout,
        support_text: str,
    ) -> tuple[OcrIteratorLayout, int]: ...


class TokenTypeClassifier(Protocol):
    def __call__(self, text: str, /) -> str | None: ...


OcrCandidate = ocr_candidates.OcrCandidate


def tiled_ocr_candidate_for_dpi(
    page: ocr_rendering.OcrRenderablePage,
    dpi: int,
    timeout: float | None,
    *,
    max_side_pixels: int | None = None,
    support_text: str = "",
    layout_repair: LayoutRepairFunction | None = None,
    token_type_classifier: TokenTypeClassifier | None = None,
) -> OcrCandidate | None:
    rendered = ocr_rendering.rendered_page_for_ocr_render(
        page,
        dpi=dpi,
        source=f"rendered_page_{dpi}dpi_tiled",
    )
    return tiled_ocr_candidate_from_rendered_page(
        rendered,
        dpi,
        timeout,
        max_side_pixels=max_side_pixels,
        support_text=support_text,
        layout_repair=layout_repair,
        token_type_classifier=token_type_classifier,
    )


def tiled_ocr_candidate_from_rendered_page(
    rendered: RenderedPage,
    dpi: int,
    timeout: float | None,
    *,
    max_side_pixels: int | None = None,
    support_text: str = "",
    layout_repair: LayoutRepairFunction | None = None,
    token_type_classifier: TokenTypeClassifier | None = None,
) -> OcrCandidate | None:
    width_points = max(1.0, float(rendered.width))
    height_points = max(1.0, float(rendered.height))
    tiles = ocr_rendering.ocr_render_tiles(
        width_points,
        height_points,
        dpi,
        max_side_pixels=max_side_pixels,
    )
    if len(tiles) < 2:
        return None
    line_rows: list[OcrRow] = []
    word_rows: list[OcrRow] = []
    symbol_rows: list[OcrRow] = []
    for tile_ocr in collect_rendered_tile_ocr_results(rendered, tiles, dpi, timeout):
        tile = tile_ocr.tile
        tile_result = tile_ocr.result
        shifted_layout = offset_tile_iterator_layout(tile_result.layout, tile, rendered.rotate)
        line_rows.extend(shifted_layout.textline_rows)
        word_rows.extend(shifted_layout.word_rows)
        symbol_rows.extend(shifted_layout.symbol_rows)
    layout = ocr_iterator_layout.deduplicate_tile_iterator_layout(
        OcrIteratorLayout(line_rows, word_rows, symbol_rows)
    )
    layout, removed_rows = ocr_iterator_layout.filter_unsupported_low_confidence_tile_lines(
        layout,
        support_text,
    )
    layout, repaired_rows = (
        layout_repair(layout, support_text) if layout_repair is not None else (layout, 0)
    )
    result = ocr_iterator_layout.iterator_tile_layout_text_result(layout)
    if not result.text:
        return None
    image_width, image_height = ocr_rendering.rotated_ocr_pixel_dimensions(
        tiles[0].page_width,
        tiles[0].page_height,
        rendered.rotate,
    )
    page_width = float(image_width) * 72.0 / float(dpi)
    page_height = float(image_height) * 72.0 / float(dpi)
    result = ocr_candidate_generation.ocr_result_with_page_observations(
        result,
        source=f"rendered_page_{dpi}dpi_tiled",
        image_width=image_width,
        image_height=image_height,
        image_resolution=dpi,
        page_width=page_width,
        page_height=page_height,
        token_type_classifier=token_type_classifier,
    )
    return ocr_candidates.OcrCandidate(
        f"rendered_page_{dpi}dpi_tiled",
        result,
        region_count=len(tiles),
        image_width=image_width,
        image_height=image_height,
        image_resolution=dpi,
        page_width=page_width,
        page_height=page_height,
    )


def collect_rendered_tile_ocr_results(
    rendered: RenderedPage,
    tiles: list[ocr_rendering.OcrRenderTile],
    dpi: int,
    timeout: float | None,
) -> list[ocr_rendering.RenderedTileOcrResult]:
    full_page_image = render_full_page_for_tiled_ocr(rendered, tiles, dpi)
    results: list[ocr_rendering.RenderedTileOcrResult] = []
    for tile in tiles:
        try:
            results.append(
                rendered_tile_ocr_layout_result(
                    rendered,
                    tile,
                    dpi,
                    timeout,
                    full_page_image=full_page_image,
                )
            )
        except BaseException:
            results.append(
                ocr_rendering.RenderedTileOcrResult(
                    tile,
                    ocr_rendering.TileOcrLayoutResult(
                        OcrIteratorLayout([], [], []),
                    ),
                )
            )
    return results


def rendered_tile_ocr_layout_result(
    rendered: RenderedPage,
    tile: ocr_rendering.OcrRenderTile,
    dpi: int,
    timeout: float | None,
    *,
    full_page_image: OcrImage | None = None,
) -> ocr_rendering.RenderedTileOcrResult:
    source = f"rendered_page_{dpi}dpi_tile_{tile.index}"
    image = (
        crop_ocr_tile_from_full_page_image(
            full_page_image,
            tile,
            source=source,
        )
        if full_page_image is not None
        else None
    )
    if image is None:
        image = render_ocr_tile_from_rendered_page(
            rendered,
            tile,
            dpi,
            source=source,
        )
    if image is None:
        return ocr_rendering.RenderedTileOcrResult(
            tile,
            ocr_rendering.TileOcrLayoutResult(
                OcrIteratorLayout([], [], []),
            ),
        )
    result = ocr_image_to_tile_iterator_result_with_timeout(image, timeout)
    return ocr_rendering.RenderedTileOcrResult(tile, result)


def render_full_page_for_tiled_ocr(
    rendered: RenderedPage,
    tiles: list[ocr_rendering.OcrRenderTile],
    dpi: int,
) -> OcrImage | None:
    if not tiles or rendered.rotate % 360 != 0:
        return None
    width_points = max(1.0, float(rendered.width))
    height_points = max(1.0, float(rendered.height))
    width, height = ocr_rendering.ocr_render_pixel_dimensions(
        width_points,
        height_points,
        dpi,
    )
    if not leptonica_pix_size_is_supported(width, height):
        return None
    data = rendered.rasterize(background=(255, 255, 255, 255), scale=dpi / 72.0)
    expected_size = width * height * 4
    if len(data) != expected_size:
        return None
    return OcrImage(
        data=data,
        width=width,
        height=height,
        bytes_per_pixel=4,
        bytes_per_line=width * 4,
        source=f"rendered_page_{dpi}dpi_tiled_full",
        resolution=dpi,
    )


def crop_ocr_tile_from_full_page_image(
    image: OcrImage,
    tile: ocr_rendering.OcrRenderTile,
    *,
    source: str,
) -> OcrImage | None:
    if (
        image.bytes_per_pixel != 4
        or tile.crop_width <= 0
        or tile.crop_height <= 0
        or tile.x_offset < 0
        or tile.y_offset < 0
        or tile.x_offset + tile.crop_width > image.width
        or tile.y_offset + tile.crop_height > image.height
    ):
        return None
    row_bytes = tile.crop_width * 4
    if row_bytes <= 0:
        return None
    data = bytearray(row_bytes * tile.crop_height)
    source_stride = image.bytes_per_line
    source_start = tile.y_offset * source_stride + tile.x_offset * 4
    target_offset = 0
    for row in range(tile.crop_height):
        source_offset = source_start + row * source_stride
        data[target_offset : target_offset + row_bytes] = image.data[
            source_offset : source_offset + row_bytes
        ]
        target_offset += row_bytes
    return OcrImage(
        data=bytes(data),
        width=tile.crop_width,
        height=tile.crop_height,
        bytes_per_pixel=4,
        bytes_per_line=row_bytes,
        source=source,
        resolution=image.resolution,
    )


def render_ocr_tile_from_rendered_page(
    rendered: RenderedPage,
    tile: ocr_rendering.OcrRenderTile,
    dpi: int,
    *,
    source: str,
) -> OcrImage | None:
    crop_x0, crop_y0, crop_x1, crop_y1 = tile.crop
    width_points = max(0.0, crop_x1 - crop_x0)
    height_points = max(0.0, crop_y1 - crop_y0)
    width, height = ocr_rendering.rotated_ocr_pixel_dimensions(
        *ocr_rendering.ocr_render_pixel_dimensions(width_points, height_points, dpi),
        rendered.rotate,
    )
    if not leptonica_pix_size_is_supported(width, height):
        return None
    tile_rendered = RenderedPage(
        page_number=rendered.page_number,
        width=rendered.width,
        height=rendered.height,
        rotate=rendered.rotate,
        display_list=rendered.display_list,
        metadata={**rendered.metadata, "crop": tile.crop},
    )
    data = tile_rendered.rasterize(
        background=(255, 255, 255, 255),
        scale=dpi / 72.0,
    )
    expected_size = width * height * 4
    if len(data) != expected_size:
        return None
    return OcrImage(
        data=data,
        width=width,
        height=height,
        bytes_per_pixel=4,
        bytes_per_line=width * 4,
        source=source,
        resolution=dpi,
    )


def ocr_image_to_tile_iterator_result_with_timeout(
    image: OcrImage,
    timeout: float | None,
) -> ocr_rendering.TileOcrLayoutResult:
    try:
        return select_tile_iterator_layout_from_image(image, timeout)
    except BaseException:
        return empty_tile_ocr_layout_result()


def select_tile_iterator_layout_from_image(
    image: OcrImage,
    timeout: float | None,
) -> ocr_rendering.TileOcrLayoutResult:
    primary = ocr_execution.ocr_image_to_iterator_layout_with_timeout(
        image,
        psm=OCR_TILE_PAGE_SEGMENTATION_MODE,
        timeout=timeout,
    )
    candidates = [primary]
    psms = [OCR_TILE_PAGE_SEGMENTATION_MODE]
    primary_result = ocr_iterator_layout.iterator_layout_text_result(primary)
    if should_try_sparse_tile_layout_ocr(primary_result):
        candidates.append(
            ocr_execution.ocr_image_to_iterator_layout_with_timeout(
                image,
                psm=OCR_TILE_SPARSE_PAGE_SEGMENTATION_MODE,
                timeout=timeout,
            )
        )
        psms.append(OCR_TILE_SPARSE_PAGE_SEGMENTATION_MODE)
        candidates.append(
            ocr_execution.ocr_image_to_iterator_layout_with_timeout(
                image,
                psm=1,
                timeout=timeout,
            )
        )
        psms.append(1)
    return select_tile_iterator_layout(list(zip(psms, candidates, strict=True)))


def empty_tile_ocr_layout_result() -> ocr_rendering.TileOcrLayoutResult:
    return ocr_rendering.TileOcrLayoutResult(OcrIteratorLayout([], [], []))


def should_try_sparse_tile_layout_ocr(result: OcrTextResult) -> bool:
    tokens = ocr_text_analysis.extracted_text_token_count(result.text)
    if tokens < 40:
        return True
    confidence = result.confidence
    if confidence is not None and confidence < 62:
        return True
    return ocr_text_analysis.text_ocr_quality_score(result.text) > 0.20


def select_tile_iterator_layout(
    candidates: list[tuple[int, OcrIteratorLayout]],
) -> ocr_rendering.TileOcrLayoutResult:
    best = empty_tile_ocr_layout_result()
    best_score = float("-inf")
    for psm, layout in candidates:
        result = ocr_iterator_layout.iterator_tile_layout_text_result(layout)
        score = tile_layout_result_score(result)
        if score > best_score:
            best_score = score
            best = ocr_rendering.TileOcrLayoutResult(
                layout,
            )
    return best


def tile_layout_result_score(result: OcrTextResult) -> float:
    if not result.text:
        return float("-inf")
    confidence = result.confidence if result.confidence is not None else 50
    tokens = ocr_text_analysis.extracted_text_token_count(result.text)
    quality = ocr_text_analysis.text_ocr_quality_score(result.text)
    score = float(confidence)
    score += min(tokens, 300) * 0.14
    score -= quality * 20.0
    noise = ocr_text_analysis.uninterpretable_char_count(result.text)
    if noise:
        score -= min(30.0, noise * 2.0)
    return score


def offset_tile_iterator_rows(
    rows: list[OcrRow],
    tile: ocr_rendering.OcrRenderTile,
    rotate: int = 0,
) -> list[OcrRow]:
    shifted: list[OcrRow] = []
    for row in rows:
        tile_left = ocr_int_value(row["left"])
        tile_top = ocr_int_value(row["top"])
        width = ocr_int_value(row["width"])
        height = ocr_int_value(row["height"])
        local_bbox = unrotate_tile_ocr_bbox(
            tile_left,
            tile_top,
            tile_left + width,
            tile_top + height,
            tile.crop_width,
            tile.crop_height,
            rotate,
        )
        if local_bbox is None:
            continue
        local_x0, local_y0, local_x1, local_y1 = local_bbox
        full_x0 = local_x0 + tile.x_offset
        full_y0 = local_y0 + tile.y_offset
        full_x1 = local_x1 + tile.x_offset
        full_y1 = local_y1 + tile.y_offset
        center_x = (full_x0 + full_x1) * 0.5
        center_y = (full_y0 + full_y1) * 0.5
        if (
            center_x < tile.core_left
            or center_x >= tile.core_right
            or center_y < tile.core_top
            or center_y >= tile.core_bottom
        ):
            continue
        rotated_bbox = rotate_full_ocr_bbox(
            full_x0,
            full_y0,
            full_x1,
            full_y1,
            tile.page_width,
            tile.page_height,
            rotate,
        )
        if rotated_bbox is None:
            continue
        left, top, right, bottom = rotated_bbox
        shifted_row: OcrRow = dict(row)
        shifted_row["page_num"] = 1
        shifted_row["tile_index"] = tile.index
        shifted_row["block_num"] = tile.index * 10000 + ocr_int_value(row.get("block_num", 1))
        shifted_row["par_num"] = ocr_int_value(row.get("par_num", 1))
        shifted_row["line_num"] = ocr_int_value(row.get("line_num", len(shifted) + 1))
        if "word_num" in row:
            shifted_row["word_num"] = ocr_int_value(row["word_num"])
        if "symbol_num" in row:
            shifted_row["symbol_num"] = ocr_int_value(row["symbol_num"])
        shifted_row["left"] = left
        shifted_row["top"] = top
        shifted_row["width"] = max(1, right - left)
        shifted_row["height"] = max(1, bottom - top)
        shifted.append(shifted_row)
    return shifted


def offset_tile_iterator_layout(
    layout: OcrIteratorLayout,
    tile: ocr_rendering.OcrRenderTile,
    rotate: int = 0,
) -> OcrIteratorLayout:
    return OcrIteratorLayout(
        offset_tile_iterator_rows(layout.textline_rows, tile, rotate),
        offset_tile_iterator_rows(layout.word_rows, tile, rotate),
        offset_tile_iterator_rows(layout.symbol_rows, tile, rotate),
    )


def unrotate_tile_ocr_bbox(
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
    rotate: int,
) -> tuple[int, int, int, int] | None:
    rotate %= 360
    if rotate == 0:
        bbox = (left, top, right, bottom)
    elif rotate == 90:
        bbox = (top, height - right, bottom, height - left)
    elif rotate == 180:
        bbox = (width - right, height - bottom, width - left, height - top)
    elif rotate == 270:
        bbox = (width - bottom, left, width - top, right)
    else:
        bbox = (left, top, right, bottom)
    return clamp_ocr_bbox(*bbox, width, height)


def rotate_full_ocr_bbox(
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
    rotate: int,
) -> tuple[int, int, int, int] | None:
    rotate %= 360
    if rotate == 0:
        bbox = (left, top, right, bottom)
        bbox_width, bbox_height = width, height
    elif rotate == 90:
        bbox = (height - bottom, left, height - top, right)
        bbox_width, bbox_height = height, width
    elif rotate == 180:
        bbox = (width - right, height - bottom, width - left, height - top)
        bbox_width, bbox_height = width, height
    elif rotate == 270:
        bbox = (top, width - right, bottom, width - left)
        bbox_width, bbox_height = height, width
    else:
        bbox = (left, top, right, bottom)
        bbox_width, bbox_height = width, height
    return clamp_ocr_bbox(*bbox, bbox_width, bbox_height)


def clamp_ocr_bbox(
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    left = max(0, min(width, left))
    right = max(left, min(width, right))
    top = max(0, min(height, top))
    bottom = max(top, min(height, bottom))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)
