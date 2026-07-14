# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Mapping

from core_pdf.impl.engine.extraction.cache import ExtractionCacheKey
from core_pdf.impl.engine.extraction.common import page_geometry
from core_pdf.impl.engine.extraction.ocr import tiling as ocr_tiling
from core_pdf.impl.engine.extraction.ocr.text_analysis import (
    extracted_text_token_count,
    text_ocr_quality_score,
)
from core_pdf.impl.engine.extraction.ocr.types import (
    OcrComponentBox,
    OcrImage,
    OcrIteratorLayout,
    OcrTextResult,
)
from core_pdf.impl.engine.extraction.ocr.backend import TesseractCtypesBackend

OCR_DEFAULT_DPI = 300
OCR_DEFAULT_PAGE_SEGMENTATION_MODE = 6
OcrVariables = Mapping[str, str | int | float | bool] | None
OcrComponentBoxCache = MutableMapping[ExtractionCacheKey, object]


@dataclass(frozen=True)
class RectangleOcrRequest:
    rectangle: tuple[int, int, int, int]
    psm: int
    variables: OcrVariables = None
    rotate_vertical: bool = False


def rotate_ocr_image_right_angle(image: OcrImage, *, clockwise: bool) -> OcrImage:
    bpp = image.bytes_per_pixel
    width = image.width
    height = image.height
    rotated_width = height
    rotated_height = width
    turns = 1 if clockwise else 3
    if image.encoded is not None:
        return OcrImage(
            data=image.data,
            width=rotated_width,
            height=rotated_height,
            bytes_per_pixel=bpp,
            bytes_per_line=image.bytes_per_line,
            encoded=image.encoded,
            source=(
                f"{image.source}_rotated_clockwise"
                if clockwise
                else f"{image.source}_rotated_counterclockwise"
            ),
            cache_key=image.cache_key,
            target_width=image.target_height,
            target_height=image.target_width,
            resolution=image.resolution,
            clockwise_quarter_turns=(image.clockwise_quarter_turns + turns) % 4,
            page_bbox=image.page_bbox,
            page_clockwise_quarter_turns=(image.page_clockwise_quarter_turns + turns)
            % 4,
        )
    row_bytes = rotated_width * bpp
    required_size = (height - 1) * image.bytes_per_line + width * bpp
    if bpp <= 0 or width <= 0 or height <= 0 or len(image.data) < required_size:
        return image
    data = bytearray(row_bytes * rotated_height)
    if bpp == 1:
        source_stop = required_size
        if clockwise:
            last_row = (height - 1) * image.bytes_per_line
            for x in range(width):
                dst = x * row_bytes
                data[dst : dst + row_bytes] = image.data[
                    last_row + x :: -image.bytes_per_line
                ][:height]
        else:
            for rotated_y in range(rotated_height):
                x = width - 1 - rotated_y
                dst = rotated_y * row_bytes
                data[dst : dst + row_bytes] = image.data[
                    x : source_stop : image.bytes_per_line
                ][:height]
    elif bpp == 4 and image.bytes_per_line == width * 4:
        source_pixels = memoryview(image.data)[:required_size].cast("I")
        rotated_pixels = memoryview(data).cast("I")
        if clockwise:
            for x in range(width):
                start = x * height
                rotated_pixels[start : start + height] = source_pixels[x::width][::-1]
        else:
            for rotated_y in range(rotated_height):
                x = width - 1 - rotated_y
                start = rotated_y * height
                rotated_pixels[start : start + height] = source_pixels[x::width]
    else:
        source = image.data
        source_stride = image.bytes_per_line
        if bpp == 3:
            if clockwise:
                for x in range(width):
                    dst = x * row_bytes
                    src = (height - 1) * source_stride + x * 3
                    for _ in range(height):
                        data[dst] = source[src]
                        data[dst + 1] = source[src + 1]
                        data[dst + 2] = source[src + 2]
                        dst += 3
                        src -= source_stride
            else:
                for rotated_y in range(rotated_height):
                    dst = rotated_y * row_bytes
                    src = (width - 1 - rotated_y) * 3
                    for _ in range(height):
                        data[dst] = source[src]
                        data[dst + 1] = source[src + 1]
                        data[dst + 2] = source[src + 2]
                        dst += 3
                        src += source_stride
        elif bpp == 4:
            if clockwise:
                for x in range(width):
                    dst = x * row_bytes
                    src = (height - 1) * source_stride + x * 4
                    for _ in range(height):
                        data[dst] = source[src]
                        data[dst + 1] = source[src + 1]
                        data[dst + 2] = source[src + 2]
                        data[dst + 3] = source[src + 3]
                        dst += 4
                        src -= source_stride
            else:
                for rotated_y in range(rotated_height):
                    dst = rotated_y * row_bytes
                    src = (width - 1 - rotated_y) * 4
                    for _ in range(height):
                        data[dst] = source[src]
                        data[dst + 1] = source[src + 1]
                        data[dst + 2] = source[src + 2]
                        data[dst + 3] = source[src + 3]
                        dst += 4
                        src += source_stride
        elif clockwise:
            for x in range(width):
                dst = x * row_bytes
                src = (height - 1) * source_stride + x * bpp
                for _ in range(height):
                    data[dst : dst + bpp] = source[src : src + bpp]
                    dst += bpp
                    src -= source_stride
        else:
            for rotated_y in range(rotated_height):
                dst = rotated_y * row_bytes
                src = (width - 1 - rotated_y) * bpp
                for _ in range(height):
                    data[dst : dst + bpp] = source[src : src + bpp]
                    dst += bpp
                    src += source_stride
    return OcrImage(
        bytes(data),
        rotated_width,
        rotated_height,
        bpp,
        rotated_width * bpp,
        source=(
            f"{image.source}_rotated_clockwise"
            if clockwise
            else f"{image.source}_rotated_counterclockwise"
        ),
        cache_key=image.cache_key,
        target_width=image.target_height,
        target_height=image.target_width,
        resolution=image.resolution,
        clockwise_quarter_turns=image.clockwise_quarter_turns,
        page_bbox=image.page_bbox,
        page_clockwise_quarter_turns=(image.page_clockwise_quarter_turns + turns) % 4,
    )


def rotate_ocr_image_half_turn(image: OcrImage) -> OcrImage:
    if image.encoded is not None:
        return OcrImage(
            data=image.data,
            width=image.width,
            height=image.height,
            bytes_per_pixel=image.bytes_per_pixel,
            bytes_per_line=image.bytes_per_line,
            encoded=image.encoded,
            source=f"{image.source}_rotated_180",
            cache_key=image.cache_key,
            target_width=image.target_width,
            target_height=image.target_height,
            resolution=image.resolution,
            clockwise_quarter_turns=(image.clockwise_quarter_turns + 2) % 4,
            page_bbox=image.page_bbox,
            page_clockwise_quarter_turns=(image.page_clockwise_quarter_turns + 2) % 4,
        )
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return image
    bpp = image.bytes_per_pixel
    width = image.width
    height = image.height
    data = bytearray(width * height * bpp)
    for y in range(height):
        for x in range(width):
            src = y * image.bytes_per_line + x * bpp
            if src + bpp > len(image.data):
                return image
            dst_x = width - 1 - x
            dst_y = height - 1 - y
            dst = (dst_y * width + dst_x) * bpp
            data[dst : dst + bpp] = image.data[src : src + bpp]
    return OcrImage(
        bytes(data),
        width,
        height,
        bpp,
        width * bpp,
        source=f"{image.source}_rotated_180",
        cache_key=image.cache_key,
        target_width=image.target_width,
        target_height=image.target_height,
        resolution=image.resolution,
        clockwise_quarter_turns=image.clockwise_quarter_turns,
        page_bbox=image.page_bbox,
        page_clockwise_quarter_turns=(image.page_clockwise_quarter_turns + 2) % 4,
    )


def ocr_image_to_text_result_with_psm_timeout(
    image: OcrImage,
    *,
    psm: int,
    timeout: float | None,
    variables: OcrVariables = None,
) -> OcrTextResult:
    del timeout
    backend = TesseractCtypesBackend.from_system()
    if backend is None:
        return OcrTextResult("", None)
    try:
        return backend.image_to_text_result(
            image,
            psm=psm,
            resolution=image.resolution or OCR_DEFAULT_DPI,
            variables=variables,
        )
    except BaseException:
        return OcrTextResult("", None)


def ocr_image_to_text_results_with_psms_timeout(
    image: OcrImage,
    *,
    psms: list[int],
    timeout: float | None,
    variables: OcrVariables = None,
) -> list[OcrTextResult]:
    del timeout
    if not psms:
        return []
    backend = TesseractCtypesBackend.from_system()
    if backend is None:
        return [OcrTextResult("", None) for _ignored in psms]
    try:
        return backend.image_to_text_results(
            image,
            psms,
            resolution=image.resolution or OCR_DEFAULT_DPI,
            variables=variables,
        )
    except BaseException:
        return [OcrTextResult("", None) for _ignored in psms]


def ocr_image_region_to_text_result_with_timeout(
    image: OcrImage,
    rectangle: tuple[int, int, int, int],
    *,
    psm: int,
    variables: OcrVariables,
    timeout: float | None,
) -> OcrTextResult:
    del timeout
    backend = TesseractCtypesBackend.from_system()
    if backend is None:
        return OcrTextResult("", None)
    try:
        return backend.image_region_to_text_result(
            image,
            rectangle_for_backend_image(image, rectangle),
            psm=psm,
            resolution=image.resolution or OCR_DEFAULT_DPI,
            variables=variables,
        )
    except BaseException:
        return OcrTextResult("", None)


def ocr_image_regions_to_text_results_with_timeout(
    image: OcrImage,
    requests: list[RectangleOcrRequest],
    timeout: float | None,
) -> list[OcrTextResult]:
    del timeout
    if not requests:
        return []
    backend = TesseractCtypesBackend.from_system()
    if backend is None:
        return [OcrTextResult("", None) for _ignored in requests]
    try:
        resolution = image.resolution or OCR_DEFAULT_DPI
        results = [OcrTextResult("", None) for _ignored in requests]
        native_indexes: list[int] = []
        native_requests: list[tuple[tuple[int, int, int, int], int, OcrVariables]] = []
        for index, request in enumerate(requests):
            if not request.rotate_vertical:
                native_indexes.append(index)
                native_requests.append(
                    (
                        rectangle_for_backend_image(image, request.rectangle),
                        request.psm,
                        request.variables,
                    )
                )
        if native_requests:
            native_results = backend.image_regions_to_text_results(
                image,
                native_requests,
                resolution,
            )
            for index, result in zip(native_indexes, native_results, strict=False):
                results[index] = result
        for index, request in enumerate(requests):
            if request.rotate_vertical:
                results[index] = ocr_rotated_rectangle_result(
                    backend,
                    image,
                    request,
                    resolution=resolution,
                )
        return results
    except BaseException:
        return [OcrTextResult("", None) for _ignored in requests]


def rectangle_for_backend_image(
    image: OcrImage,
    rectangle: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Map source-image coordinates to the scaled Leptonica image."""
    source_width = max(1, image.width)
    source_height = max(1, image.height)
    target_width = image.target_width or source_width
    target_height = image.target_height or source_height
    x0, y0, x1, y1 = rectangle
    x0 = max(0, min(source_width, x0))
    y0 = max(0, min(source_height, y0))
    x1 = max(x0, min(source_width, x1))
    y1 = max(y0, min(source_height, y1))
    return tuple(
        max(0, min(target, int(round(value * target / source))))
        for value, target, source in (
            (x0, target_width, source_width),
            (y0, target_height, source_height),
            (x1, target_width, source_width),
            (y1, target_height, source_height),
        )
    )  # type: ignore[return-value]


def ocr_image_to_iterator_layout_with_timeout(
    image: OcrImage,
    *,
    psm: int,
    timeout: float | None,
    variables: OcrVariables = None,
) -> OcrIteratorLayout:
    del timeout
    backend = TesseractCtypesBackend.from_system()
    if backend is None:
        return OcrIteratorLayout([], [], [])
    try:
        return backend.image_to_iterator_layout(
            image,
            psm=psm,
            resolution=image.resolution or OCR_DEFAULT_DPI,
            variables=variables,
        )
    except BaseException:
        return OcrIteratorLayout([], [], [])


def ocr_component_boxes_cache_key(
    image: OcrImage,
    level: int,
    *,
    psm: int,
    variables: OcrVariables,
) -> tuple[Any, ...]:
    return (
        id(image),
        image.source,
        image.width,
        image.height,
        image.bytes_per_pixel,
        image.bytes_per_line,
        image.resolution,
        image.clockwise_quarter_turns,
        image.page_clockwise_quarter_turns,
        level,
        psm,
        normalized_ocr_variables_cache_key(variables),
    )


def normalized_ocr_variables_cache_key(
    variables: OcrVariables,
) -> tuple[tuple[str, str], ...]:
    if not variables:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in variables.items()))


def ocr_component_boxes_with_timeout(
    image: OcrImage,
    level: int,
    timeout: float | None,
    *,
    psm: int = OCR_DEFAULT_PAGE_SEGMENTATION_MODE,
    variables: OcrVariables = None,
    cache: OcrComponentBoxCache | None = None,
) -> list[OcrComponentBox]:
    if cache is not None:
        cache_key = ocr_component_boxes_cache_key(
            image,
            level,
            psm=psm,
            variables=variables,
        )
        cached = cache.get(cache_key)
        if isinstance(cached, tuple):
            return list(cached)
    boxes = _ocr_component_boxes_with_timeout_uncached(
        image,
        level,
        timeout,
        psm=psm,
        variables=variables,
    )
    if cache is not None:
        cache[cache_key] = tuple(boxes)
    return boxes


def _ocr_component_boxes_with_timeout_uncached(
    image: OcrImage,
    level: int,
    timeout: float | None,
    *,
    psm: int = OCR_DEFAULT_PAGE_SEGMENTATION_MODE,
    variables: OcrVariables = None,
) -> list[OcrComponentBox]:
    del timeout
    backend = TesseractCtypesBackend.from_system()
    if backend is None:
        return []
    try:
        return backend.image_to_component_boxes(
            image,
            psm=psm,
            resolution=image.resolution or OCR_DEFAULT_DPI,
            level=level,
            text_only=True,
            variables=variables,
        )
    except BaseException:
        return []


def ocr_rotated_rectangle_result(
    backend: TesseractCtypesBackend,
    image: OcrImage,
    request: RectangleOcrRequest,
    *,
    resolution: int,
) -> OcrTextResult:
    region = crop_ocr_image_region(image, request.rectangle)
    if region is None:
        return OcrTextResult("", None)
    clockwise = rotate_ocr_image_right_angle(region, clockwise=True)
    counter_clockwise = rotate_ocr_image_right_angle(region, clockwise=False)
    clockwise_result = backend.image_to_text_result(
        clockwise,
        psm=request.psm,
        resolution=resolution,
        variables=request.variables,
    )
    counter_clockwise_result = backend.image_to_text_result(
        counter_clockwise,
        psm=request.psm,
        resolution=resolution,
        variables=request.variables,
    )
    return select_vertical_rectangle_ocr_result(
        clockwise_result,
        counter_clockwise_result,
    )


def select_vertical_rectangle_ocr_result(
    first: OcrTextResult,
    second: OcrTextResult,
) -> OcrTextResult:
    if not first.text:
        return second
    if not second.text:
        return first
    first_tokens = extracted_text_token_count(first.text)
    second_tokens = extracted_text_token_count(second.text)
    first_confidence = first.confidence if first.confidence is not None else 50
    second_confidence = second.confidence if second.confidence is not None else 50
    first_quality = text_ocr_quality_score(first.text)
    second_quality = text_ocr_quality_score(second.text)
    first_score = first_confidence + min(first_tokens, 20) * 1.5 - first_quality * 40.0
    second_score = (
        second_confidence + min(second_tokens, 20) * 1.5 - second_quality * 40.0
    )
    return second if second_score > first_score else first


def crop_ocr_image_region(
    image: OcrImage,
    rectangle: tuple[int, int, int, int],
) -> OcrImage | None:
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return None
    clamped = ocr_tiling.clamp_ocr_bbox(*rectangle, image.width, image.height)
    if clamped is None:
        return None
    left, top, right, bottom = clamped
    width = right - left
    height = bottom - top
    bpp = image.bytes_per_pixel
    row_bytes = width * bpp
    data = bytearray(row_bytes * height)
    for y in range(height):
        src = (top + y) * image.bytes_per_line + left * bpp
        dst = y * row_bytes
        data[dst : dst + row_bytes] = image.data[src : src + row_bytes]
    geometry = page_geometry.ImageSpace.from_dimensions(
        image_width=image.width,
        image_height=image.height,
        image_resolution=image.resolution,
        page_bbox=image.page_bbox,
        clockwise_quarter_turns=image.page_clockwise_quarter_turns,
        source=image.source,
    )
    return OcrImage(
        bytes(data),
        width,
        height,
        bpp,
        row_bytes,
        source=f"{image.source}_region",
        resolution=image.resolution,
        page_bbox=page_geometry.image_bbox_to_page_bbox(clamped, geometry),
        page_clockwise_quarter_turns=image.page_clockwise_quarter_turns,
    )
