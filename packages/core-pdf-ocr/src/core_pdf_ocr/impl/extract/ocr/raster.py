# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, ClassVar

import numpy

from core_pdf.impl.array_views import (
    contiguous_bytes,
    uint8_image_view,
)
from core_pdf.impl.capture_records import CapturedDrawing
from core_pdf.impl.extract_contracts import FULL_PAGE_IMAGE_COVERAGE
from core_pdf.impl.geometry import bbox_union
from core_pdf.impl.graphics_images import decode_image, decode_pdf_image
from core_pdf.impl.render_model import RasterImage
from core_pdf.impl.types import Record, frozen_setattr
from core_pdf_ocr.impl.extract.contracts import MAX_OCR_PIXELS, PageAnalysis
from core_pdf_ocr.impl.extract.ocr.resampling import resample_bilinear, resample_nearest
from core_pdf_ocr.impl.extract.ocr.types import Raster

DIRECT_OCR_TARGET_RESOLUTION = 400


DIRECT_OCR_MIN_UPSCALE = 1.05


DIRECT_OCR_WHOLE_SCALE_TOLERANCE = 0.06


def visible_intensity(samples: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
    channels = samples.shape[2]
    if channels == 1:
        return samples[:, :, 0]
    if channels in {2, 4}:
        intensity = numpy.min(samples[:, :, :-1], axis=2).astype(numpy.uint16)
        alpha = samples[:, :, -1].astype(numpy.uint16)
        return (255 - ((255 - intensity) * alpha + 127) // 255).astype(numpy.uint8)
    return numpy.min(samples, axis=2)


def raster_ink_grid(raster: Raster, rows: int, columns: int) -> numpy.ndarray[Any, Any]:
    if rows <= 0 or columns <= 0:
        return numpy.zeros(max(0, rows * columns), dtype=numpy.float32)
    pixels = raster.image.array()
    y_step = max(1, raster.height // 512)
    x_step = max(1, raster.width // 512)
    sampled = pixels[::y_step, ::x_step]
    intensity = visible_intensity(sampled)
    ink = intensity < 245
    integral = numpy.pad(
        ink.cumsum(axis=0, dtype=numpy.int32).cumsum(axis=1, dtype=numpy.int32),
        ((1, 0), (1, 0)),
    )
    y_bounds = numpy.arange(rows + 1, dtype=numpy.intp) * len(ink) // rows
    x_bounds = numpy.arange(columns + 1, dtype=numpy.intp) * ink.shape[1] // columns
    y0, y1 = y_bounds[:-1], y_bounds[1:]
    x0, x1 = x_bounds[:-1], x_bounds[1:]
    sums = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    counts = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    grid_output = numpy.zeros((rows, columns), dtype=numpy.float32)
    numpy.divide(sums, counts, out=grid_output, where=counts != 0)
    return grid_output.reshape(-1)


LUMA_RED = 77
LUMA_GREEN = 150
LUMA_BLUE = 29


def luma(samples: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
    gray = samples[:, :, 0].astype(numpy.uint16)
    gray *= LUMA_RED
    channel = samples[:, :, 1].astype(numpy.uint16)
    channel *= LUMA_GREEN
    gray += channel
    channel = samples[:, :, 2].astype(numpy.uint16)
    channel *= LUMA_BLUE
    gray += channel
    gray += 128
    gray >>= 8
    return gray.astype(numpy.uint8)


def flatten_onto_white(samples: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
    alpha = samples[:, :, 3].astype(numpy.uint16)
    flattened = numpy.empty(samples.shape[:2] + (3,), dtype=numpy.uint8)
    for index in range(3):
        blended = samples[:, :, index].astype(numpy.uint16)
        blended *= alpha
        blended += 255 * (255 - alpha)
        blended += 127
        blended //= 255
        flattened[:, :, index] = blended.astype(numpy.uint8)
    return flattened


def compact_ocr_image(image: RasterImage, *, grayscale: bool = False) -> RasterImage:
    if image.channels == 3 and grayscale:
        if image.width * image.height < 5_000_000:
            return image
        return RasterImage(contiguous_bytes(luma(image.array())), image.width, image.height, 1)
    if image.channels not in {2, 4}:
        return image
    samples = image.array()
    alpha_index = image.channels - 1
    opaque = int(samples[:, :, alpha_index].min()) == 255
    if image.channels == 2:
        if opaque:
            return RasterImage(contiguous_bytes(samples[:, :, 0]), image.width, image.height, 1)
        distance_from_white = numpy.multiply(
            255 - samples[:, :, 0],
            samples[:, :, 1],
            dtype=numpy.uint16,
        )
        distance_from_white += 127
        distance_from_white //= 255
        gray_alpha = 255 - distance_from_white.astype(numpy.uint8)
        return RasterImage(contiguous_bytes(gray_alpha), image.width, image.height, 1)
    colour = samples if opaque else flatten_onto_white(samples)
    return RasterImage(contiguous_bytes(luma(colour)), image.width, image.height, 1)


OCR_IMAGE_TEXT_SAMPLE_PIXELS = 300_000


OCR_IMAGE_TEXT_EDGE_DELTA = 24


OCR_IMAGE_TEXT_MIN_HORIZONTAL_EDGES = 0.015


OCR_IMAGE_TEXT_PHOTO_MAX_WHITE = 0.20


OCR_IMAGE_TEXT_PHOTO_MIN_ENTROPY = 3.0


OCR_IMAGE_TEXT_STRONG_HORIZONTAL_EDGES = 0.09


OCR_IMAGE_TEXT_MIN_HORIZONTAL_EDGE_SHARE = 0.85


class RasterTextSignal(Record):
    __slots__ = ("likely_text", "horizontal_edge_ratio")

    likely_text: bool
    horizontal_edge_ratio: float

    __fields__: ClassVar[tuple[str, ...]] = ("likely_text", "horizontal_edge_ratio")
    __match_args__ = ("likely_text", "horizontal_edge_ratio")

    def __init__(self, likely_text: bool, horizontal_edge_ratio: float) -> None:
        frozen_setattr(self, "likely_text", likely_text)
        frozen_setattr(self, "horizontal_edge_ratio", horizontal_edge_ratio)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.likely_text == other.likely_text
            and self.horizontal_edge_ratio == other.horizontal_edge_ratio
        )

    def __hash__(self) -> int:
        return hash((self.likely_text, self.horizontal_edge_ratio))


def raster_text_signal(image: RasterImage) -> RasterTextSignal:
    pixels = image.array()
    sample_step = max(
        1,
        math.ceil(math.sqrt(image.width * image.height / OCR_IMAGE_TEXT_SAMPLE_PIXELS)),
    )
    sampled = pixels[::sample_step, ::sample_step]
    gray = visible_intensity(sampled)

    gray_16 = gray.astype(numpy.int16)
    horizontal_edges = (
        float(numpy.mean(numpy.abs(numpy.diff(gray_16, axis=1)) >= OCR_IMAGE_TEXT_EDGE_DELTA))
        if gray.shape[1] > 1
        else 0.0
    )
    vertical_edges = (
        float(numpy.mean(numpy.abs(numpy.diff(gray_16, axis=0)) >= OCR_IMAGE_TEXT_EDGE_DELTA))
        if gray.shape[0] > 1
        else 0.0
    )
    white_ratio = float(numpy.mean(gray >= 245))
    histogram = numpy.bincount((gray // 8).reshape(-1), minlength=32).astype(numpy.float64)
    histogram /= max(1.0, float(numpy.sum(histogram)))
    occupied = histogram > 0.0
    entropy = float(-numpy.sum(histogram[occupied] * numpy.log2(histogram[occupied])))

    likely_text = True
    if horizontal_edges < OCR_IMAGE_TEXT_MIN_HORIZONTAL_EDGES:
        likely_text = False
    else:
        horizontal_edge_share = horizontal_edges / max(1e-9, vertical_edges)
        strongly_structured = bool(
            horizontal_edges >= OCR_IMAGE_TEXT_STRONG_HORIZONTAL_EDGES
            and horizontal_edge_share >= OCR_IMAGE_TEXT_MIN_HORIZONTAL_EDGE_SHARE
        )
        if (
            white_ratio < OCR_IMAGE_TEXT_PHOTO_MAX_WHITE
            and entropy >= OCR_IMAGE_TEXT_PHOTO_MIN_ENTROPY
            and not strongly_structured
        ):
            likely_text = False
    return RasterTextSignal(
        likely_text=likely_text,
        horizontal_edge_ratio=horizontal_edges,
    )


def adaptive_ocr_raster(raster: Raster) -> Raster:
    pixels = raster.image.array()
    gray = visible_intensity(pixels).astype(numpy.float32)
    radius = max(8, min(24, min(raster.width, raster.height) // 80))
    integral = numpy.pad(gray, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    y = numpy.arange(raster.height)
    x = numpy.arange(raster.width)
    y0 = numpy.maximum(0, y - radius)
    y1 = numpy.minimum(raster.height, y + radius + 1)
    x0 = numpy.maximum(0, x - radius)
    x1 = numpy.minimum(raster.width, x + radius + 1)
    local_sum = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    local_area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(numpy.float32)
    threshold = local_sum / local_area - 9.0
    binary = numpy.where(gray <= threshold, numpy.uint8(0), numpy.uint8(255))
    return Raster(
        RasterImage(contiguous_bytes(binary), raster.width, raster.height, 1),
        raster.resolution,
    )


def decoded_image_raster(
    image: CapturedDrawing,
    display_area: float,
    *,
    user_unit: float = 1.0,
    max_pixels: int = MAX_OCR_PIXELS,
    upscale: bool = True,
) -> Raster | None:
    source = image.image_source
    shared = decode_image(source) if source is not None else None
    samples: numpy.ndarray[Any, Any] | None
    data: bytes | memoryview | None
    if shared is not None:
        samples = shared.array
        data = None
        decoded_width = shared.width
        decoded_height = shared.height
        decoded_channels = shared.channels
    else:
        raw = image.raw_data
        dictionary = image.dictionary
        if not isinstance(raw, (bytes, bytearray, memoryview)) or not isinstance(dictionary, dict):
            return None
        decoded = decode_pdf_image(raw, dictionary)
        if decoded is None:
            return None
        if isinstance(decoded.data, numpy.ndarray):
            array = numpy.asarray(decoded.data)
            samples = array.reshape(decoded.height, decoded.width, decoded.channels)
            data = None
        elif isinstance(decoded.data, (bytes, memoryview)):
            samples = None
            data = decoded.data
        else:
            samples = None
            data = memoryview(decoded.data).cast("B")
        decoded_width = decoded.width
        decoded_height = decoded.height
        decoded_channels = decoded.channels
    pixels_per_point = (
        math.sqrt(decoded_width * decoded_height / max(1.0, display_area)) / user_unit
    )
    resolution = int(round(max(70.0, min(600.0, 72.0 * pixels_per_point))))
    width = decoded_width
    height = decoded_height
    channels = decoded_channels
    if width * height > max_pixels:
        reduction = (
            min(
                math.sqrt(max_pixels / (width * height)),
                max_pixels / width,
                max_pixels / height,
            )
            * 0.999
        )
        target_width = max(1, int(width * reduction))
        target_height = max(1, int(height * reduction))
        if samples is None:
            assert data is not None
            samples = uint8_image_view(data, (height, width, channels))
        samples = resample_nearest(samples, target_height, target_width)
        data = None
        resolution = max(70, int(round(resolution * target_width / width)))
        width = target_width
        height = target_height
    headroom = math.sqrt(max_pixels / max(1, width * height))
    scale = min(DIRECT_OCR_TARGET_RESOLUTION / max(1, resolution), headroom)
    if upscale and scale > DIRECT_OCR_MIN_UPSCALE:
        if samples is None:
            assert data is not None
            samples = uint8_image_view(data, (height, width, channels))
        whole_factor = round(scale)
        if (
            1 <= whole_factor <= headroom
            and abs(scale - whole_factor) <= DIRECT_OCR_WHOLE_SCALE_TOLERANCE
        ):
            target_width = width * whole_factor
            target_height = height * whole_factor
            samples = resample_nearest(samples, target_height, target_width)
        else:
            target_width = max(1, int(width * scale))
            target_height = max(1, int(height * scale))
            samples = resample_bilinear(samples, target_height, target_width)
        data = None
        resolution = max(70, int(round(resolution * target_width / width)))
        width = target_width
        height = target_height
    if data is None:
        assert samples is not None
        data = contiguous_bytes(samples)
    return Raster(RasterImage(data, width, height, channels), resolution)


class DirectImageOrientation(StrEnum):
    IDENTITY = "identity"
    FLIP_X = "flip-x"
    FLIP_Y = "flip-y"
    FLIP_XY = "flip-xy"
    TRANSPOSE = "transpose"
    TRANSPOSE_FLIP_X = "transpose-flip-x"
    TRANSPOSE_FLIP_Y = "transpose-flip-y"
    TRANSPOSE_FLIP_XY = "transpose-flip-xy"


DIRECT_IMAGE_ORIENTATIONS: dict[DirectImageOrientation, tuple[int, int, int, int]] = {
    DirectImageOrientation.IDENTITY: (0, 1, 2, 3),
    DirectImageOrientation.FLIP_X: (1, 0, 3, 2),
    DirectImageOrientation.FLIP_Y: (2, 3, 0, 1),
    DirectImageOrientation.FLIP_XY: (3, 2, 1, 0),
    DirectImageOrientation.TRANSPOSE: (0, 2, 1, 3),
    DirectImageOrientation.TRANSPOSE_FLIP_X: (2, 0, 3, 1),
    DirectImageOrientation.TRANSPOSE_FLIP_Y: (1, 3, 0, 2),
    DirectImageOrientation.TRANSPOSE_FLIP_XY: (3, 1, 2, 0),
}


def direct_image_orientation(
    image: CapturedDrawing,
    *,
    maximum_axis_deviation: float = 1e-5,
) -> DirectImageOrientation | None:
    items = image.items
    quad = next(
        (
            value
            for kind, value in items
            if kind == "quad" and isinstance(value, (list, tuple)) and len(value) == 4
        ),
        None,
    )
    if quad is None:
        return None
    try:
        points = tuple((float(point[0]), float(point[1])) for point in quad)
    except IndexError, TypeError, ValueError:
        return None
    if not all(math.isfinite(value) for point in points for value in point):
        return None
    xs, ys = zip(*points, strict=True)
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    if x1 <= x0 or y1 <= y0:
        return None
    target_corners = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
    tolerance = max(0.01, max(x1 - x0, y1 - y0) * maximum_axis_deviation)
    target_to_raw = [-1, -1, -1, -1]
    for raw_index, point in enumerate(points):
        target_index = min(
            range(4),
            key=lambda index: (
                abs(point[0] - target_corners[index][0]) + abs(point[1] - target_corners[index][1])
            ),
        )
        target = target_corners[target_index]
        if max(abs(point[0] - target[0]), abs(point[1] - target[1])) > tolerance:
            return None
        if target_to_raw[target_index] != -1:
            return None
        target_to_raw[target_index] = raw_index
    orientation_corners = tuple(target_to_raw)
    return next(
        (
            orientation
            for orientation, corners in DIRECT_IMAGE_ORIENTATIONS.items()
            if corners == orientation_corners
        ),
        None,
    )


def orient_direct_image_raster(
    image: CapturedDrawing,
    raster: Raster,
    *,
    orientation: DirectImageOrientation | None = None,
) -> Raster:
    orientation = orientation or direct_image_orientation(image)
    if orientation is None or orientation is DirectImageOrientation.IDENTITY:
        return raster
    samples = raster.image.array()
    origin, right, below, _ = DIRECT_IMAGE_ORIENTATIONS[orientation]
    oriented = samples.transpose(1, 0, 2) if abs(right - origin) == 2 else samples
    if right < origin:
        oriented = oriented[:, ::-1]
    if below < origin:
        oriented = oriented[::-1]
    height, width, channels = oriented.shape
    return Raster(
        RasterImage(contiguous_bytes(oriented), int(width), int(height), int(channels)),
        raster.resolution,
    )


def fit_raster_scale(
    rendered: Any,
    scale: float,
    max_pixels: int,
    *,
    crop: tuple[float, float, float, float] | None = None,
) -> float:
    if max_pixels < 1:
        raise ValueError("OCR raster pixel budget must be positive")
    width, height = rendered.unrotated_raster_size(scale, crop=crop)
    while width * height > max_pixels:
        scale *= min(max(1, width - 1) / width, max(1, height - 1) / height)
        width, height = rendered.unrotated_raster_size(scale, crop=crop)
    return scale


def rendered_page_raster(
    capture: PageAnalysis,
    requested_scale: float,
    *,
    rendered: Any,
    crop: tuple[float, float, float, float] | None = None,
    max_pixels: int = MAX_OCR_PIXELS,
) -> Raster | None:
    page = capture.page
    if crop is None:
        raster_area = max(1.0, float(page.width) * float(page.height))
    else:
        raster_area = max(1.0, (crop[2] - crop[0]) * (crop[3] - crop[1]))
    user_unit = float(getattr(page, "user_unit", 1.0))
    safe_scale = math.sqrt(max_pixels / raster_area) * 0.999 / user_unit
    scale = min(requested_scale, safe_scale)
    scale = fit_raster_scale(rendered, scale, max_pixels, crop=crop)
    try:
        data = rendered.rasterize(
            background=(255, 255, 255, 255),
            scale=scale,
            max_pixels=max_pixels,
            crop=crop,
        )
    except IndexError:
        return None
    return Raster(
        data,
        max(70, int(round(72.0 * scale))),
    )


def safe_image_crop(capture: PageAnalysis) -> tuple[float, float, float, float] | None:
    evidence = capture.evidence
    if not evidence.image_boxes or not (
        evidence.full_page_image or evidence.image_area_ratio >= 0.65
    ):
        return None
    page_width = capture.width
    page_height = capture.height
    bounds = bbox_union(evidence.image_boxes)
    assert bounds is not None
    x0, y0, x1, y1 = bounds
    crop = (max(0.0, x0), max(0.0, y0), min(page_width, x1), min(page_height, y1))
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        return None
    crop_area = (crop[2] - crop[0]) * (crop[3] - crop[1])
    if crop_area >= max(1.0, page_width * page_height * FULL_PAGE_IMAGE_COVERAGE):
        return None
    return crop
