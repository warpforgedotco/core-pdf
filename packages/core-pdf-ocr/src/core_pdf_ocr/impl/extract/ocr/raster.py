# SPDX-License-Identifier: AGPL-3.0-only
"""Producing the rasters recognition runs against.

Three sources feed OCR: an embedded image decoded directly, the page rendered
through the rasterizer, and crops taken from either. This module owns turning any
of them into pixels at a resolution Tesseract can read, deciding a direct image is
upright, and judging from the pixels alone whether a raster carries text at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy

from core_pdf.impl._impl.model.geometry import bbox_union, points_bbox
from core_pdf.impl._impl.render.model import RasterImage
from core_pdf.impl._impl.runtime.array_views import (
    contiguous_bytes,
    uint8_image_view,
)
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing
from core_pdf.impl.spec.s_08_graphics.image_decode import decode_pdf_image
from core_pdf_ocr.impl.extract.contracts import (
    FULL_PAGE_IMAGE_COVERAGE,
    MAX_OCR_PIXELS,
    PageAnalysis,
)
from core_pdf_ocr.impl.extract.ocr.resampling import resample_bilinear, resample_nearest
from core_pdf_ocr.impl.extract.ocr.types import internal_Raster

# Tesseract's LSTM was trained near 300-400 DPI. Scans below that are enlarged to
# reach it; the gain comes from the resampling being smooth, not from pixel count,
# so enlarging past this target only costs recognition time.
DIRECT_OCR_TARGET_RESOLUTION = 400


DIRECT_OCR_MIN_UPSCALE = 1.05


DIRECT_OCR_WHOLE_SCALE_TOLERANCE = 0.06


def internal_raster_ink_grid(
    raster: internal_Raster, rows: int, columns: int
) -> numpy.ndarray[Any, Any]:
    """Measure visual ink per coarse region from a bounded zero-copy raster sample."""
    if rows <= 0 or columns <= 0:
        return numpy.zeros(max(0, rows * columns), dtype=numpy.float32)
    pixels = raster.image.array()
    y_step = max(1, raster.height // 512)
    x_step = max(1, raster.width // 512)
    sampled = pixels[::y_step, ::x_step]
    if raster.image.channels == 1:
        intensity = sampled[:, :, 0]
    else:
        intensity = numpy.min(sampled[:, :, :3], axis=2)
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


# ITU-R BT.601 luma, in the fixed-point form Tesseract's own conversion uses. The
# weights sum to 256, so an achromatic pixel round-trips to its exact input value.
LUMA_RED = 77
LUMA_GREEN = 150
LUMA_BLUE = 29


def internal_luma(samples: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
    """Reduce interleaved RGB(A) samples to one grayscale plane.

    Accumulates in place: the naive expression allocates a uint16 temporary per
    channel, which costs more than the rasterization it follows on a megapixel page.
    """
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


def internal_flatten_onto_white(samples: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
    """Composite RGBA over white, matching what Tesseract's pixRemoveAlpha would do."""
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


def internal_compact_ocr_image(image: RasterImage, *, grayscale: bool = False) -> RasterImage:
    """Reduce a raster to the narrowest layout Tesseract can read.

    Tesseract discards colour before recognizing: it strips alpha with
    ``pixRemoveAlpha`` (blending onto white) and then runs ``pixConvertTo8``, so RGB
    and RGBA input buy no accuracy and cost two conversion passes plus four times the
    bytes across the API boundary. Doing the reduction here is measurably cheaper --
    19% off the combined SetImageBytes-and-Recognize time on a rendered page -- and
    leaves recognition accuracy unchanged.

    Note the reduction must happen for *large* rasters above all. An earlier version
    returned any four-channel image of a megapixel or more untouched, to skip an alpha
    scan, which meant every rendered page -- the only rasters big enough for the cost
    to matter -- was handed over as RGBA while small crops were compacted.
    """
    if image.channels == 3 and grayscale:
        if image.width * image.height < 5_000_000:
            return image
        return RasterImage(
            contiguous_bytes(internal_luma(image.array())), image.width, image.height, 1
        )
    if image.channels not in {2, 4}:
        return image
    samples = image.array()
    alpha_index = image.channels - 1
    opaque = int(samples[:, :, alpha_index].min()) == 255
    if image.channels == 2:
        if opaque:
            return RasterImage(contiguous_bytes(samples[:, :, 0]), image.width, image.height, 1)
        # Tesseract accepts gray, RGB, and RGBA byte layouts, but not the
        # gray-plus-alpha layout produced by PDF soft masks. Composite it
        # onto the same white background used by page rendering.
        distance_from_white = numpy.multiply(
            255 - samples[:, :, 0],
            samples[:, :, 1],
            dtype=numpy.uint16,
        )
        distance_from_white += 127
        distance_from_white //= 255
        gray_alpha = 255 - distance_from_white.astype(numpy.uint8)
        return RasterImage(contiguous_bytes(gray_alpha), image.width, image.height, 1)
    colour = samples if opaque else internal_flatten_onto_white(samples)
    return RasterImage(contiguous_bytes(internal_luma(colour)), image.width, image.height, 1)


OCR_IMAGE_TEXT_SAMPLE_PIXELS = 300_000


OCR_IMAGE_TEXT_EDGE_DELTA = 24


OCR_IMAGE_TEXT_MIN_HORIZONTAL_EDGES = 0.015


OCR_IMAGE_TEXT_PHOTO_MAX_WHITE = 0.20


OCR_IMAGE_TEXT_PHOTO_MIN_ENTROPY = 3.0


OCR_IMAGE_TEXT_STRONG_HORIZONTAL_EDGES = 0.09


OCR_IMAGE_TEXT_MIN_HORIZONTAL_EDGE_SHARE = 0.85


@dataclass(frozen=True, slots=True)
class internal_RasterTextSignal:
    likely_text: bool
    horizontal_edge_ratio: float


def internal_raster_text_signal(image: RasterImage) -> internal_RasterTextSignal:
    """Reject obvious non-text image supplements using a bounded pixel sample.

    This gate is intentionally limited to image supplements on pages that already
    have native text.  Full-page scan OCR and compositor fallbacks never use it.
    Text and line art have frequent horizontal intensity transitions; continuous-
    tone photographs may also have many edges, but those edges are less strongly
    horizontal and occur without a light document background.
    """
    pixels = image.array()
    sample_step = max(
        1,
        math.ceil(math.sqrt(image.width * image.height / OCR_IMAGE_TEXT_SAMPLE_PIXELS)),
    )
    sampled = pixels[::sample_step, ::sample_step]
    if image.channels == 1:
        gray = sampled[:, :, 0]
    elif image.channels == 2:
        source = sampled[:, :, 0].astype(numpy.uint16)
        alpha = sampled[:, :, 1].astype(numpy.uint16)
        gray = (255 - ((255 - source) * alpha + 127) // 255).astype(numpy.uint8)
    else:
        colour = sampled[:, :, :3]
        if image.channels == 4:
            alpha = sampled[:, :, 3:].astype(numpy.uint16)
            colour = (255 - ((255 - colour.astype(numpy.uint16)) * alpha + 127) // 255).astype(
                numpy.uint8
            )
        gray = numpy.min(colour, axis=2)

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
    return internal_RasterTextSignal(
        likely_text=likely_text,
        horizontal_edge_ratio=horizontal_edges,
    )


def internal_adaptive_ocr_raster(raster: internal_Raster) -> internal_Raster:
    """Binarize faded scans against their local background for a fallback pass."""
    pixels = raster.image.array()
    gray = (
        pixels[:, :, 0] if raster.image.channels == 1 else numpy.min(pixels[:, :, :3], axis=2)
    ).astype(numpy.float32)
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
    return internal_Raster(
        RasterImage(contiguous_bytes(binary), raster.width, raster.height, 1),
        raster.resolution,
    )


def internal_decoded_image_raster(
    image: CapturedDrawing,
    display_area: float,
    *,
    max_pixels: int = MAX_OCR_PIXELS,
    upscale: bool = True,
) -> internal_Raster | None:
    source = image.image_source
    shared = source.decode() if source is not None else None
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
    pixels_per_point = math.sqrt(decoded_width * decoded_height / max(1.0, display_area))
    resolution = max(70, min(600, int(round(72.0 * pixels_per_point))))
    width = decoded_width
    height = decoded_height
    channels = decoded_channels
    if width * height > max_pixels:
        reduction = math.sqrt(max_pixels / (width * height)) * 0.999
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
        # Tesseract's line classifier wants roughly 300-400 DPI. How to get there
        # depends on the factor: a whole-number enlargement is exact pixel
        # replication and keeps stems crisp, while interpolating one only blurs
        # them. Fractional factors have no such option, and there replication
        # staircases the strokes badly enough to change which glyph is read.
        whole_factor = round(scale)
        if whole_factor >= 1 and abs(scale - whole_factor) <= DIRECT_OCR_WHOLE_SCALE_TOLERANCE:
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
    return internal_Raster(RasterImage(data, width, height, channels), resolution)


class DirectImageOrientation(StrEnum):
    IDENTITY = "identity"
    FLIP_X = "flip-x"
    FLIP_Y = "flip-y"
    FLIP_XY = "flip-xy"
    TRANSPOSE = "transpose"
    TRANSPOSE_FLIP_X = "transpose-flip-x"
    TRANSPOSE_FLIP_Y = "transpose-flip-y"
    TRANSPOSE_FLIP_XY = "transpose-flip-xy"


internal_DIRECT_IMAGE_ORIENTATIONS: dict[DirectImageOrientation, tuple[int, int, int, int]] = {
    DirectImageOrientation.IDENTITY: (0, 1, 2, 3),
    DirectImageOrientation.FLIP_X: (1, 0, 3, 2),
    DirectImageOrientation.FLIP_Y: (2, 3, 0, 1),
    DirectImageOrientation.FLIP_XY: (3, 2, 1, 0),
    DirectImageOrientation.TRANSPOSE: (0, 2, 1, 3),
    DirectImageOrientation.TRANSPOSE_FLIP_X: (2, 0, 3, 1),
    DirectImageOrientation.TRANSPOSE_FLIP_Y: (1, 3, 0, 2),
    DirectImageOrientation.TRANSPOSE_FLIP_XY: (3, 1, 2, 0),
}


def internal_direct_image_orientation(
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
    except (IndexError, TypeError, ValueError):
        return None
    bounds = points_bbox(points)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
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
            for orientation, corners in internal_DIRECT_IMAGE_ORIENTATIONS.items()
            if corners == orientation_corners
        ),
        None,
    )


def internal_orient_direct_image_raster(
    image: CapturedDrawing,
    raster: internal_Raster,
    *,
    orientation: DirectImageOrientation | None = None,
) -> internal_Raster:
    orientation = orientation or internal_direct_image_orientation(image)
    if orientation in {None, DirectImageOrientation.IDENTITY}:
        return raster
    samples = raster.image.array()
    match orientation:
        case DirectImageOrientation.FLIP_X:
            oriented = samples[:, ::-1]
        case DirectImageOrientation.FLIP_Y:
            oriented = samples[::-1]
        case DirectImageOrientation.FLIP_XY:
            oriented = samples[::-1, ::-1]
        case DirectImageOrientation.TRANSPOSE:
            oriented = samples.transpose(1, 0, 2)
        case DirectImageOrientation.TRANSPOSE_FLIP_X:
            oriented = samples.transpose(1, 0, 2)[::-1]
        case DirectImageOrientation.TRANSPOSE_FLIP_Y:
            oriented = samples.transpose(1, 0, 2)[:, ::-1]
        case DirectImageOrientation.TRANSPOSE_FLIP_XY:
            oriented = samples.transpose(1, 0, 2)[::-1, ::-1]
        case _:
            return raster
    height, width, channels = oriented.shape
    return internal_Raster(
        RasterImage(contiguous_bytes(oriented), int(width), int(height), int(channels)),
        raster.resolution,
    )


def internal_rendered_page_raster(
    capture: PageAnalysis,
    requested_scale: float,
    *,
    rendered: Any,
    crop: tuple[float, float, float, float] | None = None,
    max_pixels: int = MAX_OCR_PIXELS,
) -> internal_Raster | None:
    page = capture.page
    if crop is None:
        raster_area = max(1.0, float(page.width) * float(page.height))
    else:
        raster_area = max(1.0, (crop[2] - crop[0]) * (crop[3] - crop[1]))
    safe_scale = math.sqrt(max_pixels / raster_area) * 0.999
    scale = min(requested_scale, safe_scale)
    try:
        data = rendered.rasterize(
            background=(255, 255, 255, 255),
            scale=scale,
            max_pixels=max_pixels,
            crop=crop,
        )
    except IndexError:
        # A malformed embedded image can produce a source sample outside its
        # decoded raster during compositing.  Keep native extraction usable and
        # let OCR continue without the rendered-page fallback.
        return None
    return internal_Raster(
        data,
        max(70, int(round(72.0 * scale))),
    )


def internal_safe_image_crop(capture: PageAnalysis) -> tuple[float, float, float, float] | None:
    """Return a useful crop when OCR is known to be image-dominated.

    A crop is only safe when the image coverage is substantial.  Sparse images
    must not hide page text outside the image bounds from the page OCR path.
    """
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
