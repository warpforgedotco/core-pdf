# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable

from core_pdf.impl.engine.extraction.ocr.types import (
    OcrAffineTransform,
    OcrComponentBox,
    OcrDeskewInfo,
    OcrImage,
    OcrObservation,
    OcrRow,
    OcrTextResult,
    ocr_float_value,
    ocr_int_value,
)

OCR_DESKEW_BINARY_THRESHOLD = 130
OCR_DESKEW_CONFIDENCE_THRESHOLD = 4.0
OCR_DESKEW_MIN_ANGLE_DEGREES = 0.1
OCR_DESKEW_MAX_ANGLE_DEGREES = 5.0

IDENTITY_AFFINE_TRANSFORM: OcrAffineTransform = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def full_page_ocr_image_should_be_deskewed(image: OcrImage) -> bool:
    source = str(image.source)
    return (
        source.startswith(("full_page_", "rendered_page_"))
        and "_tile_" not in source
        and image.width > 0
        and image.height > 0
    )


def rotation_about_image_center(
    angle_degrees: float,
    width: int,
    height: int,
) -> OcrAffineTransform:
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    center_x = width / 2.0
    center_y = height / 2.0
    # Pixel coordinates have a downward-positive y axis, so this is a
    # clockwise-positive rotation, matching Leptonica's deskew angle.
    offset_x = center_x - cosine * center_x + sine * center_y
    offset_y = center_y - sine * center_x - cosine * center_y
    return (cosine, sine, -sine, cosine, offset_x, offset_y)


def deskew_info(
    *,
    source: str,
    angle_degrees: float | None,
    confidence: float | None,
    applied: bool,
    reason: str,
    image_width: int,
    image_height: int,
) -> OcrDeskewInfo:
    if applied and angle_degrees is not None:
        source_to_deskew = rotation_about_image_center(
            angle_degrees,
            image_width,
            image_height,
        )
        deskew_to_source = rotation_about_image_center(
            -angle_degrees,
            image_width,
            image_height,
        )
    else:
        source_to_deskew = IDENTITY_AFFINE_TRANSFORM
        deskew_to_source = IDENTITY_AFFINE_TRANSFORM
    return OcrDeskewInfo(
        source=source,
        angle_degrees=angle_degrees,
        confidence=confidence,
        applied=applied,
        reason=reason,
        image_width=image_width,
        image_height=image_height,
        source_to_deskew=source_to_deskew,
        deskew_to_source=deskew_to_source,
    )


def transform_point(
    point: tuple[float, float],
    transform: OcrAffineTransform,
) -> tuple[float, float]:
    x, y = point
    a, b, c, d, offset_x, offset_y = transform
    return (
        a * x + c * y + offset_x,
        b * x + d * y + offset_y,
    )


def transform_pixel_bbox(
    bbox: tuple[int, int, int, int],
    transform: OcrAffineTransform,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = bbox
    points = (
        transform_point((x0, y0), transform),
        transform_point((x1, y0), transform),
        transform_point((x0, y1), transform),
        transform_point((x1, y1), transform),
    )
    left = max(0, min(width, math.floor(min(point[0] for point in points))))
    top = max(0, min(height, math.floor(min(point[1] for point in points))))
    right = max(0, min(width, math.ceil(max(point[0] for point in points))))
    bottom = max(0, min(height, math.ceil(max(point[1] for point in points))))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def source_rectangle_to_deskew_rectangle(
    rectangle: tuple[int, int, int, int],
    image: OcrImage,
    info: OcrDeskewInfo,
) -> tuple[int, int, int, int] | None:
    source_width = max(1, image.width)
    source_height = max(1, image.height)
    scaled = (
        round(rectangle[0] * info.image_width / source_width),
        round(rectangle[1] * info.image_height / source_height),
        round(rectangle[2] * info.image_width / source_width),
        round(rectangle[3] * info.image_height / source_height),
    )
    return transform_pixel_bbox(
        scaled,
        info.source_to_deskew,
        width=info.image_width,
        height=info.image_height,
    )


def restore_ocr_result_geometry(
    result: OcrTextResult,
    info: OcrDeskewInfo,
) -> OcrTextResult:
    if not info.applied:
        return replace(result, deskew_info=info)

    restore_transform = info.deskew_to_source
    output_width = info.image_width
    output_height = info.image_height

    line_rows = _transform_rows(result.line_rows, restore_transform, output_width, output_height)
    word_rows = _transform_rows(result.word_rows, restore_transform, output_width, output_height)
    symbol_rows = _transform_rows(
        result.symbol_rows,
        restore_transform,
        output_width,
        output_height,
    )
    component_boxes = _transform_component_boxes(
        result.component_boxes,
        restore_transform,
        output_width,
        output_height,
    )
    observations = _transform_observations(
        result.observations,
        restore_transform,
        output_width,
        output_height,
        info,
    )
    return replace(
        result,
        line_rows=line_rows,
        word_rows=word_rows,
        symbol_rows=symbol_rows,
        component_boxes=component_boxes,
        observations=observations,
        deskew_info=info,
    )


def deskew_diagnostic(info: OcrDeskewInfo) -> dict[str, object]:
    return {
        "source": info.source,
        "angle_degrees": info.angle_degrees,
        "confidence": info.confidence,
        "applied": info.applied,
        "reason": info.reason,
        "image_width": info.image_width,
        "image_height": info.image_height,
        "source_to_deskew": info.source_to_deskew,
        "deskew_to_source": info.deskew_to_source,
    }


def _transform_rows(
    rows: Iterable[OcrRow],
    transform: OcrAffineTransform,
    width: int,
    height: int,
) -> tuple[OcrRow, ...]:
    transformed: list[OcrRow] = []
    for row in rows:
        try:
            left = ocr_int_value(row["left"])
            top = ocr_int_value(row["top"])
            row_width = ocr_int_value(row["width"])
            row_height = ocr_int_value(row["height"])
        except (KeyError, TypeError, ValueError):
            transformed.append(dict(row))
            continue
        bbox = transform_pixel_bbox(
            (left, top, left + row_width, top + row_height),
            transform,
            width=width,
            height=height,
        )
        if bbox is None:
            continue
        mapped = dict(row)
        mapped["left"] = bbox[0]
        mapped["top"] = bbox[1]
        mapped["width"] = bbox[2] - bbox[0]
        mapped["height"] = bbox[3] - bbox[1]
        baseline = _transform_baseline(row.get("baseline"), transform, width, height)
        if baseline is not None:
            mapped["baseline"] = baseline
        transformed.append(mapped)
    return tuple(transformed)


def _transform_component_boxes(
    boxes: Iterable[OcrComponentBox],
    transform: OcrAffineTransform,
    width: int,
    height: int,
) -> tuple[OcrComponentBox, ...]:
    transformed: list[OcrComponentBox] = []
    for box in boxes:
        bbox = transform_pixel_bbox(
            (box.left, box.top, box.left + box.width, box.top + box.height),
            transform,
            width=width,
            height=height,
        )
        if bbox is None:
            continue
        transformed.append(
            replace(
                box,
                left=bbox[0],
                top=bbox[1],
                width=bbox[2] - bbox[0],
                height=bbox[3] - bbox[1],
            )
        )
    return tuple(transformed)


def _transform_observations(
    observations: Iterable[OcrObservation],
    transform: OcrAffineTransform,
    width: int,
    height: int,
    info: OcrDeskewInfo,
) -> tuple[OcrObservation, ...]:
    transformed: list[OcrObservation] = []
    provenance = (
        ("deskew_angle_degrees", info.angle_degrees),
        ("deskew_confidence", info.confidence),
        ("deskew_applied", info.applied),
    )
    for observation in observations:
        bbox = transform_pixel_bbox(
            observation.bbox,
            transform,
            width=width,
            height=height,
        )
        if bbox is None:
            continue
        transformed.append(
            replace(
                observation,
                bbox=bbox,
                baseline=_transform_baseline(
                    observation.baseline,
                    transform,
                    width,
                    height,
                ),
                image_width=width,
                image_height=height,
                provenance=(*observation.provenance, *provenance),
            )
        )
    return tuple(transformed)


def _transform_baseline(
    baseline: object,
    transform: OcrAffineTransform,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if not isinstance(baseline, (list, tuple)) or len(baseline) != 4:
        return None
    try:
        start = transform_point(
            (ocr_float_value(baseline[0]), ocr_float_value(baseline[1])),
            transform,
        )
        end = transform_point(
            (ocr_float_value(baseline[2]), ocr_float_value(baseline[3])),
            transform,
        )
    except (TypeError, ValueError):
        return None
    return (
        max(0, min(width, round(start[0]))),
        max(0, min(height, round(start[1]))),
        max(0, min(width, round(end[0]))),
        max(0, min(height, round(end[1]))),
    )
