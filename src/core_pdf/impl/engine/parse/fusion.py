# SPDX-License-Identifier: AGPL-3.0-only
"""Merge observations from native text and OCR into one set."""

from __future__ import annotations

import numpy

from core_pdf.impl.engine.layout.spatial import (
    SpatialIndex,
    bbox_intersection_area,
)
from core_pdf.impl.engine.parse.model import (
    FusionPolicy,
    ObservationBatch,
    PageRoute,
    WorkPlan,
    internal_candidate,
)

# Confidence a recognized pass must reach before it may displace or
# supplement noisy native text during fusion. Deliberately stricter than the
# recognition pass floor in route.py: a low pass floor only admits more
# candidate words, while this gate decides whether recognized text replaces
# text the document actually carries.
FUSION_NOISY_NATIVE_MIN_CONFIDENCE = 90.0

FUSION_GEOMETRY_CHUNK = 256

# Upper bound on elements materialized per vectorized overlap chunk.  The
# chunked path broadcasts FUSION_GEOMETRY_CHUNK candidates against the full
# native set, so only the native box count bounds per-chunk memory.
FUSION_VECTORIZED_ELEMENTS = 1_000_000


def internal_compact_text(text: str) -> str:
    return "".join(character.casefold() for character in text if character.isalnum())


def internal_text_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (internal_compact_text(part) for part in text.casefold().split())
        if len(token) >= 2
    )


def internal_duplicate_of_native_text(
    native_compact: str,
    native_tokens: frozenset[str],
    ocr_text: str,
) -> bool:
    """Detect raster OCR that repeats the page's native text.

    Vector pages can have different coordinate systems after rasterization, so
    geometry alone cannot identify duplicate OCR.  A compact text containment
    check is deliberately limited to reasonably long observations to avoid
    discarding short schematic labels such as ``R1`` or ``+5V``.
    """
    compact = internal_compact_text(ocr_text)
    if len(compact) < 8:
        tokens = internal_text_tokens(ocr_text)
        return bool(tokens) and all(token in native_tokens for token in tokens)
    if compact in native_compact:
        return True
    tokens = internal_text_tokens(ocr_text)
    return bool(tokens) and all(token in native_tokens for token in tokens)


def maximum_candidate_coverage(
    candidate_boxes: numpy.ndarray,
    native_boxes: numpy.ndarray,
) -> numpy.ndarray:
    """Return each candidate's maximum covered-area ratio in bounded chunks."""
    if not len(candidate_boxes) or not len(native_boxes):
        return numpy.zeros(len(candidate_boxes), dtype=numpy.float32)
    if len(native_boxes) * FUSION_GEOMETRY_CHUNK > FUSION_VECTORIZED_ELEMENTS:
        native_index = SpatialIndex.from_boxes(native_boxes)
        output = numpy.zeros(len(candidate_boxes), dtype=numpy.float32)
        for index, box in enumerate(candidate_boxes):
            area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
            maximum = 0.0
            for hit in native_index.intersecting_hits(box):
                maximum = max(maximum, bbox_intersection_area(box, hit.bbox))
            output[index] = maximum / area
        return output
    output = numpy.zeros(len(candidate_boxes), dtype=numpy.float32)
    native_x0 = native_boxes[:, 0][None, :]
    native_y0 = native_boxes[:, 1][None, :]
    native_x1 = native_boxes[:, 2][None, :]
    native_y1 = native_boxes[:, 3][None, :]
    for start in range(0, len(candidate_boxes), FUSION_GEOMETRY_CHUNK):
        stop = min(len(candidate_boxes), start + FUSION_GEOMETRY_CHUNK)
        boxes = candidate_boxes[start:stop]
        widths = numpy.maximum(
            0.0,
            numpy.minimum(boxes[:, None, 2], native_x1)
            - numpy.maximum(boxes[:, None, 0], native_x0),
        )
        heights = numpy.maximum(
            0.0,
            numpy.minimum(boxes[:, None, 3], native_y1)
            - numpy.maximum(boxes[:, None, 1], native_y0),
        )
        areas = numpy.maximum(
            1.0,
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
        )
        output[start:stop] = numpy.max(widths * heights, axis=1) / areas
    return output


def fuse_observations(
    native: ObservationBatch,
    ocr: ObservationBatch,
    plan: WorkPlan,
) -> ObservationBatch:
    route = plan.route
    if route is PageRoute.NATIVE or not len(ocr):
        return native
    if route is PageRoute.OCR or not len(native):
        return ocr
    if (
        plan.fusion_policy is FusionPolicy.SPARSE_NATIVE
        and len(native) < 16
        and len(ocr) >= len(native) * 4
        and sum(character.isalnum() for text in native.text for character in text) <= 4
    ):
        return ocr
    if plan.fusion_policy is FusionPolicy.NOISY_NATIVE:
        native_candidate = internal_candidate(-1, native)
        ocr_candidate = internal_candidate(-1, ocr)
        if (
            len(ocr) >= 4
            and ocr_candidate.metrics.mean_confidence >= FUSION_NOISY_NATIVE_MIN_CONFIDENCE
            and ocr_candidate.metrics.utility >= native_candidate.metrics.utility * 1.05
            and ocr_candidate.metrics.alphanumeric_characters
            >= native_candidate.metrics.alphanumeric_characters * 0.80
        ):
            return ocr

    minimum_confidence = (
        75.0
        if plan.image_regions_only
        else FUSION_NOISY_NATIVE_MIN_CONFIDENCE
        if plan.fusion_policy is FusionPolicy.NOISY_NATIVE
        else 30.0
        if plan.fusion_policy is FusionPolicy.UNCOVERED_VECTOR
        else 45.0
    )
    confidence_mask = ocr.confidence >= minimum_confidence
    if plan.image_regions_only or plan.fusion_policy is FusionPolicy.UNCOVERED_VECTOR:
        alphanumeric_mask = numpy.fromiter(
            (sum(character.isalnum() for character in text) >= 1 for text in ocr.text),
            dtype=numpy.bool_,
            count=len(ocr),
        )
    else:
        alphanumeric_mask = numpy.ones(len(ocr), dtype=numpy.bool_)
    if len(native):
        native_compact = "".join(internal_compact_text(text) for text in native.text)
        native_tokens = frozenset(
            token for text in native.text for token in internal_text_tokens(text)
        )
        duplicate_mask = numpy.fromiter(
            (
                internal_duplicate_of_native_text(native_compact, native_tokens, text)
                for text in ocr.text
            ),
            dtype=numpy.bool_,
            count=len(ocr),
        )
    else:
        duplicate_mask = numpy.zeros(len(ocr), dtype=numpy.bool_)
    overlap = maximum_candidate_coverage(ocr.bbox, native.bbox)
    additions = confidence_mask & alphanumeric_mask & ~duplicate_mask & (overlap < 0.30)
    return ObservationBatch.concatenate_selected(native, ocr, additions)
