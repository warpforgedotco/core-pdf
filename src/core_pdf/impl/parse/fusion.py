# SPDX-License-Identifier: AGPL-3.0-only
"""Merge observations from native text and OCR into one set."""

from __future__ import annotations

import numpy

from core_pdf.impl.layout.spatial import (
    maximum_candidate_coverage,
)
from core_pdf.impl.parse.model import (
    FusionPolicy,
    ObservationBatch,
    PageRoute,
    WorkPlan,
    internal_candidate,
)
from core_pdf.impl.text import compact_text, text_tokens

# Confidence a recognized pass must reach before it may displace or
# supplement noisy native text during fusion. Deliberately stricter than the
# recognition pass floor in route.py: a low pass floor only admits more
# candidate words, while this gate decides whether recognized text replaces
# text the document actually carries.
FUSION_NOISY_NATIVE_MIN_CONFIDENCE = 90.0


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
    compact = compact_text(ocr_text)
    if len(compact) < 8:
        tokens = text_tokens(ocr_text)
        return bool(tokens) and all(token in native_tokens for token in tokens)
    if compact in native_compact:
        return True
    tokens = text_tokens(ocr_text)
    return bool(tokens) and all(token in native_tokens for token in tokens)


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
        native_compact = "".join(compact_text(text) for text in native.text)
        native_tokens = frozenset(token for text in native.text for token in text_tokens(text))
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
