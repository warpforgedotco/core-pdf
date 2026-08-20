# SPDX-License-Identifier: AGPL-3.0-only
"""Typed diagnostic contracts shared by extraction stages and benchmark tooling."""

from __future__ import annotations

from typing import TypedDict

PARSE_PLAN_KEY = "parse_plan"
PARSE_METRICS_KEY = "parse_metrics"
OCR_PASS_DIAGNOSTICS_KEY = "ocr_pass_diagnostics"
CAPTURE_DIAGNOSTICS_KEY = "capture_diagnostics"
HIDDEN_TEXT_VERIFICATION_KEY = "hidden_text_verification"
STROKED_VECTOR_DECODE_KEY = "stroked_vector_decode"
STROKED_VECTOR_PACKED_KEY = "stroked_vector_packed"
DOCUMENT_STROKED_GLYPHS_KEY = "document_stroked_glyphs"


class OcrPassDiagnostic(TypedDict, total=False):
    name: str
    scope: str
    scale: float
    modes: tuple[int, ...]
    task_count: int
    raster_pixels: int
    skipped_raster_pixels: int
    full_page_fallback: bool
    elapsed_seconds: float
    accepted_additions: int
    selected: bool


class ParseMetrics(TypedDict, total=False):
    route: str
    preflight_class: str
    content_stream_passes: int
    capture_product_count: int
    native_observations: int
    ocr_observations: int
    fused_observations: int
    ocr_raster_pixels: int
    layout_strategy: str


__all__ = (
    "CAPTURE_DIAGNOSTICS_KEY",
    "DOCUMENT_STROKED_GLYPHS_KEY",
    "HIDDEN_TEXT_VERIFICATION_KEY",
    "OCR_PASS_DIAGNOSTICS_KEY",
    "PARSE_METRICS_KEY",
    "PARSE_PLAN_KEY",
    "STROKED_VECTOR_DECODE_KEY",
    "STROKED_VECTOR_PACKED_KEY",
    "OcrPassDiagnostic",
    "ParseMetrics",
)
