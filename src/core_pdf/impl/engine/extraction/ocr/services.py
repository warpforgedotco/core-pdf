# SPDX-License-Identifier: AGPL-3.0-only
"""PDF adapters for the reusable OCR candidate-generation services."""

from __future__ import annotations

from typing import Any, Mapping

from core_ocr.impl import execution as ocr_execution
from core_ocr.impl import iterator_layout as ocr_iterator_layout
from core_ocr.impl import layout as ocr_layout
from core_ocr.impl import selection as ocr_selection
from core_ocr.impl.services import configure_candidate_services
from core_ocr.impl.types import OcrImage, OcrTextResult

from core_pdf.impl.engine.extraction.common import page_geometry
from core_pdf.impl.engine.extraction.ocr import rendering as ocr_rendering
from core_pdf.impl.engine.extraction.ocr import tiling as ocr_tiling


class PdfOcrCandidateServices:
    page_geometry = page_geometry
    iterator_layout = ocr_iterator_layout
    layout = ocr_layout
    selection = ocr_selection
    rectangle_request_type = ocr_execution.RectangleOcrRequest
    dense_vector_render_tile_min_tokens = ocr_rendering.OCR_DENSE_VECTOR_RENDER_TILE_MIN_TOKENS
    rendering = ocr_rendering
    clamp_ocr_bbox = staticmethod(ocr_tiling.clamp_ocr_bbox)

    @staticmethod
    def crop_ocr_image_region(
        image: OcrImage,
        rectangle: tuple[int, int, int, int],
    ) -> OcrImage | None:
        return ocr_execution.crop_ocr_image_region(image, rectangle)

    @staticmethod
    def ocr_image_regions_to_text_results(
        image: OcrImage,
        requests: list[Any],
        timeout: float | None,
    ) -> list[OcrTextResult]:
        return ocr_execution.ocr_image_regions_to_text_results_with_timeout(
            image, requests, timeout
        )

    @staticmethod
    def ocr_image_to_text_result_with_psm(
        image: OcrImage,
        *,
        psm: int,
        timeout: float | None,
        variables: Mapping[str, str | int | float | bool] | None = None,
    ) -> OcrTextResult:
        return ocr_execution.ocr_image_to_text_result_with_psm_timeout(
            image,
            psm=psm,
            timeout=timeout,
            variables=variables,
        )


configure_candidate_services(PdfOcrCandidateServices())
