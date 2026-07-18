# SPDX-License-Identifier: AGPL-3.0-only
"""PDF adapters for the reusable OCR candidate-generation services."""

from __future__ import annotations

from typing import Any, Mapping

from core_filters.impl.flate import apply_flate
from core_filters.impl.pipeline import decode_stream_data
from core_layout.impl.layout.geometry_quality import (
    layout_geometry_summary_record,
    page_layout_geometry_summary,
)
from core_ocr.impl import execution as ocr_execution
from core_ocr.impl import iterator_layout as ocr_iterator_layout
from core_ocr.impl import layout as ocr_layout
from core_ocr.impl import rendering as ocr_rendering
from core_ocr.impl import selection as ocr_selection
from core_ocr.impl import tiling as ocr_tiling
from core_ocr.impl.rendering import OcrRenderTile
from core_ocr.impl.services import configure_candidate_services
from core_ocr.impl.types import OcrImage, OcrTextResult

from core_pdf.impl.engine.extraction.common import observation_resolver, page_geometry, page_profile
from core_pdf.impl.engine.extraction.common.ordering import LayoutAnalyzer
from core_pdf.impl.engine.extraction.common.render import (
    MarkdownRenderer,
    render_page_observation_lines,
    render_resolved_text_lines,
)
from core_pdf.impl.engine.extraction.page_text import decisions as page_text_decisions
from core_pdf.impl.engine.extraction.page_text import native as native_text
from core_pdf.impl.engine.extraction.tables.grid import detect_grid, merge_grids
from core_pdf.impl.engine.rendering import RenderOptions, compose_page
from core_pdf.impl.engine.rendering.models import RenderedPage, image_filter_names, pdf_int
from core_pdf.impl.engine.spec.s_07_document.page_boxes import rotate_page_lines
from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_08_graphics.color import ImageColorManager
from core_pdf.impl.engine.spec.s_09_fonts.glyphs import glyph_name_to_unicode
from core_pdf.impl.engine.spec.s_09_fonts.helpers import parse_differences
from core_pdf.impl.objects import PdfStream


class PdfAnalysisServices:
    stream_type = PdfStream
    lookup_dict_key = staticmethod(lookup_dict_key)
    normalize_pdf_name = staticmethod(normalize_pdf_name)
    glyph_name_to_unicode = staticmethod(glyph_name_to_unicode)
    parse_differences = staticmethod(parse_differences)


class PdfHostServices:
    page_geometry = page_geometry
    iterator_layout = ocr_iterator_layout
    layout = ocr_layout
    selection = ocr_selection
    rectangle_request_type = ocr_execution.RectangleOcrRequest
    dense_vector_render_tile_min_tokens = ocr_rendering.OCR_DENSE_VECTOR_RENDER_TILE_MIN_TOKENS
    rendering = ocr_rendering
    layout_analyzer = LayoutAnalyzer
    pdf_analysis = PdfAnalysisServices
    observation_resolver = observation_resolver
    page_profile = page_profile
    render_options = RenderOptions
    image_color_manager = ImageColorManager
    markdown_renderer = MarkdownRenderer
    render_resolved_text_lines = staticmethod(render_resolved_text_lines)
    render_page_observation_lines = staticmethod(render_page_observation_lines)
    compose_page = staticmethod(compose_page)
    apply_flate = staticmethod(apply_flate)
    decode_stream_data = staticmethod(decode_stream_data)
    lookup_dict_key = staticmethod(lookup_dict_key)
    image_filter_names = staticmethod(image_filter_names)
    pdf_int = staticmethod(pdf_int)
    page_extraction_decision = staticmethod(page_text_decisions.page_extraction_decision)
    layout_geometry_summary_record = staticmethod(layout_geometry_summary_record)
    page_layout_geometry_summary = staticmethod(page_layout_geometry_summary)
    native_text = native_text
    try_extract_native_text_fast = staticmethod(native_text.try_extract_native_text_fast)
    native_text_runs_for_extraction = staticmethod(native_text.native_text_runs_for_extraction)
    native_text_runs_inside_page_bounds = staticmethod(
        native_text.native_text_runs_inside_page_bounds
    )
    native_text_runs_inside_visible_row_bands = staticmethod(
        native_text.native_text_runs_inside_visible_row_bands
    )
    select_native_text_layout = staticmethod(native_text.select_native_text_layout)
    should_try_rendered_glyph_repair = staticmethod(native_text.should_try_rendered_glyph_repair)
    apply_rendered_glyph_repair_to_native_text = staticmethod(
        native_text.apply_rendered_glyph_repair_to_native_text
    )
    native_layout_geometry_summary_for_runs = staticmethod(
        native_text.native_layout_geometry_summary_for_runs
    )
    native_invisible_text_layer_has_fragmented_geometry = staticmethod(
        native_text.native_invisible_text_layer_has_fragmented_geometry
    )
    native_invisible_text_layer_is_trustworthy = staticmethod(
        native_text.native_invisible_text_layer_is_trustworthy
    )
    detect_grid = staticmethod(detect_grid)
    merge_grids = staticmethod(merge_grids)
    rotate_page_lines = staticmethod(rotate_page_lines)
    clamp_ocr_bbox = staticmethod(ocr_tiling.clamp_ocr_bbox)

    @staticmethod
    def render_page_for_ocr(page: Any, *, include_text: bool) -> Any:
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
        rendered = page.render(
            RenderOptions(
                include_text=include_text,
                include_annotations=True,
                include_layers=True,
            )
        )
        if cache is not None:
            cache[cache_key] = rendered
        return rendered

    @staticmethod
    def render_page_for_ocr_analysis(page: Any) -> Any:
        cache = page.extraction_cache
        cache_key = "ocr_analysis_rendered_page"
        if cache is not None:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        rendered = page.render(RenderOptions(include_annotations=False, include_layers=False))
        if cache is not None:
            cache[cache_key] = rendered
        return rendered

    @staticmethod
    def rasterize_ocr_page(rendered: RenderedPage, dpi: int) -> OcrImage | None:
        width = max(1, int(round(float(rendered.width) * dpi / 72.0)))
        height = max(1, int(round(float(rendered.height) * dpi / 72.0)))
        if rendered.rotate % 180:
            width, height = height, width
        data = rendered.rasterize(background=(255, 255, 255, 255), scale=dpi / 72.0)
        if len(data) != width * height * 4:
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

    @staticmethod
    def rasterize_ocr_tile(
        rendered: RenderedPage,
        tile: OcrRenderTile,
        dpi: int,
        *,
        source: str,
    ) -> OcrImage | None:
        width_points = max(0.0, tile.crop[2] - tile.crop[0])
        height_points = max(0.0, tile.crop[3] - tile.crop[1])
        width, height = ocr_rendering.rotated_ocr_pixel_dimensions(
            *ocr_rendering.ocr_render_pixel_dimensions(width_points, height_points, dpi),
            rendered.rotate,
        )
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
        if len(data) != width * height * 4:
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


configure_candidate_services(PdfHostServices())
