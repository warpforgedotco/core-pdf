# SPDX-License-Identifier: AGPL-3.0-only
"""Protocols used to connect OCR candidate generation to a host application."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from core_ocr.impl.types import OcrImage, OcrTextResult

_candidate_services: OcrCandidateServices | None = None


class _ServiceModuleProxy:
    def __init__(self, attribute: str) -> None:
        self.attribute = attribute

    def __getattr__(self, name: str) -> Any:
        return getattr(getattr(get_candidate_services(), self.attribute), name)


def service_module(attribute: str) -> Any:
    return _ServiceModuleProxy(attribute)


def service_function(attribute: str) -> Any:
    def call(*args: Any, **kwargs: Any) -> Any:
        return getattr(get_candidate_services(), attribute)(*args, **kwargs)

    return call


def configure_candidate_services(services: OcrCandidateServices) -> None:
    global _candidate_services
    _candidate_services = services


def get_candidate_services() -> OcrCandidateServices:
    if _candidate_services is None:
        raise RuntimeError("OCR candidate services have not been configured")
    return _candidate_services


class OcrCandidateServices(Protocol):
    """Host services required by OCR candidate generation."""

    page_geometry: Any
    iterator_layout: Any
    layout: Any
    selection: Any
    rectangle_request_type: Any
    dense_vector_render_tile_min_tokens: int
    rendering: Any
    layout_analyzer: Any
    pdf_analysis: Any
    observation_resolver: Any
    page_profile: Any
    render_options: Any
    image_color_manager: Any
    markdown_renderer: Any
    render_resolved_text_lines: Any
    render_page_observation_lines: Any
    render_page_text: Any
    compose_page: Any
    apply_flate: Any
    decode_stream_data: Any
    lookup_dict_key: Any
    image_filter_names: Any
    pdf_int: Any
    page_extraction_decision: Any
    layout_geometry_summary_record: Any
    page_layout_geometry_summary: Any
    native_text: Any
    detect_grid: Any
    merge_grids: Any
    rotate_page_lines: Any

    def render_page_for_ocr(self, page: Any, *, include_text: bool) -> Any: ...

    def rasterize_ocr_page(self, rendered: Any, dpi: int) -> OcrImage | None: ...

    def rasterize_ocr_tile(
        self,
        rendered: Any,
        tile: Any,
        dpi: int,
        *,
        source: str,
    ) -> OcrImage | None: ...

    def render_page_for_ocr_analysis(self, page: Any) -> Any: ...

    def clamp_ocr_bbox(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None: ...

    def crop_ocr_image_region(
        self,
        image: OcrImage,
        rectangle: tuple[int, int, int, int],
    ) -> OcrImage | None: ...

    def ocr_image_regions_to_text_results(
        self,
        image: OcrImage,
        requests: list[Any],
        timeout: float | None,
    ) -> list[OcrTextResult]: ...

    def ocr_image_to_text_result_with_psm(
        self,
        image: OcrImage,
        *,
        psm: int,
        timeout: float | None,
        variables: Mapping[str, str | int | float | bool] | None = None,
    ) -> OcrTextResult: ...
