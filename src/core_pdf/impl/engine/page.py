# SPDX-License-Identifier: AGPL-3.0-only
"""Public page API over the canonical parse pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from core_pdf.impl.engine.layout.geometry_quality import (
    LayoutGeometrySummary,
    page_layout_geometry_issues,
    page_layout_geometry_summary,
    text_run_geometry_issues,
)
from core_pdf.impl.engine.layout.lines import LayoutLine
from core_pdf.impl.engine.model.geometry import rect_tuple
from core_pdf.impl.engine.model.runs import TextRun
from core_pdf.impl.engine.parse.pipeline import extract_page, page_extraction
from core_pdf.impl.engine.render.display import RenderOptions
from core_pdf.impl.engine.render.page import compose_page
from core_pdf.impl.engine.structured.model import TextDiagnostics
from core_pdf.impl.engine.structured.model import TextRun as StructuredTextRun
from core_pdf.impl.exceptions import PdfContractError
from core_pdf.impl.models import (
    DrawingRecord,
    ImageMetadata,
    ImageRecord,
)
from core_pdf.impl.runtime.cache import ExtractionCache
from core_pdf.impl.runtime.execution import RUNTIME
from core_pdf.impl.spec.s_07_document.page import PdfPage as SpecPdfPage
from core_pdf.impl.spec.s_08_graphics.image_decode import ImageSource


def text_rotation_correction_for_runs(runs: list[TextRun], threshold: float = 0.95) -> int:
    weighted: dict[int, int] = {}
    total = 0
    for run in runs:
        weight = len(run.text.strip())
        if not weight:
            continue
        angle = int(run.rotation_angle) % 360
        weighted[angle] = weighted.get(angle, 0) + weight
        total += weight
    if not total:
        return 0
    angle, weight = max(weighted.items(), key=lambda item: item[1])
    return (-angle) % 360 if weight / total >= threshold else 0


class PdfPage(SpecPdfPage):
    document: Any

    @property
    def structured_view(self) -> Any:
        """Return this page's canonical high-level structured representation."""
        return self.extract()

    @property
    def parse_report(self) -> Any | None:
        """Return the report owned by this page's base extraction, when materialized."""
        return page_extraction(self).report

    def internal_cache(self) -> ExtractionCache:
        cache = self.extraction_cache
        if cache is None:
            # The spec page always registers a cache at construction; a missing one
            # would silently escape document-level invalidation, so fail loudly.
            raise PdfContractError("page extraction cache was not initialized")
        return cache

    def extract(self) -> Any:
        with self.document.acquire_operation() as operation:
            with RUNTIME.task_scope(
                cancelled=lambda: operation.cancelled,
                metrics=True,
            ) as context:
                return extract_page(self, context)

    def text_diagnostics(self, *, include_invisible: bool = True) -> TextDiagnostics:
        with self.internal_page_lock:
            return TextDiagnostics(
                runs=tuple(
                    StructuredTextRun(
                        text=run.text,
                        bbox=(run.x0, run.y0, run.x1, run.y1),
                        font_name=run.font_name,
                        font_size=run.font_size,
                        is_vertical=run.is_vertical,
                        visible=run.visible,
                        rotation=run.rotation_angle,
                        seqno=run.seqno,
                        geometry_issues=text_run_geometry_issues(run),
                    )
                    for run in self.get_page_program().products.runs
                    if include_invisible or run.visible
                )
            )

    def get_text_lines(self) -> list[LayoutLine]:
        with self.internal_page_lock:
            if self.text_lines is None:
                self.text_lines = [LayoutLine([run]) for run in self.chars if run.text]
            return self.text_lines

    def extract_geometry_issues(self) -> tuple[object, ...]:
        with self.internal_page_lock:
            return page_layout_geometry_issues(self.get_text_lines())

    def extract_geometry_summary(self) -> LayoutGeometrySummary:
        with self.internal_page_lock:
            return page_layout_geometry_summary(self.get_text_lines())

    def get_drawings(self) -> tuple[DrawingRecord, ...]:
        cache_key = "page_drawing_records_v2"
        with self.internal_page_lock:
            cache = self.internal_cache()
            cached = cache.get_as(cache_key, tuple)
            if cached is not None:
                return cached
            result = tuple(
                DrawingRecord.from_captured(
                    drawing,
                    raw_data=bytes(drawing.raw_data) if drawing.raw_data is not None else None,
                    image_clip=rect_tuple(drawing.image_clip),
                    items=tuple(drawing.items),
                    rect=rect_tuple(drawing.rect),
                )
                for drawing in self.get_page_program().products.drawings
            )
            cache[cache_key] = result
            return result

    def extract_images(
        self,
        *,
        include_inline: bool = True,
        include_xobjects: bool = True,
    ) -> tuple[ImageRecord, ...]:
        images: list[ImageRecord] = []
        if include_xobjects:
            images.extend(
                ImageRecord.from_captured(drawing)
                for drawing in self.get_drawings()
                if drawing.kind == "image"
            )
        if include_inline:
            images.extend(
                ImageRecord(
                    kind="inline-image",
                    seqno=image.seqno,
                    fill=None,
                    fill_pattern=None,
                    fill_opacity=None,
                    stroke_color=None,
                    stroke_pattern=None,
                    stroke_opacity=None,
                    line_width=1.0,
                    line_cap=0,
                    line_join=0,
                    dash_pattern=None,
                    fill_rule="nonzero",
                    blend_mode=None,
                    soft_mask_alpha=None,
                    raw_data=image.data,
                    dictionary=image.dictionary,
                    image_source=image.image_source,
                    image_clip=image.image_clip,
                    path=None,
                    items=(),
                    rect=None,
                )
                for image in self.get_page_program().products.inline_images
            )
        for index, image in enumerate(images):
            source = cast(ImageSource | None, image.image_source)
            raster = source.decode() if source is not None else None
            if raster is not None:
                images[index] = replace(
                    image,
                    data=raster,
                    image_metadata=ImageMetadata(
                        width=raster.width,
                        height=raster.height,
                        channels=raster.channels,
                        color_model=raster.color_model,
                        alpha=raster.has_alpha,
                        stride=raster.stride,
                        source_rect=(0.0, 0.0, raster.width, raster.height),
                        transform=None,
                        clipping=rect_tuple(image.image_clip),
                    ),
                )
        return tuple(images)

    def render(self, options: RenderOptions | None = None) -> Any:
        options = options or RenderOptions()
        key = (
            "rendered_page_v2",
            options.rotate,
            options.crop,
            options.include_text,
            options.include_annotations,
            options.include_layers,
        )
        with self.internal_page_lock:
            cache = self.internal_cache()
            cached = cache.get(key)
            if cached is None:
                cached = compose_page(self, options, page_program=self.get_page_program())
                cache[key] = cached
            return cached


__all__ = ("PdfPage", "text_rotation_correction_for_runs")
