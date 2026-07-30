# SPDX-License-Identifier: AGPL-3.0-only
"""Public page API over the canonical parse pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from core_pdf.impl.engine.cache import ExtractionCache
from core_pdf.impl.engine.execution import RUNTIME
from core_pdf.impl.engine.layout import (
    LayoutGeometrySummary,
    LayoutLine,
    TextRun,
    page_layout_geometry_issues,
    page_layout_geometry_summary,
    text_run_geometry_issue_records,
)
from core_pdf.impl.engine.layout.geometry import BBox, rect_tuple
from core_pdf.impl.engine.parse import extract_page, page_extraction
from core_pdf.impl.engine.rendering import RenderOptions, compose_page
from core_pdf.impl.engine.spec.s_07_document.page import PdfPage as SpecPdfPage
from core_pdf.impl.engine.spec.s_08_graphics.image_decode import ImageSource
from core_pdf.impl.models import (
    AnnotationContentRecord,
    DrawingRecord,
    ImageMetadata,
    ImageRecord,
    LineRecord,
    TextRunRecord,
    WordRecord,
)


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

    def internal_cache(self) -> ExtractionCache:
        cache = self.extraction_cache
        if cache is None:
            cache = ExtractionCache()
            self.extraction_cache = cache
        return cache

    def extract(self) -> Any:
        with self.document.acquire_operation() as operation:
            with RUNTIME.task_scope(
                cancelled=lambda: operation.cancelled,
                metrics=True,
            ) as context:
                return extract_page(self, context)

    def extract_text(self) -> str:
        return "\n".join(record.text for record in self.internal_line_records())

    def internal_line_records(self) -> tuple[LineRecord, ...]:
        with self.document.acquire_operation() as operation:
            with RUNTIME.task_scope(
                cancelled=lambda: operation.cancelled,
                metrics=True,
            ) as context:
                with self.internal_page_lock:
                    return page_extraction(self).line_records(context)

    def extract_text_runs(self, *, include_invisible: bool = True) -> tuple[TextRunRecord, ...]:
        return tuple(
            TextRunRecord(
                text=run.text,
                bbox=(run.x0, run.y0, run.x1, run.y1),
                font_name=run.font_name,
                font_size=run.font_size,
                is_vertical=run.is_vertical,
                visible=run.visible,
                rotation=run.rotation_angle,
                seqno=run.seqno,
                geometry_issues=tuple(text_run_geometry_issue_records(run)),
            )
            for run in self.get_page_program().products.runs
            if include_invisible or run.visible
        )

    def get_text_lines(self) -> list[LayoutLine]:
        if self.text_lines is None:
            self.text_lines = [LayoutLine([run]) for run in self.chars if run.text]
        return self.text_lines

    def extract_geometry_issues(self) -> tuple[object, ...]:
        return page_layout_geometry_issues(self.get_text_lines())

    def extract_geometry_summary(self) -> LayoutGeometrySummary:
        return page_layout_geometry_summary(self.get_text_lines())

    def extract_lines(self, *, include_words: bool = False) -> tuple[LineRecord, ...]:
        records = self.internal_line_records()
        if not include_words:
            return tuple(records)
        return tuple(
            replace(
                record,
                words=tuple(
                    WordRecord(
                        text=word,
                        bbox=record.bbox or (0.0, 0.0, 0.0, 0.0),
                        line_index=index,
                        word_index=word_index,
                        source=record.source,
                    )
                    for word_index, word in enumerate(record.text.split())
                ),
            )
            for index, record in enumerate(records)
        )

    def extract_words(self) -> tuple[WordRecord, ...]:
        words: list[WordRecord] = []
        for line_index, line in enumerate(self.internal_line_records()):
            for word_index, word in enumerate(line.text.split()):
                words.append(
                    WordRecord(
                        text=word,
                        bbox=line.bbox or (0.0, 0.0, 0.0, 0.0),
                        line_index=line_index,
                        word_index=word_index,
                        source=line.source,
                    )
                )
        return tuple(words)

    def get_drawings(self) -> tuple[DrawingRecord, ...]:
        cache_key = "page_drawing_records_v2"
        cache = self.internal_cache()
        cached = cache.get(cache_key)
        if isinstance(cached, tuple):
            cached_drawings = tuple(item for item in cached if isinstance(item, DrawingRecord))
            if len(cached_drawings) == len(cached):
                return cached_drawings
        records = [
            DrawingRecord(
                kind=drawing.kind,
                seqno=drawing.seqno,
                fill=drawing.fill,
                fill_pattern=drawing.fill_pattern,
                fill_opacity=drawing.fill_opacity,
                stroke_color=drawing.stroke_color,
                stroke_pattern=drawing.stroke_pattern,
                stroke_opacity=drawing.stroke_opacity,
                line_width=drawing.line_width,
                line_cap=drawing.line_cap,
                line_join=drawing.line_join,
                dash_pattern=drawing.dash_pattern,
                fill_rule=drawing.fill_rule,
                blend_mode=drawing.blend_mode,
                soft_mask_alpha=drawing.soft_mask_alpha,
                raw_data=bytes(drawing.raw_data) if drawing.raw_data is not None else None,
                dictionary=drawing.dictionary,
                image_source=drawing.image_source,
                image_clip=rect_tuple(drawing.image_clip),
                path=drawing.path,
                items=tuple(drawing.items),
                rect=rect_tuple(drawing.rect),
            )
            for drawing in self.get_page_program().products.drawings
        ]
        result = tuple(records)
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
                ImageRecord(
                    kind=drawing.kind,
                    seqno=drawing.seqno,
                    fill=drawing.fill,
                    fill_pattern=drawing.fill_pattern,
                    fill_opacity=drawing.fill_opacity,
                    stroke_color=drawing.stroke_color,
                    stroke_pattern=drawing.stroke_pattern,
                    stroke_opacity=drawing.stroke_opacity,
                    line_width=drawing.line_width,
                    line_cap=drawing.line_cap,
                    line_join=drawing.line_join,
                    dash_pattern=drawing.dash_pattern,
                    fill_rule=drawing.fill_rule,
                    blend_mode=drawing.blend_mode,
                    soft_mask_alpha=drawing.soft_mask_alpha,
                    raw_data=drawing.raw_data,
                    dictionary=drawing.dictionary,
                    image_source=drawing.image_source,
                    image_clip=drawing.image_clip,
                    path=drawing.path,
                    items=drawing.items,
                    rect=drawing.rect,
                )
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
                        clipping=(
                            BBox.from_rect(image.image_clip)
                            if image.image_clip is not None
                            else None
                        ),
                    ),
                )
        return tuple(images)

    def extract_content(
        self,
        *,
        include_words: bool = False,
        include_drawings: bool = True,
        include_images: bool = True,
        include_annotations: bool = True,
    ) -> tuple[object, ...]:
        content: list[object] = list(self.extract_lines(include_words=include_words))
        if include_drawings:
            content.extend(
                drawing
                for drawing in self.get_drawings()
                if drawing.kind not in {"image", "inline-image"}
            )
        if include_images:
            content.extend(self.extract_images())
        if include_annotations:
            content.extend(
                AnnotationContentRecord(annotation.subtype, annotation.rect, annotation.contents)
                for annotation in self.get_annotations()
            )
        return tuple(content)

    def extract_tables(self) -> list[list[list[str]]]:
        return [
            [[cell.text for cell in row] for row in table.rows]
            for table in self.internal_extracted_tables()
        ]

    def extract_table_bboxes(self) -> list[tuple[float, float, float, float]]:
        return [table.bbox for table in self.internal_extracted_tables() if table.bbox is not None]

    def internal_extracted_tables(self) -> tuple[Any, ...]:
        with self.document.acquire_operation() as operation:
            with RUNTIME.task_scope(
                cancelled=lambda: operation.cancelled,
                metrics=True,
            ) as context:
                with self.internal_page_lock:
                    return page_extraction(self).tables(context)

    def table_extraction_payload(self) -> dict[str, object]:
        tables = self.internal_extracted_tables()
        return {
            "tables": [[[cell.text for cell in row] for row in table.rows] for table in tables],
            "spans": [
                [
                    [
                        {
                            "row_span": cell.row_span,
                            "col_span": cell.column_span,
                        }
                        for cell in row
                    ]
                    for row in table.rows
                ]
                for table in tables
            ],
            "bboxes": [table.bbox for table in tables],
            "header": [],
        }

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

    def to_markdown(self) -> str:
        return self.extract().to_markdown()

    def text_rotation_correction(self, threshold: float = 0.95) -> int:
        return text_rotation_correction_for_runs(self.chars, threshold)


__all__ = ("PdfPage", "text_rotation_correction_for_runs")
