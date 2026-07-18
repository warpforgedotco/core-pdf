# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from core_layout.impl.layout.geometry import RectBox
from core_ocr.impl.coordinator import (
    PageExtractionHost,
    PageExtractionSnapshot,
    extract_page_text,
    resolved_line_content_record,
)
from core_ocr.impl.services import configure_candidate_services

from core_pdf.impl.engine.extraction.cache import (
    ExtractionCache,
)
from core_pdf.impl.engine.extraction.common import (
    page_profile,
)
from core_pdf.impl.engine.extraction.common.ordering import LayoutAnalyzer
from core_pdf.impl.engine.extraction.common.page_content import PageContentMixin
from core_pdf.impl.engine.extraction.common.render import (
    MarkdownRenderer,
)
from core_pdf.impl.engine.extraction.pdf_host_services import PdfHostServices
from core_pdf.impl.engine.rendering import RenderOptions, compose_page
from core_pdf.impl.models import TextSpan

configure_candidate_services(PdfHostServices())

if TYPE_CHECKING:
    from core_layout.impl.layout.models import LayoutLine


class PageExtractionMixin(PageContentMixin):
    extraction_cache: ExtractionCache | None
    text_lines: list[LayoutLine] | None
    text_spans: list[TextSpan] | None

    def get_page_profile(self: PageExtractionHost) -> page_profile.PageProfile:
        return page_profile.get_page_profile(self)

    def get_text_lines(self: PageExtractionHost) -> list[LayoutLine]:
        if self.text_lines is None:
            self.text_lines = LayoutAnalyzer.cluster_into_lines(
                [run for run in self.chars if run.text]
            )
        return self.text_lines

    def get_drawings(self: PageExtractionHost) -> list[dict[str, Any]]:
        graphics = self.get_graphics()
        return [
            {
                "kind": drawing.kind,
                "seqno": drawing.seqno,
                "fill": drawing.fill,
                "fill_pattern": drawing.fill_pattern,
                "fill_opacity": drawing.fill_opacity,
                "stroke_color": drawing.stroke_color,
                "stroke_pattern": drawing.stroke_pattern,
                "stroke_opacity": drawing.stroke_opacity,
                "line_width": drawing.line_width,
                "line_cap": drawing.line_cap,
                "line_join": drawing.line_join,
                "dash_pattern": drawing.dash_pattern,
                "fill_rule": drawing.fill_rule,
                "blend_mode": drawing.blend_mode,
                "soft_mask_alpha": drawing.soft_mask_alpha,
                "raw_data": drawing.raw_data,
                "dictionary": drawing.dictionary,
                "path": drawing.path,
                "items": list(drawing.items),
                "rect": drawing.rect,
            }
            for drawing in graphics.drawings
        ]

    def get_text_spans(self: PageExtractionHost) -> list[TextSpan]:
        if self.text_spans is None:
            state = (
                self.state
                if self.state is not None and self.state.glyphs
                else self.capture_text_state()
            )
            spans: dict[tuple[int, tuple[float, ...] | None, bool], TextSpan] = {}
            for glyph in state.glyphs:
                key = (glyph.seqno, glyph.fill, glyph.visible)
                span = spans.get(key)
                if span is None:
                    span = TextSpan(
                        seqno=glyph.seqno,
                        color=glyph.fill,
                        bbox=glyph.ink_rect,
                        chars=[],
                    )
                    spans[key] = span
                else:
                    rect = span["bbox"]
                    span["bbox"] = RectBox(
                        min(rect.x0, glyph.ink_rect.x0),
                        min(rect.y0, glyph.ink_rect.y0),
                        max(rect.x1, glyph.ink_rect.x1),
                        max(rect.y1, glyph.ink_rect.y1),
                        seqno=glyph.seqno,
                        fill=glyph.fill,
                        fill_opacity=None,
                    )
                for ch in glyph.text:
                    span["chars"].append((ord(ch), 0, 0, glyph.ink_rect))
            self.text_spans = list(spans.values())
        return self.text_spans

    def extract_text(self: PageExtractionHost) -> str:
        return extract_page_text(self)

    def extract_resolved_lines(self: PageExtractionHost) -> list[dict[str, Any]]:
        cache = self.extraction_cache
        if cache is None:
            self.extraction_cache = cache = ExtractionCache()
        self.extract_text()
        snapshot = cache.get_as("page_extraction_snapshot", PageExtractionSnapshot)
        if snapshot is None:
            return []
        return [
            resolved_line_content_record(line, line_index=line_index)
            for line_index, line in enumerate(snapshot.resolved_output_lines)
        ]

    def to_markdown(self: PageExtractionHost) -> str:
        return MarkdownRenderer.render_page(cast(Any, self))

    def render(self: PageExtractionHost, options: RenderOptions | None = None) -> Any:
        options = options or RenderOptions()
        cache = self.extraction_cache
        if cache is None:
            self.extraction_cache = cache = ExtractionCache()
        cache_key = (
            "rendered_page",
            options.page_number,
            options.rotate,
            options.crop,
            options.include_text,
            options.include_annotations,
            options.include_layers,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        rendered = compose_page(self, options)
        cache[cache_key] = rendered
        return rendered
