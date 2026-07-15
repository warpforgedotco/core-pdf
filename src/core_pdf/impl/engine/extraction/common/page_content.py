# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypedDict, cast

from core_pdf.impl.engine.extraction.cache import ExtractionCache
from core_pdf.impl.engine.extraction.common import page_geometry
from core_pdf.impl.engine.layout.geometry_quality import (
    LayoutGeometryIssueRecord,
    LayoutGeometrySummaryRecord,
    layout_geometry_summary_record,
    page_layout_geometry_issue_records,
    page_layout_geometry_summary,
    text_run_geometry_issue_records,
)

if TYPE_CHECKING:
    from core_pdf.impl.engine.layout.glyphs import GlyphCluster
    from core_pdf.impl.engine.layout.models import LayoutLine, TextRun


PageContentRecord: TypeAlias = dict[str, object]


class PageAnnotation(Protocol):
    @property
    def subtype(self) -> str | None: ...

    @property
    def rect(self) -> tuple[float, float, float, float] | None: ...

    @property
    def contents(self) -> str | None: ...

    @property
    def dest(self) -> object: ...

    @property
    def action(self) -> object: ...


class PageRenderItem(Protocol):
    @property
    def kind(self) -> str: ...

    @property
    def seqno(self) -> int: ...

    @property
    def data(self) -> Mapping[str, object]: ...


class PageRenderDisplayList(Protocol):
    @property
    def items(self) -> Sequence[PageRenderItem]: ...


class PageRenderResult(Protocol):
    @property
    def display_list(self) -> PageRenderDisplayList: ...


class GlyphClusterRecord(TypedDict):
    cluster_id: int
    text: str
    kind: str
    advance_bbox: tuple[float, float, float, float]
    ink_bbox: tuple[float, float, float, float]
    baseline: tuple[float, float, float, float] | None
    writing_mode: str
    rotation_angle: int
    font_name: str | None
    seqno: int
    confidence: float | None
    provenance: tuple[tuple[str, object], ...]
    glyph_count: int


class PageContentHost(Protocol):
    extraction_cache: ExtractionCache | None

    @property
    def chars(self) -> list[TextRun]: ...

    def get_text_lines(self) -> list[LayoutLine]: ...

    def get_drawings(self) -> list[PageContentRecord]: ...

    def get_annotations(self) -> Sequence[PageAnnotation]: ...

    def render(self) -> PageRenderResult: ...

    def extract_lines(self, *, include_words: bool = False) -> list[PageContentRecord]: ...

    def extract_images(
        self,
        *,
        include_inline: bool = True,
        include_xobjects: bool = True,
    ) -> list[PageContentRecord]: ...

    def extract_resolved_lines(self) -> list[PageContentRecord]: ...


class PageContentMixin:
    extraction_cache: ExtractionCache | None

    def get_page_observations(
        self: PageContentHost,
        *,
        include_images: bool = True,
    ) -> page_geometry.PageObservationSet:
        cache = self.extraction_cache
        if cache is None:
            self.extraction_cache = cache = ExtractionCache()
        cache_key = ("page_observations", include_images)
        cached = cache.get(cache_key)
        if isinstance(cached, page_geometry.PageObservationSet):
            return cached
        observations = page_geometry.collect_page_observations(
            self,
            include_images=include_images,
        )
        cache[cache_key] = observations
        return observations

    def extract_text_runs(
        self: PageContentHost,
        *,
        include_invisible: bool = True,
    ) -> list[PageContentRecord]:
        runs: list[PageContentRecord] = []
        for run in self.chars:
            if not include_invisible and not run.visible:
                continue
            runs.append(
                {
                    "text": run.text,
                    "bbox": (run.x0, run.y0, run.x1, run.y1),
                    "advance_bbox": run.advance_bbox,
                    "ink_bbox": run.ink_bbox,
                    "x0": run.x0,
                    "y0": run.y0,
                    "x1": run.x1,
                    "y1": run.y1,
                    "tx": run.tx,
                    "ty": run.ty,
                    "font_name": run.font_name,
                    "font_size": run.font_size,
                    "space_width": run.space_width,
                    "order": run.order,
                    "stream_order": run.stream_order,
                    "xobject_depth": run.xobject_depth,
                    "is_vertical": run.is_vertical,
                    "rotation_angle": run.rotation_angle,
                    "visible": run.visible,
                    "line_break_before": run.line_break_before,
                    "seqno": run.seqno,
                    "fill_color": run.fill_color,
                    "baseline": run.baseline,
                    "provenance": run.provenance,
                    "confidence": run.confidence,
                    "glyph_clusters": [
                        glyph_cluster_record(cluster) for cluster in run.glyph_clusters
                    ],
                    "geometry_issues": text_run_geometry_issue_records(run),
                    "bold": run.is_bold(),
                    "italic": run.is_italic(),
                }
            )
        return runs

    def extract_words(self: PageContentHost) -> list[PageContentRecord]:
        words: list[PageContentRecord] = []
        page_height = getattr(self, "height", None)
        for line_index, line in enumerate(self.get_text_lines()):
            line_bbox = (line.x0, line.y0, line.x1, line.y1)
            for word_index, word in enumerate(line.words()):
                word_bbox = word.bbox
                words.append(
                    {
                        "text": word.text,
                        "bbox": word_bbox,
                        "page_bbox": _page_bbox(word_bbox, page_height),
                        "x0": word_bbox[0],
                        "y0": word_bbox[1],
                        "x1": word_bbox[2],
                        "y1": word_bbox[3],
                        "start_index": word.start_index,
                        "word_index": word_index,
                        "line_index": line_index,
                        "line_bbox": line_bbox,
                        "line_page_bbox": _page_bbox(line_bbox, page_height),
                        "line_text": line.text(),
                        "rotation_angle": line.rotation_angle,
                        "is_vertical": line.is_vertical,
                        "min_order": line.min_order,
                        "max_order": line.max_order,
                        "max_depth": line.max_depth,
                    }
                )
        return words

    def extract_lines(
        self: PageContentHost, *, include_words: bool = False
    ) -> list[PageContentRecord]:
        lines: list[PageContentRecord] = []
        page_height = getattr(self, "height", None)
        for line_index, line in enumerate(self.get_text_lines()):
            line_bbox = (line.x0, line.y0, line.x1, line.y1)
            line_record: PageContentRecord = {
                "text": line.text(),
                "bbox": line_bbox,
                "page_bbox": _page_bbox(line_bbox, page_height),
                "x0": line.x0,
                "y0": line.y0,
                "x1": line.x1,
                "y1": line.y1,
                "line_index": line_index,
                "rotation_angle": line.rotation_angle,
                "is_vertical": line.is_vertical,
                "min_order": line.min_order,
                "max_order": line.max_order,
                "max_depth": line.max_depth,
                "mid_y": line.mid_y,
                "height": line.height,
                "max_font_size": line.max_font_size,
                "is_all_caps_text": line.is_all_caps_text,
                "run_count": len(line.runs),
            }
            if include_words:
                line_record["words"] = [
                    {
                        "text": word.text,
                        "bbox": word.bbox,
                        "page_bbox": _page_bbox(word.bbox, page_height),
                        "x0": word.bbox[0],
                        "y0": word.bbox[1],
                        "x1": word.bbox[2],
                        "y1": word.bbox[3],
                        "start_index": word.start_index,
                        "word_index": word_index,
                    }
                    for word_index, word in enumerate(line.words())
                ]
            lines.append(line_record)
        return lines

    def extract_images(
        self: PageContentHost,
        *,
        include_inline: bool = True,
        include_xobjects: bool = True,
    ) -> list[PageContentRecord]:
        kinds = set()
        if include_xobjects:
            kinds.add("image")
        if include_inline:
            kinds.add("inline-image")
        if not kinds:
            return []

        images: list[PageContentRecord] = []
        page_height = getattr(self, "height", None)
        for item in self.render().display_list.items:
            if item.kind not in kinds:
                continue
            metadata = item.data.get("image_metadata")
            if not isinstance(metadata, dict):
                continue
            image = cast(PageContentRecord, dict(metadata))
            image["kind"] = item.kind
            image["seqno"] = item.seqno
            bbox = page_geometry.rect_box_tuple(item.data.get("bbox"))
            if bbox is not None:
                image["bbox"] = bbox
                image["page_bbox"] = _page_bbox(bbox, page_height)
            soft_mask_alpha = item.data.get("soft_mask_alpha")
            if soft_mask_alpha is not None:
                image["soft_mask_alpha"] = soft_mask_alpha
            images.append(image)
        return images

    def image_metadata(self: PageContentHost) -> list[PageContentRecord]:
        return self.extract_images()

    def extract_content(
        self: PageContentHost,
        *,
        include_words: bool = False,
        include_drawings: bool = True,
        include_images: bool = True,
        include_annotations: bool = True,
        use_resolved_text: bool = False,
    ) -> list[PageContentRecord]:
        content: list[PageContentRecord] = []

        text_lines = (
            self.extract_resolved_lines()
            if use_resolved_text
            else self.extract_lines(include_words=include_words)
        )
        for line in text_lines:
            item = dict(line)
            item["kind"] = "text-line"
            item["seqno"] = item.get("min_order")
            content.append(item)

        if include_drawings:
            for drawing in self.get_drawings():
                kind = drawing.get("kind")
                if kind in {"image", "inline-image"}:
                    continue
                item = {
                    "kind": kind or "drawing",
                    "seqno": drawing.get("seqno"),
                    "bbox": drawing.get("rect"),
                    "fill": drawing.get("fill"),
                    "fill_opacity": drawing.get("fill_opacity"),
                    "stroke_color": drawing.get("stroke_color"),
                    "stroke_opacity": drawing.get("stroke_opacity"),
                    "line_width": drawing.get("line_width"),
                    "line_cap": drawing.get("line_cap"),
                    "line_join": drawing.get("line_join"),
                    "dash_pattern": drawing.get("dash_pattern"),
                    "fill_rule": drawing.get("fill_rule"),
                    "blend_mode": drawing.get("blend_mode"),
                    "soft_mask_alpha": drawing.get("soft_mask_alpha"),
                    "path": drawing.get("path"),
                    "items": drawing.get("items"),
                }
                content.append(item)

        if include_images:
            for image in self.extract_images():
                item = dict(image)
                item["kind"] = f"{image.get('kind')}"
                content.append(item)

        if include_annotations:
            for index, annotation in enumerate(self.get_annotations()):
                content.append(
                    {
                        "kind": "annotation",
                        "annotation_index": index,
                        "subtype": annotation.subtype,
                        "bbox": annotation.rect,
                        "contents": annotation.contents,
                        "dest": annotation.dest,
                        "action": annotation.action,
                    }
                )

        def sort_key(item: PageContentRecord) -> tuple[int, int, float, int]:
            seqno = item.get("seqno")
            seq_key = int(seqno) if type(seqno) is int else 1_000_000_000
            line_index = item.get("line_index")
            line_key = int(line_index) if type(line_index) is int else 1_000_000_000
            bbox = item.get("bbox")
            bbox_type = type(bbox)
            if bbox_type is list or bbox_type is tuple:
                bbox_seq = cast(list[object] | tuple[object, ...], bbox)
                y_value = bbox_seq[3] if len(bbox_seq) == 4 else None
                if isinstance(y_value, (int, float, str, bytes, bytearray)):
                    y_key = -float(y_value)
                else:
                    y_key = 0.0
            else:
                y_key = 0.0
            kind_rank = {
                "text-line": 0,
                "image": 1,
                "inline-image": 1,
                "annotation": 3,
            }.get(str(item.get("kind")), 2)
            return (seq_key, line_key, y_key, kind_rank)

        content.sort(key=sort_key)
        return content

    def extract_geometry_issues(
        self: PageContentHost,
    ) -> list[LayoutGeometryIssueRecord]:
        return page_layout_geometry_issue_records(self.get_text_lines())

    def extract_geometry_summary(self: PageContentHost) -> LayoutGeometrySummaryRecord:
        return layout_geometry_summary_record(page_layout_geometry_summary(self.get_text_lines()))

    def find_text_near(
        self: PageContentHost,
        target_box: tuple[float, float, float, float],
        direction: str = "left",
        distance: float = 100.0,
    ) -> list[TextRun]:
        x0, y0, x1, y1 = target_box
        runs = self.chars

        candidates: list[tuple[float, TextRun]] = []
        mid_x = (x0 + x1) * 0.5
        mid_y = (y0 + y1) * 0.5

        for run in runs:
            if not run.text.strip():
                continue

            run_mid_x = run.mid_x
            run_mid_y = run.mid_y

            dist = -1.0

            if direction == "left":
                if run.x1 <= x0 and abs(run_mid_y - mid_y) < max(run.height, y1 - y0, 10.0):
                    dist = x0 - run.x1
            elif direction == "right":
                if run.x0 >= x1 and abs(run_mid_y - mid_y) < max(run.height, y1 - y0, 10.0):
                    dist = run.x0 - x1
            elif direction == "above":
                if run.y0 >= y1 and abs(run_mid_x - mid_x) < max(run.x1 - run.x0, x1 - x0, 20.0):
                    dist = run.y0 - y1
            elif (
                direction == "below"
                and run.y1 <= y0
                and abs(run_mid_x - mid_x) < max(run.x1 - run.x0, x1 - x0, 20.0)
            ):
                dist = y0 - run.y1

            if 0 <= dist <= distance:
                candidates.append((dist, run))

        candidates.sort(key=lambda candidate: candidate[0])
        return [candidate[1] for candidate in candidates]


def glyph_cluster_record(cluster: GlyphCluster) -> GlyphClusterRecord:
    return {
        "cluster_id": cluster.cluster_id,
        "text": cluster.text,
        "kind": cluster.kind,
        "advance_bbox": cluster.advance_bbox,
        "ink_bbox": cluster.ink_bbox,
        "baseline": cluster.baseline,
        "writing_mode": cluster.writing_mode,
        "rotation_angle": cluster.rotation_angle,
        "font_name": cluster.font_name,
        "seqno": cluster.seqno,
        "confidence": cluster.confidence,
        "provenance": cluster.provenance,
        "glyph_count": len(cluster.glyphs),
    }


def _page_bbox(
    bbox: tuple[float, float, float, float],
    page_height: object,
) -> tuple[float, float, float, float] | None:
    if not isinstance(page_height, (int, float)):
        return None
    x0, y1, x1, y0 = bbox
    return (x0, float(page_height) - y0, x1, float(page_height) - y1)


__all__ = (
    "GlyphClusterRecord",
    "PageAnnotation",
    "PageContentHost",
    "PageContentMixin",
    "PageContentRecord",
    "PageRenderDisplayList",
    "PageRenderItem",
    "PageRenderResult",
)
