# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterator, Protocol, cast

from core_pdf.impl.engine.extraction.common.page_content import PageContentRecord
from core_pdf.impl.engine.extraction.tables.types import (
    Rect,
    TableHeaderResult,
    TableRows,
    TableSet,
    TableSpanResult,
    TableSpanRows,
)
from core_pdf.impl.engine.rendering import RenderOptions
from core_pdf.impl.types import PageSelection

class DocumentTablePage(Protocol):
    def extract_tables(self, **table_options: object) -> object: ...

    def extract_table_bboxes(self, **table_options: object) -> list[Rect | None]: ...

    def extract_lines(
        self, *, include_words: bool = False
    ) -> list[PageContentRecord]: ...


class DocumentExtractionMixin:
    @staticmethod
    def normalized_text(value: object) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).casefold()

    @classmethod
    def table_row_text(cls, row: object) -> str:
        if isinstance(row, (list, tuple)):
            parts = [str(cell).strip() for cell in row if str(cell).strip()]
            return cls.normalized_text(" ".join(parts))
        return cls.normalized_text(row)

    @classmethod
    def table_row_tokens(cls, row: object) -> list[str]:
        row_text = cls.table_row_text(row)
        if not row_text:
            return []
        return [token for token in row_text.split() if token]

    @staticmethod
    def record_bbox(
        item: Mapping[str, object],
    ) -> tuple[float, float, float, float] | None:
        value = item.get("bbox", item.get("rect"))
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            x0 = float(value[0])
            y0 = float(value[1])
            x1 = float(value[2])
            y1 = float(value[3])
        except TypeError, ValueError:
            return None
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    @classmethod
    def table_bbox_from_lines(
        cls,
        lines: list[PageContentRecord],
        rows: TableRows,
    ) -> tuple[float, float, float, float] | None:
        row_specs = [
            (cls.table_row_text(row), cls.table_row_tokens(row)) for row in rows
        ]
        row_specs = [
            (row_text, row_tokens)
            for row_text, row_tokens in row_specs
            if row_text or row_tokens
        ]
        if not row_specs:
            return None

        line_records: list[
            tuple[
                str,
                tuple[float, float, float, float],
                list[PageContentRecord] | None,
            ]
        ] = []
        for line in lines:
            bbox = cls.record_bbox(line)
            if bbox is None:
                continue
            line_text = cls.normalized_text(line.get("text"))
            if not line_text:
                continue
            words = line.get("words")
            line_records.append(
                (
                    line_text,
                    bbox,
                    cast(list[PageContentRecord], words)
                    if isinstance(words, list)
                    else None,
                )
            )
        if not line_records:
            return None

        matched_bboxes: list[tuple[float, float, float, float]] = []
        cursor = 0
        for row_text, row_tokens in row_specs:
            for index in range(cursor, len(line_records)):
                line_text, bbox, words = line_records[index]
                word_bbox = cls.table_row_bbox_from_words(words, row_tokens)
                if word_bbox is not None:
                    matched_bboxes.append(word_bbox)
                    cursor = index + 1
                    break
                if row_text in line_text or line_text in row_text:
                    matched_bboxes.append(bbox)
                    cursor = index + 1
                    break
        if not matched_bboxes:
            return None

        return (
            min(bbox[0] for bbox in matched_bboxes),
            min(bbox[1] for bbox in matched_bboxes),
            max(bbox[2] for bbox in matched_bboxes),
            max(bbox[3] for bbox in matched_bboxes),
        )

    @classmethod
    def table_row_bbox_from_words(
        cls,
        words: list[PageContentRecord] | None,
        tokens: list[str],
    ) -> tuple[float, float, float, float] | None:
        if not words or not tokens:
            return None
        normalized_words = [
            cls.normalized_text(word.get("text"))
            for word in words
            if isinstance(word, dict)
        ]
        for start in range(0, len(normalized_words) - len(tokens) + 1):
            if normalized_words[start : start + len(tokens)] != tokens:
                continue
            matched_boxes = [
                cls.record_bbox(words[index])
                for index in range(start, start + len(tokens))
            ]
            usable = [bbox for bbox in matched_boxes if bbox is not None]
            if usable:
                return (
                    min(bbox[0] for bbox in usable),
                    min(bbox[1] for bbox in usable),
                    max(bbox[2] for bbox in usable),
                    max(bbox[3] for bbox in usable),
                )
        return None

    def render_page(
        self: Any, page_index: int, options: RenderOptions | None = None
    ) -> object:
        if page_index < 0 or page_index >= len(self.pages):
            raise IndexError("page index out of range")
        return self.pages[page_index].render(options)

    def iter_content(
        self: Any,
        *,
        pages: PageSelection | None = None,
        include_words: bool = False,
        include_drawings: bool = True,
        include_images: bool = True,
        include_annotations: bool = True,
        use_resolved_text: bool = False,
    ) -> Iterator[PageContentRecord]:
        for page_index, page in self.iter_selected_pages(pages):
            page_number = getattr(page, "page_number", page_index + 1)
            page_label = getattr(page, "label", None)
            page_width = getattr(page, "width", 0.0)
            page_height = getattr(page, "height", 0.0)
            for item in page.extract_content(
                include_words=include_words,
                include_drawings=include_drawings,
                include_images=include_images,
                include_annotations=include_annotations,
                use_resolved_text=use_resolved_text,
            ):
                record = dict(item)
                record["page_index"] = page_index
                record["page_number"] = page_number
                record["page_label"] = page_label
                record["page_width"] = page_width
                record["page_height"] = page_height
                yield record

    def extract_content(
        self: Any,
        *,
        pages: PageSelection | None = None,
        include_words: bool = False,
        include_drawings: bool = True,
        include_images: bool = True,
        include_annotations: bool = True,
        use_resolved_text: bool = False,
    ) -> list[PageContentRecord]:
        return list(
            self.iter_content(
                pages=pages,
                include_words=include_words,
                include_drawings=include_drawings,
                include_images=include_images,
                include_annotations=include_annotations,
                use_resolved_text=use_resolved_text,
            )
        )

    def with_page_metadata(
        self: Any,
        page_index: int,
        page: object,
        item: Mapping[str, object],
    ) -> PageContentRecord:
        record: PageContentRecord = dict(item)
        record["page_index"] = page_index
        record["page_number"] = getattr(page, "page_number", page_index + 1)
        record["page_label"] = getattr(page, "label", None)
        record["page_width"] = getattr(page, "width", 0.0)
        record["page_height"] = getattr(page, "height", 0.0)
        return record

    def extract_text_runs(
        self: Any,
        *,
        pages: PageSelection | None = None,
        include_invisible: bool = True,
    ) -> list[PageContentRecord]:
        runs: list[PageContentRecord] = []
        for page_index, page in self.iter_selected_pages(pages):
            for run in page.extract_text_runs(include_invisible=include_invisible):
                runs.append(self.with_page_metadata(page_index, page, run))
        return runs

    def extract_words(
        self: Any, *, pages: PageSelection | None = None
    ) -> list[PageContentRecord]:
        words: list[PageContentRecord] = []
        for page_index, page in self.iter_selected_pages(pages):
            for word in page.extract_words():
                words.append(self.with_page_metadata(page_index, page, word))
        return words

    def extract_lines(
        self: Any,
        *,
        pages: PageSelection | None = None,
        include_words: bool = False,
    ) -> list[PageContentRecord]:
        lines: list[PageContentRecord] = []
        for page_index, page in self.iter_selected_pages(pages):
            for line in page.extract_lines(include_words=include_words):
                lines.append(self.with_page_metadata(page_index, page, line))
        return lines

    def extract_geometry_issues(
        self: Any,
        *,
        pages: PageSelection | None = None,
    ) -> list[PageContentRecord]:
        issues: list[PageContentRecord] = []
        for page_index, page in self.iter_selected_pages(pages):
            for issue in page.extract_geometry_issues():
                issues.append(self.with_page_metadata(page_index, page, issue))
        return issues

    def extract_geometry_summary(
        self: Any,
        *,
        pages: PageSelection | None = None,
    ) -> list[PageContentRecord]:
        summaries: list[PageContentRecord] = []
        for page_index, page in self.iter_selected_pages(pages):
            summaries.append(
                self.with_page_metadata(
                    page_index, page, page.extract_geometry_summary()
                )
            )
        return summaries

    def extract_images(
        self: Any,
        *,
        pages: PageSelection | None = None,
        include_inline: bool = True,
        include_xobjects: bool = True,
    ) -> list[PageContentRecord]:
        images: list[PageContentRecord] = []
        for page_index, page in self.iter_selected_pages(pages):
            for image in page.extract_images(
                include_inline=include_inline,
                include_xobjects=include_xobjects,
            ):
                images.append(self.with_page_metadata(page_index, page, image))
        return images

    def build_page_table_records(
        self: Any,
        page_index: int,
        page: DocumentTablePage,
        *,
        table_options: Mapping[str, object] | None = None,
    ) -> list[PageContentRecord]:
        table_records: list[PageContentRecord] = []
        options = dict(table_options or {})
        include_span_info = bool(options.get("include_span_info", False))
        detect_header = bool(options.get("detect_header", False))
        result = page.extract_tables(**options)
        bbox_options = dict(options)
        bbox_options.pop("include_span_info", None)
        page_bboxes = page.extract_table_bboxes(**bbox_options)
        page_spans: TableSpanRows | None = None
        if include_span_info:
            page_tables, page_spans = cast(TableSpanResult, result)
        elif detect_header:
            ignored_header, page_tables = cast(TableHeaderResult, result)
        else:
            page_tables = cast(TableSet, result)
        page_lines = page.extract_lines(include_words=True) if page_tables else []

        for table_index, table in enumerate(page_tables):
            record: PageContentRecord = {
                "table_index": table_index,
                "table": table,
                "rows": table,
            }
            if page_spans is not None and table_index < len(page_spans):
                record["spans"] = page_spans[table_index]
            bbox = (
                page_bboxes[table_index]
                if table_index < len(page_bboxes)
                else self.table_bbox_from_lines(page_lines, table)
            )
            if bbox is not None:
                record["bbox"] = bbox
            table_records.append(self.with_page_metadata(page_index, page, record))
        return table_records

    def extract_tables(
        self: Any, *, pages: PageSelection | None = None, **table_options: object
    ) -> list[PageContentRecord]:
        tables: list[PageContentRecord] = []
        for page_index, page in self.iter_selected_pages(pages):
            tables.extend(
                self.build_page_table_records(
                    page_index,
                    page,
                    table_options=dict(table_options),
                )
            )
        return tables

    def extract_annotations(
        self: Any, *, pages: PageSelection | None = None
    ) -> list[PageContentRecord]:
        annotations: list[PageContentRecord] = []
        for page_index, page in self.iter_selected_pages(pages):
            for annotation_index, annotation in enumerate(page.get_annotations()):
                record = {
                    "annotation_index": annotation_index,
                    "kind": "annotation",
                    "subtype": annotation.subtype,
                    "bbox": annotation.rect,
                    "contents": annotation.contents,
                    "dest": annotation.dest,
                    "action": annotation.action,
                    "dict": annotation.dict,
                }
                annotations.append(self.with_page_metadata(page_index, page, record))
        return annotations

    def extract_fields(
        self: Any, *, pages: PageSelection | None = None
    ) -> list[PageContentRecord]:
        fields: list[PageContentRecord] = []
        field_index = 0
        for page_index, page in self.iter_selected_pages(pages):
            for page_field_index, field in enumerate(page.get_fields()):
                record = {
                    "field_index": field_index,
                    "page_field_index": page_field_index,
                    "kind": "field",
                    "name": field.name,
                    "type": field.type,
                    "value": field.value,
                    "value_text": field.value_text,
                    "value_words": field.value_words(),
                    "bbox": field.rect,
                    "rect": field.rect,
                    "dict": field.dict,
                    "has_widget": field.widget is not None,
                    "kid_count": len(field.kids),
                }
                fields.append(self.with_page_metadata(page_index, page, record))
                field_index += 1
        return fields


__all__ = ("DocumentExtractionMixin", "DocumentTablePage")
