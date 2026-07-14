# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import csv
import html
import io
from html.parser import HTMLParser
from typing import Sequence, cast

from core_pdf.impl.engine.extraction.tables.protocols import PageTableHost
from core_pdf.impl.engine.extraction.tables.quality import (
    bbox_overlap_ratio,
    bboxes_match,
    table_quality_score,
    table_text_length,
)
from core_pdf.impl.engine.extraction.tables.types import TableSet


def normalize_output_rows(
    rows: list[list[str]],
    *,
    normalize_whitespace: bool = False,
) -> list[list[str]]:
    non_empty_rows = [row for row in rows if any(str(cell).strip() for cell in row)]
    if not non_empty_rows:
        return []
    width = max(len(row) for row in non_empty_rows)
    non_empty_rows = [row + [""] * (width - len(row)) for row in non_empty_rows]

    def cell_text(cell: str) -> str:
        value = str(cell).strip()
        return " ".join(value.split()) if normalize_whitespace else value

    return [[cell_text(cell) for cell in row] for row in non_empty_rows]


def rows_to_html(
    rows: list[list[str]],
    *,
    normalize_whitespace: bool = False,
) -> str:
    output_rows = normalize_output_rows(
        rows,
        normalize_whitespace=normalize_whitespace,
    )
    if not output_rows:
        return ""
    parts = ["<table>"]
    header, *body = output_rows
    parts.append("<thead><tr>")
    for cell in header:
        parts.append(f"<th>{html.escape(cell)}</th>")
    parts.append("</tr></thead>")
    if body:
        parts.append("<tbody>")
        for row in body:
            parts.append("<tr>")
            for cell in row:
                parts.append(f"<td>{html.escape(cell)}</td>")
            parts.append("</tr>")
        parts.append("</tbody>")
    parts.append("</table>")
    return "".join(parts)


def rows_to_tsv(
    rows: list[list[str]],
    *,
    normalize_whitespace: bool = False,
) -> str:
    output_rows = normalize_output_rows(
        rows,
        normalize_whitespace=normalize_whitespace,
    )
    return "\n".join("\t".join(row) for row in output_rows)


def rows_to_csv(
    rows: list[list[str]],
    *,
    normalize_whitespace: bool = False,
) -> str:
    output_rows = normalize_output_rows(
        rows,
        normalize_whitespace=normalize_whitespace,
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(output_rows)
    return buffer.getvalue().rstrip("\n")


def html_table_text(html_table: str) -> str:
    result = []
    in_tag = False
    for char in html_table:
        if char == "<":
            in_tag = True
        elif char == ">":
            in_tag = False
        elif not in_tag:
            result.append(char)
    return html.unescape("".join(result)).strip()


def html_table_plain_text(html_table: str) -> str:
    parser = PlainTextTableParser()
    parser.feed(html_table)
    parser.close()
    return "\n".join("\t".join(row) for row in parser.rows).strip()


class PlainTextTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"}:
            self.current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"}:
            if self.current_row is not None and self.current_cell is not None:
                self.current_row.append("".join(self.current_cell).strip())
            self.current_cell = None
        elif tag == "tr":
            if self.current_row is not None:
                self.rows.append(self.current_row)
            self.current_row = None

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)


class PageTableExportMixin:
    def extract_table_html(
        self: PageTableHost,
        bbox: tuple[float, float, float, float],
        *,
        min_text_coverage: float = 0.85,
        flavor: str | Sequence[str] = ("lattice", "stream"),
        **extract_options: object,
    ) -> str:
        rows = self.extract_table_rows(
            bbox,
            min_text_coverage=min_text_coverage,
            flavor=flavor,
            **extract_options,
        )
        return rows_to_html(rows) if rows else ""

    def extract_table_csv(
        self: PageTableHost,
        bbox: tuple[float, float, float, float],
        *,
        min_text_coverage: float = 0.85,
        normalize_whitespace: bool = False,
        flavor: str | Sequence[str] = ("lattice", "stream"),
        **extract_options: object,
    ) -> str:
        rows = self.extract_table_rows(
            bbox,
            min_text_coverage=min_text_coverage,
            flavor=flavor,
            **extract_options,
        )
        return (
            rows_to_csv(
                rows,
                normalize_whitespace=normalize_whitespace,
            )
            if rows
            else ""
        )

    def extract_table_tsv(
        self: PageTableHost,
        bbox: tuple[float, float, float, float],
        *,
        min_text_coverage: float = 0.85,
        normalize_whitespace: bool = False,
        flavor: str | Sequence[str] = ("lattice", "stream"),
        **extract_options: object,
    ) -> str:
        rows = self.extract_table_rows(
            bbox,
            min_text_coverage=min_text_coverage,
            flavor=flavor,
            **extract_options,
        )
        return (
            rows_to_tsv(
                rows,
                normalize_whitespace=normalize_whitespace,
            )
            if rows
            else ""
        )

    def extract_table_rows(
        self: PageTableHost,
        bbox: tuple[float, float, float, float],
        *,
        min_text_coverage: float = 0.85,
        flavor: str | Sequence[str] = ("lattice", "stream"),
        **extract_options: object,
    ) -> list[list[str]]:
        flavors = (flavor,) if isinstance(flavor, str) else tuple(flavor)
        page_level_candidates: list[
            tuple[int, float, float, int, int, int, list[list[str]]]
        ] = []
        for current_flavor in flavors:
            page_tables = cast(
                TableSet,
                self.extract_tables(
                    flavor=current_flavor,
                    **extract_options,
                ),
            )
            page_bboxes = self.extract_table_bboxes(
                flavor=current_flavor,
                **extract_options,
            )
            for table_index, page_bbox in enumerate(page_bboxes):
                if table_index >= len(page_tables):
                    continue
                if page_bbox is None:
                    continue
                overlap = bbox_overlap_ratio(bbox, page_bbox)
                if not bboxes_match(bbox, page_bbox) and overlap < 0.85:
                    continue
                table = page_tables[table_index]
                substantial, quality, text_length, populated_cells, negative_total = (
                    table_quality_score(
                        table,
                        text_length=table_text_length(table),
                    )
                )
                page_level_candidates.append(
                    (
                        substantial,
                        quality,
                        overlap,
                        text_length,
                        populated_cells,
                        negative_total,
                        table,
                    )
                )
        if page_level_candidates:
            return max(
                page_level_candidates,
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2],
                    item[3],
                    item[4],
                    item[5],
                ),
            )[6]

        cropped_page = self.crop(bbox)
        visible_text_len = len(
            "".join(run.text for run in cropped_page.chars if run.visible).strip()
        )
        candidates: list[tuple[float, int, int, list[list[str]]]] = []
        for current_flavor in flavors:
            cropped_tables = cast(
                TableSet,
                cropped_page.extract_tables(
                    flavor=current_flavor,
                    **extract_options,
                ),
            )
            for table in cropped_tables:
                html_text = rows_to_html(table)
                if not html_text:
                    continue
                if visible_text_len <= 0:
                    coverage = 1.0
                else:
                    table_text = html_table_text(html_text)
                    if not table_text:
                        continue
                    coverage = len(table_text) / visible_text_len
                    if coverage < min_text_coverage:
                        continue
                populated_cells = sum(
                    1 for row in table for cell in row if str(cell).strip()
                )
                total_cells = sum(len(row) for row in table)
                candidates.append((coverage, populated_cells, -total_cells, table))
        if not candidates:
            return []
        return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


__all__ = (
    "PageTableExportMixin",
    "html_table_plain_text",
    "html_table_text",
    "normalize_output_rows",
    "rows_to_csv",
    "rows_to_html",
    "rows_to_tsv",
)
