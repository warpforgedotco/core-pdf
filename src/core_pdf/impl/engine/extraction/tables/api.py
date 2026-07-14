# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Literal, Mapping, Sequence, overload

from core_pdf.impl.engine.extraction.tables.core import (
    PageTableCoreMixin,
)
from core_pdf.impl.engine.extraction.tables.exports import (
    PageTableExportMixin,
)
from core_pdf.impl.engine.extraction.tables.options import (
    derive_table_areas_from_markers,
    derive_table_areas_from_regions,
    normalize_text_markers,
    parse_columns,
    parse_table_areas,
    table_cache_key,
)
from core_pdf.impl.engine.extraction.tables.protocols import PageTableHost
from core_pdf.impl.engine.extraction.tables.text_geometry import (
    PageTableTextGeometryMixin,
)
from core_pdf.impl.engine.extraction.tables.types import (
    ExtractTablesResult,
    Rect,
    TableExtractionResult,
    TableHeaderResult,
    TableSet,
    TableSpanResult,
    TableSpanRows,
)


class PageTableMixin(PageTableTextGeometryMixin, PageTableExportMixin, PageTableCoreMixin):
    def extract_table_bboxes(
        self: PageTableHost,
        flavor: str = "lattice",
        detect_header: bool = False,
        header_text: str | Sequence[str] | None = None,
        footer_text: str | Sequence[str] | None = None,
        table_areas: Sequence[str] | None = None,
        table_regions: Sequence[str] | None = None,
        columns: Sequence[str] | Sequence[float] | None = None,
        split_text: bool = False,
        flag_size: bool = False,
        strip_text: str | Sequence[str] | None = None,
        replace_text: Mapping[str, str] | None = None,
        copy_text: str | Sequence[str] | None = None,
        shift_text: str | Sequence[str] | None = None,
        edge_tolerance: float = 50.0,
        row_tolerance: float = 2.0,
        column_tolerance: float = 8.0,
        **kwargs: object,
    ) -> list[Rect | None]:
        payload = self.table_extraction_payload(
            flavor=flavor,
            detect_header=detect_header,
            include_span_info=False,
            header_text=header_text,
            footer_text=footer_text,
            table_areas=table_areas,
            table_regions=table_regions,
            columns=columns,
            split_text=split_text,
            flag_size=flag_size,
            strip_text=strip_text,
            replace_text=replace_text,
            copy_text=copy_text,
            shift_text=shift_text,
            edge_tolerance=edge_tolerance,
            row_tolerance=row_tolerance,
            column_tolerance=column_tolerance,
            **kwargs,
        )
        return list(payload.get("bboxes", []))

    def table_extraction_payload(
        self: PageTableHost,
        flavor: str = "lattice",
        detect_header: bool = False,
        include_span_info: bool = False,
        header_text: str | Sequence[str] | None = None,
        footer_text: str | Sequence[str] | None = None,
        table_areas: Sequence[str] | None = None,
        table_regions: Sequence[str] | None = None,
        columns: Sequence[str] | Sequence[float] | None = None,
        split_text: bool = False,
        flag_size: bool = False,
        strip_text: str | Sequence[str] | None = None,
        replace_text: Mapping[str, str] | None = None,
        copy_text: str | Sequence[str] | None = None,
        shift_text: str | Sequence[str] | None = None,
        edge_tolerance: float = 50.0,
        row_tolerance: float = 2.0,
        column_tolerance: float = 8.0,
        **kwargs: object,
    ) -> TableExtractionResult:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported table extraction option(s): {unknown}")

        explicit_areas = parse_table_areas(table_areas)
        region_areas = parse_table_areas(table_regions)
        normalized_columns = parse_columns(columns)
        normalized_header_text = normalize_text_markers(header_text)
        normalized_footer_text = normalize_text_markers(footer_text)
        normalized_copy_text = tuple(normalize_text_markers(copy_text))
        normalized_shift_text = tuple(normalize_text_markers(shift_text))

        if (normalized_header_text or normalized_footer_text) and flavor == "lattice":
            raise ValueError("header_text and footer_text cannot be used with flavor='lattice'")

        extract_options = dict(
            flavor=flavor,
            detect_header=detect_header,
            include_span_info=include_span_info,
            header_text=tuple(normalized_header_text),
            footer_text=tuple(normalized_footer_text),
            columns=tuple(normalized_columns),
            edge_tolerance=edge_tolerance,
            row_tolerance=row_tolerance,
            column_tolerance=column_tolerance,
            split_text=split_text,
            flag_size=flag_size,
            strip_text=strip_text,
            replace_text=replace_text,
            copy_text=normalized_copy_text,
            shift_text=normalized_shift_text or (("l", "t") if flavor == "lattice" else ()),
        )

        resolved_table_areas: list[tuple[float, float, float, float]] = []
        if explicit_areas:
            resolved_table_areas.extend(explicit_areas)
        else:
            if normalized_header_text or normalized_footer_text:
                marker_areas = derive_table_areas_from_markers(
                    [run for run in self.chars if run.visible],
                    header_text=normalized_header_text,
                    footer_text=normalized_footer_text,
                    page_width=self.width,
                    page_height=self.height,
                    table_regions=region_areas,
                )
                resolved_table_areas.extend(marker_areas)

            if not resolved_table_areas:
                resolved_table_areas.extend(
                    derive_table_areas_from_regions(
                        [run for run in self.chars if run.visible], region_areas
                    )
                )

        if resolved_table_areas:
            if detect_header:
                raise ValueError("table_areas/table_regions are not supported with detect_header")

            tables: TableSet = []
            spans: TableSpanRows = []
            bboxes: list[Rect | None] = []
            for area in resolved_table_areas:
                table_page = self.crop(area)
                payload = table_page.table_extraction_payload(
                    **{
                        **extract_options,
                        "table_areas": None,
                        "table_regions": None,
                    }
                )
                tables.extend(payload["tables"])
                spans.extend(payload["spans"])
                payload_bboxes = list(payload.get("bboxes", []))
                if payload_bboxes:
                    bboxes.extend([bbox if bbox is not None else area for bbox in payload_bboxes])
                else:
                    bboxes.extend([area] * len(payload["tables"]))
            return PageTableCoreMixin.table_result(tables, spans=spans, bboxes=bboxes)

        cache_key = table_cache_key(extract_options)
        if cache_key in self.tables:
            return self.tables[cache_key]

        return self.extract_tables_core(**extract_options)

    @overload
    def extract_tables(
        self: PageTableHost,
        flavor: str = "lattice",
        detect_header: bool = False,
        include_span_info: Literal[True] = True,
        header_text: str | Sequence[str] | None = None,
        footer_text: str | Sequence[str] | None = None,
        table_areas: Sequence[str] | None = None,
        table_regions: Sequence[str] | None = None,
        columns: Sequence[str] | Sequence[float] | None = None,
        split_text: bool = False,
        flag_size: bool = False,
        strip_text: str | Sequence[str] | None = None,
        replace_text: Mapping[str, str] | None = None,
        copy_text: str | Sequence[str] | None = None,
        shift_text: str | Sequence[str] | None = None,
        edge_tolerance: float = 50.0,
        row_tolerance: float = 2.0,
        column_tolerance: float = 8.0,
        **kwargs: object,
    ) -> TableSpanResult: ...

    @overload
    def extract_tables(
        self: PageTableHost,
        flavor: str = "lattice",
        detect_header: Literal[True] = True,
        include_span_info: Literal[False] = False,
        header_text: str | Sequence[str] | None = None,
        footer_text: str | Sequence[str] | None = None,
        table_areas: Sequence[str] | None = None,
        table_regions: Sequence[str] | None = None,
        columns: Sequence[str] | Sequence[float] | None = None,
        split_text: bool = False,
        flag_size: bool = False,
        strip_text: str | Sequence[str] | None = None,
        replace_text: Mapping[str, str] | None = None,
        copy_text: str | Sequence[str] | None = None,
        shift_text: str | Sequence[str] | None = None,
        edge_tolerance: float = 50.0,
        row_tolerance: float = 2.0,
        column_tolerance: float = 8.0,
        **kwargs: object,
    ) -> TableHeaderResult: ...

    @overload
    def extract_tables(
        self: PageTableHost,
        flavor: str = "lattice",
        detect_header: Literal[False] = False,
        include_span_info: Literal[False] = False,
        header_text: str | Sequence[str] | None = None,
        footer_text: str | Sequence[str] | None = None,
        table_areas: Sequence[str] | None = None,
        table_regions: Sequence[str] | None = None,
        columns: Sequence[str] | Sequence[float] | None = None,
        split_text: bool = False,
        flag_size: bool = False,
        strip_text: str | Sequence[str] | None = None,
        replace_text: Mapping[str, str] | None = None,
        copy_text: str | Sequence[str] | None = None,
        shift_text: str | Sequence[str] | None = None,
        edge_tolerance: float = 50.0,
        row_tolerance: float = 2.0,
        column_tolerance: float = 8.0,
        **kwargs: object,
    ) -> TableSet: ...

    @overload
    def extract_tables(
        self: PageTableHost,
        flavor: str = "lattice",
        detect_header: bool = False,
        include_span_info: bool = False,
        header_text: str | Sequence[str] | None = None,
        footer_text: str | Sequence[str] | None = None,
        table_areas: Sequence[str] | None = None,
        table_regions: Sequence[str] | None = None,
        columns: Sequence[str] | Sequence[float] | None = None,
        split_text: bool = False,
        flag_size: bool = False,
        strip_text: str | Sequence[str] | None = None,
        replace_text: Mapping[str, str] | None = None,
        copy_text: str | Sequence[str] | None = None,
        shift_text: str | Sequence[str] | None = None,
        edge_tolerance: float = 50.0,
        row_tolerance: float = 2.0,
        column_tolerance: float = 8.0,
        **kwargs: object,
    ) -> ExtractTablesResult: ...

    def extract_tables(
        self: PageTableHost,
        flavor: str = "lattice",
        detect_header: bool = False,
        include_span_info: bool = False,
        header_text: str | Sequence[str] | None = None,
        footer_text: str | Sequence[str] | None = None,
        table_areas: Sequence[str] | None = None,
        table_regions: Sequence[str] | None = None,
        columns: Sequence[str] | Sequence[float] | None = None,
        split_text: bool = False,
        flag_size: bool = False,
        strip_text: str | Sequence[str] | None = None,
        replace_text: Mapping[str, str] | None = None,
        copy_text: str | Sequence[str] | None = None,
        shift_text: str | Sequence[str] | None = None,
        edge_tolerance: float = 50.0,
        row_tolerance: float = 2.0,
        column_tolerance: float = 8.0,
        **kwargs: object,
    ) -> ExtractTablesResult:
        payload = self.table_extraction_payload(
            flavor=flavor,
            detect_header=detect_header,
            include_span_info=include_span_info,
            header_text=header_text,
            footer_text=footer_text,
            table_areas=table_areas,
            table_regions=table_regions,
            columns=columns,
            split_text=split_text,
            flag_size=flag_size,
            strip_text=strip_text,
            replace_text=replace_text,
            copy_text=copy_text,
            shift_text=shift_text,
            edge_tolerance=edge_tolerance,
            row_tolerance=row_tolerance,
            column_tolerance=column_tolerance,
            **kwargs,
        )
        if detect_header and not include_span_info:
            return payload["header"], payload["tables"]
        if include_span_info:
            return payload["tables"], payload["spans"]
        return payload["tables"]


__all__ = ("PageTableMixin",)
