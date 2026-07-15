# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypedDict, cast

from core_pdf.impl.engine.extraction.tables.extract import extract_grid, extract_heuristic
from core_pdf.impl.engine.extraction.tables.grid import detect_grids_from_edges, merge_grids
from core_pdf.impl.engine.extraction.tables.options import table_cache_key
from core_pdf.impl.engine.extraction.tables.protocols import PageTableHost
from core_pdf.impl.engine.extraction.tables.quality import (
    bbox_overlap_ratio,
    table_quality_score,
    table_text_length,
)
from core_pdf.impl.engine.extraction.tables.stream import (
    detect_network_grids,
    detect_stream_grid,
    detect_stream_grids,
)
from core_pdf.impl.engine.extraction.tables.types import (
    Rect,
    TableCellSpanRecord,
    TableExtractionResult,
    TableHeaderResultWithBBoxes,
    TableRows,
    TableSet,
    TableSetWithBBoxes,
    TableSpanResultWithBBoxes,
    TableSpanRows,
)
from core_pdf.impl.engine.layout.models import TableGrid, TextRun
from core_pdf.impl.engine.spec.s_07_document.page_boxes import (
    rotate_page_lines,
    rotate_page_runs,
)


class GridExtractOptions(TypedDict, total=False):
    split_text: bool
    flag_size: bool
    strip_text: str | Sequence[str] | None
    replace_text: Mapping[str, str] | None
    copy_text: Sequence[str] | None
    shift_text: Sequence[str] | None


class HeuristicExtractOptions(GridExtractOptions, total=False):
    row_tolerance: float
    column_tolerance: float
    columns: list[float]


@dataclass(frozen=True)
class TableExtractionOptions:
    flavor: str
    canonicalize: bool
    detect_header: bool
    include_span_info: bool
    columns: tuple[float, ...]
    edge_tolerance: float
    row_tolerance: float
    column_tolerance: float
    split_text: bool
    flag_size: bool
    strip_text: str | Sequence[str] | None
    replace_text: Mapping[str, str] | None
    copy_text: Sequence[str]
    shift_text: Sequence[str]


@dataclass(frozen=True)
class TableExtractionContext:
    visible_runs: list[TextRun]
    grids: list[TableGrid]
    grid: TableGrid | None
    run_chars: dict[int, list[tuple[str, float, float, float, float]]] | None


class PageTableCoreMixin:
    @staticmethod
    def table_result(
        tables: TableSet,
        *,
        spans: TableSpanRows | None = None,
        bboxes: list[Rect | None] | None = None,
        header: TableRows | None = None,
    ) -> TableExtractionResult:
        return {
            "tables": tables,
            "spans": spans or [],
            "bboxes": bboxes or [None] * len(tables),
            "header": header or [],
        }

    @staticmethod
    def table_bbox_from_grid(
        table_grid: TableGrid,
    ) -> tuple[float, float, float, float]:
        return (
            float(table_grid.cols[0]),
            float(table_grid.rows[-1]),
            float(table_grid.cols[-1]),
            float(table_grid.rows[0]),
        )

    @staticmethod
    def table_bbox_from_runs(
        runs: Sequence[TextRun],
    ) -> tuple[float, float, float, float] | None:
        if not runs:
            return None
        return (
            min(float(run.x0) for run in runs),
            min(float(run.y0) for run in runs),
            max(float(run.x1) for run in runs),
            max(float(run.y1) for run in runs),
        )

    @staticmethod
    def total_bbox_area(
        bboxes: Sequence[tuple[float, float, float, float] | None],
    ) -> float:
        total = 0.0
        for bbox in bboxes:
            if bbox is None:
                continue
            total += max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        return total

    @staticmethod
    def nonempty_cell_count(table: Sequence[Sequence[object]]) -> int:
        return sum(1 for row in table for cell in row if str(cell).strip())

    @staticmethod
    def column_count(table: Sequence[Sequence[object]]) -> int:
        return max((len(row) for row in table), default=0)

    @staticmethod
    def nonempty_cells(table: Sequence[Sequence[object]]) -> list[str]:
        return [str(cell).strip() for row in table for cell in row if str(cell).strip()]

    @classmethod
    def substantial_table_indexes(cls, tables: Sequence[Sequence[Sequence[object]]]) -> list[int]:
        substantial: list[int] = []
        for index, table in enumerate(tables):
            row_count = len(table)
            col_count = max((len(row) for row in table), default=0)
            if row_count < 2 or col_count < 2:
                continue
            if cls.nonempty_cell_count(table) < 4:
                continue
            substantial.append(index)
        return substantial

    @classmethod
    def filtered_substantial_result(cls, result: TableExtractionResult) -> TableExtractionResult:
        indexes = cls.substantial_table_indexes(result.get("tables", []))
        if not indexes:
            return cls.table_result([])
        tables = [result["tables"][index] for index in indexes]
        spans = [result.get("spans", [])[index] for index in indexes] if result.get("spans") else []
        bboxes = (
            [result.get("bboxes", [])[index] for index in indexes] if result.get("bboxes") else []
        )
        return cls.table_result(tables, spans=spans, bboxes=bboxes)

    @classmethod
    def prune_trivial_fragments(cls, result: TableExtractionResult) -> TableExtractionResult:
        filtered = cls.filtered_substantial_result(result)
        if len(filtered.get("tables", [])) == len(result.get("tables", [])):
            return result
        if filtered.get("tables"):
            return filtered
        return result

    @classmethod
    def suppress_probable_prose_result(cls, result: TableExtractionResult) -> TableExtractionResult:
        tables = result.get("tables", [])
        if len(tables) != 1:
            return result
        table = tables[0]
        row_count = len(table)
        col_count = cls.column_count(table)
        cells = cls.nonempty_cells(table)
        if row_count < 20 or col_count > 3 or not cells:
            return result
        average_length = sum(len(cell) for cell in cells) / len(cells)
        long_ratio = sum(1 for cell in cells if len(cell) >= 25) / len(cells)
        numeric_ratio = sum(
            1 for cell in cells if cell.replace(",", "").replace(".", "").isdigit()
        ) / len(cells)
        if average_length >= 35.0 and long_ratio >= 0.75 and numeric_ratio <= 0.05:
            return cls.table_result([])
        return result

    @classmethod
    def normalize_result(cls, result: TableExtractionResult) -> TableExtractionResult:
        result = cls.prune_trivial_fragments(result)
        result = cls.suppress_probable_prose_result(result)
        return result

    @classmethod
    def is_tiny_single_result(cls, result: TableExtractionResult) -> bool:
        tables = result.get("tables", [])
        if len(tables) != 1:
            return False
        table = tables[0]
        return len(table) <= 2 and cls.nonempty_cell_count(table) <= 8

    @classmethod
    def total_nonempty_cells(cls, tables: Sequence[Sequence[Sequence[object]]]) -> int:
        return sum(cls.nonempty_cell_count(table) for table in tables)

    @classmethod
    def total_column_count(cls, tables: Sequence[Sequence[Sequence[Any]]]) -> int:
        return sum(cls.column_count(table) for table in tables)

    @classmethod
    def should_prefer_ruled_result(
        cls,
        candidate: TableExtractionResult,
        ruled: TableExtractionResult,
    ) -> bool:
        candidate = cls.filtered_substantial_result(candidate)
        ruled = cls.filtered_substantial_result(ruled)
        candidate_tables = candidate.get("tables", [])
        ruled_tables = ruled.get("tables", [])
        if not ruled_tables:
            return False
        if not candidate_tables:
            return True
        if len(ruled_tables) < len(candidate_tables):
            return False
        candidate_area = cls.total_bbox_area(candidate.get("bboxes", []))
        ruled_area = cls.total_bbox_area(ruled.get("bboxes", []))
        if ruled_area <= 0.0:
            return False
        if candidate_area < ruled_area * 0.6:
            return True

        candidate_nonempty = cls.total_nonempty_cells(candidate_tables)
        ruled_nonempty = cls.total_nonempty_cells(ruled_tables)
        if ruled_area >= candidate_area * 0.9 and ruled_nonempty >= candidate_nonempty + 8:
            return True
        return bool(
            ruled_area >= candidate_area * 0.9
            and ruled_nonempty + 4 >= candidate_nonempty
            and cls.total_column_count(ruled_tables) >= cls.total_column_count(candidate_tables) + 2
        )

    @classmethod
    def should_prefer_richer_result(
        cls,
        candidate: TableExtractionResult,
        richer: TableExtractionResult,
    ) -> bool:
        candidate = cls.filtered_substantial_result(candidate)
        richer = cls.filtered_substantial_result(richer)
        candidate_tables = candidate.get("tables", [])
        richer_tables = richer.get("tables", [])
        if not richer_tables:
            return False
        if not candidate_tables:
            return True
        if len(richer_tables) != len(candidate_tables):
            return False

        candidate_area = cls.total_bbox_area(candidate.get("bboxes", []))
        richer_area = cls.total_bbox_area(richer.get("bboxes", []))
        if candidate_area <= 0.0 or richer_area <= 0.0:
            return False

        candidate_nonempty = cls.total_nonempty_cells(candidate_tables)
        richer_nonempty = cls.total_nonempty_cells(richer_tables)
        if richer_area >= candidate_area * 0.85 and candidate_area >= richer_area * 0.85:
            return richer_nonempty >= max(candidate_nonempty + 8, int(candidate_nonempty * 1.5))
        return richer_area >= candidate_area * 3.0 and richer_nonempty >= max(
            candidate_nonempty + 8, candidate_nonempty * 2
        )

    @classmethod
    def should_prefer_split_result(
        cls,
        candidate: TableExtractionResult,
        split: TableExtractionResult,
    ) -> bool:
        candidate = cls.filtered_substantial_result(candidate)
        split = cls.filtered_substantial_result(split)
        candidate_tables = candidate.get("tables", [])
        split_tables = split.get("tables", [])
        if len(candidate_tables) != 1 or len(split_tables) <= 1:
            return False

        candidate_area = cls.total_bbox_area(candidate.get("bboxes", []))
        split_area = cls.total_bbox_area(split.get("bboxes", []))
        if candidate_area <= 0.0 or split_area < candidate_area * 0.8:
            return False

        candidate_nonempty = cls.total_nonempty_cells(candidate_tables)
        split_nonempty = cls.total_nonempty_cells(split_tables)
        return split_nonempty >= candidate_nonempty + 8

    @classmethod
    def result_quality_key(cls, result: TableExtractionResult) -> tuple[int, float, int, int, int]:
        normalized = cls.normalize_result(result)
        score_tuples = [
            table_quality_score(table, text_length=table_text_length(table))
            for table in normalized.get("tables", [])
        ]
        if not score_tuples:
            return (0, 0.0, 0, 0, 0)
        return (
            sum(score[0] for score in score_tuples),
            sum(score[1] for score in score_tuples),
            sum(score[2] for score in score_tuples),
            sum(score[3] for score in score_tuples),
            sum(score[4] for score in score_tuples),
        )

    @classmethod
    def canonicalize_results(
        cls, results: Sequence[TableExtractionResult]
    ) -> TableExtractionResult:
        normalized_results = [cls.normalize_result(result) for result in results]
        candidates: list[
            tuple[
                tuple[int, float, int, int, int],
                tuple[float, float, float, float],
                TableRows,
                list[TableCellSpanRecord] | None,
            ]
        ] = []
        for result in normalized_results:
            tables = result.get("tables", [])
            result_bboxes = result.get("bboxes", [])
            spans = result.get("spans", [])
            for index, table in enumerate(tables):
                if index >= len(result_bboxes):
                    continue
                bbox = result_bboxes[index]
                if bbox is None:
                    continue
                score = table_quality_score(table, text_length=table_text_length(table))
                span = spans[index] if index < len(spans) else None
                candidates.append((score, bbox, table, span))

        if not candidates:
            return max(
                normalized_results,
                key=cls.result_quality_key,
                default=cls.table_result([]),
            )

        selected: list[
            tuple[
                tuple[int, float, int, int, int],
                tuple[float, float, float, float],
                TableRows,
                list[TableCellSpanRecord] | None,
            ]
        ] = []
        for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
            if any(bbox_overlap_ratio(candidate[1], chosen[1]) >= 0.85 for chosen in selected):
                continue
            selected.append(candidate)

        selected.sort(key=lambda item: (item[1][1], item[1][0]))
        tables = [item[2] for item in selected]
        bboxes: list[tuple[float, float, float, float] | None] = [item[1] for item in selected]
        spans = [item[3] for item in selected if item[3] is not None]
        return cls.table_result(
            tables,
            spans=spans if len(spans) == len(tables) else [],
            bboxes=bboxes,
        )

    @classmethod
    def table_options_from_kwargs(cls, kwargs: Mapping[str, Any]) -> TableExtractionOptions:
        return TableExtractionOptions(
            flavor=str(kwargs["flavor"]),
            canonicalize=bool(kwargs.get("canonicalize", True)),
            detect_header=bool(kwargs["detect_header"]),
            include_span_info=bool(kwargs["include_span_info"]),
            columns=tuple(float(value) for value in kwargs.get("columns") or ()),
            edge_tolerance=float(kwargs.get("edge_tolerance", 50.0)),
            row_tolerance=float(kwargs.get("row_tolerance", 2.0)),
            column_tolerance=float(kwargs.get("column_tolerance", 8.0)),
            split_text=bool(kwargs.get("split_text", False)),
            flag_size=bool(kwargs.get("flag_size", False)),
            strip_text=kwargs.get("strip_text"),
            replace_text=kwargs.get("replace_text"),
            copy_text=tuple(kwargs.get("copy_text") or ()),
            shift_text=tuple(kwargs.get("shift_text") or ()),
        )

    @classmethod
    def canonical_table_options(
        cls,
        base: TableExtractionOptions,
        *,
        flavor: str,
    ) -> TableExtractionOptions:
        shift_text = base.shift_text or (("l", "t") if flavor == "lattice" else ())
        return TableExtractionOptions(
            flavor=flavor,
            canonicalize=False,
            detect_header=base.detect_header,
            include_span_info=base.include_span_info,
            columns=base.columns,
            edge_tolerance=base.edge_tolerance,
            row_tolerance=base.row_tolerance,
            column_tolerance=base.column_tolerance,
            split_text=base.split_text,
            flag_size=base.flag_size,
            strip_text=base.strip_text,
            replace_text=base.replace_text,
            copy_text=base.copy_text,
            shift_text=shift_text,
        )

    @classmethod
    def build_table_extraction_context(
        cls,
        page: PageTableHost,
        options: TableExtractionOptions,
    ) -> TableExtractionContext:
        grid = None
        grids: list[TableGrid] = []
        page_rotate = page.rotation
        page_width = page.width
        page_height = page.height

        source_runs = rotate_page_runs(
            page.chars,
            rotate=page_rotate,
            page_width=page_width,
            page_height=page_height,
        )
        if options.flavor in ("lattice", "hybrid", "auto", "stream", "network"):
            grid_lines = page.grid_lines if page.grid_lines is not None else page.get_grid_lines()
            grid_lines = rotate_page_lines(
                grid_lines,
                rotate=page_rotate,
                page_width=page_width,
                page_height=page_height,
            )
            if grid_lines:
                grids = detect_grids_from_edges(
                    grid_lines,
                    snap_tolerance=options.column_tolerance,
                    join_tolerance=max(3.0, options.column_tolerance),
                )
                grid = grids[0] if grids else None

        visible_runs = [r for r in source_runs if r.visible]
        return TableExtractionContext(
            visible_runs=visible_runs,
            grids=grids,
            grid=grid,
            run_chars=page.display_text_span_chars() if options.split_text else None,
        )

    @classmethod
    def extract_table_strategy(
        cls,
        context: TableExtractionContext,
        options: TableExtractionOptions,
    ) -> TableExtractionResult:
        visible_runs = context.visible_runs
        grids = context.grids
        grid = context.grid
        normalized_columns = list(options.columns)
        if not visible_runs:
            return cls.table_result([])

        grid_options: GridExtractOptions = {
            "split_text": options.split_text,
            "flag_size": options.flag_size,
            "strip_text": options.strip_text,
            "replace_text": options.replace_text,
            "copy_text": options.copy_text,
            "shift_text": options.shift_text,
        }
        heuristic_options: HeuristicExtractOptions = {
            "row_tolerance": options.row_tolerance,
            "column_tolerance": options.column_tolerance,
            "columns": normalized_columns,
            "split_text": options.split_text,
            "flag_size": options.flag_size,
            "strip_text": options.strip_text,
            "replace_text": options.replace_text,
            "copy_text": options.copy_text,
            "shift_text": options.shift_text,
        }

        def extract_from_grid(
            table_grid: TableGrid,
            *,
            compact_network: bool = False,
            include_run_chars: bool = True,
        ) -> TableExtractionResult:
            raw_result = extract_grid(
                visible_runs,
                table_grid,
                options.include_span_info,
                compact_network=compact_network,
                run_chars=context.run_chars if include_run_chars else None,
                **grid_options,
            )
            if options.include_span_info:
                tables, spans = cast(tuple[TableSet, TableSpanRows], raw_result)
            else:
                tables = cast(TableSet, raw_result)
                spans = []
            bbox = PageTableCoreMixin.table_bbox_from_grid(table_grid)
            return PageTableCoreMixin.table_result(
                tables,
                spans=spans,
                bboxes=[bbox] * len(tables),
            )

        def extract_from_grids(
            table_grids: Sequence[TableGrid],
            *,
            compact_network: bool = False,
            include_run_chars: bool = True,
        ) -> TableExtractionResult:
            tables: TableSet = []
            spans: TableSpanRows = []
            bboxes: list[Rect | None] = []
            for table_grid in table_grids:
                grid_result = extract_from_grid(
                    table_grid,
                    compact_network=compact_network,
                    include_run_chars=include_run_chars,
                )
                tables.extend(grid_result["tables"])
                spans.extend(grid_result["spans"])
                bboxes.extend(grid_result["bboxes"])
            return PageTableCoreMixin.table_result(
                tables,
                spans=spans,
                bboxes=bboxes,
            )

        def extract_fallback_heuristic(
            *,
            include_run_chars: bool = False,
        ) -> TableExtractionResult:
            raw_result = extract_heuristic(
                visible_runs,
                options.detect_header,
                options.include_span_info,
                include_bboxes=True,
                run_chars=context.run_chars if include_run_chars else None,
                **heuristic_options,
            )
            if options.detect_header:
                header_result, heuristic_bboxes = cast(TableHeaderResultWithBBoxes, raw_result)
                header_rows, body_tables = header_result
                return PageTableCoreMixin.table_result(
                    body_tables,
                    bboxes=heuristic_bboxes,
                    header=header_rows,
                )
            elif options.include_span_info:
                span_result, heuristic_bboxes = cast(TableSpanResultWithBBoxes, raw_result)
                tables, spans = span_result
                return PageTableCoreMixin.table_result(
                    tables,
                    spans=spans,
                    bboxes=heuristic_bboxes,
                )
            else:
                tables, heuristic_bboxes = cast(TableSetWithBBoxes, raw_result)
                return PageTableCoreMixin.table_result(tables, bboxes=heuristic_bboxes)

        def extract_stream_result() -> TableExtractionResult:
            stream_grids = detect_stream_grids(
                visible_runs,
                row_tolerance=options.row_tolerance,
                column_tolerance=options.column_tolerance,
                columns=normalized_columns,
            )
            if stream_grids:
                return extract_from_grids(stream_grids)
            stream_grid = detect_stream_grid(
                visible_runs,
                row_tolerance=options.row_tolerance,
                column_tolerance=options.column_tolerance,
                columns=normalized_columns,
            )
            if stream_grid is not None and stream_grid.is_valid():
                return extract_from_grid(stream_grid)
            return extract_fallback_heuristic()

        def extract_network_result() -> TableExtractionResult:
            network_grids = detect_network_grids(
                visible_runs,
                edge_tolerance=options.edge_tolerance,
                row_tolerance=options.row_tolerance,
                column_tolerance=options.column_tolerance,
                columns=normalized_columns,
            )
            if not network_grids:
                return extract_fallback_heuristic(include_run_chars=True)
            tables: TableSet = []
            spans: TableSpanRows = []
            bboxes: list[Rect | None] = []
            for network_grid in network_grids:
                grid_result = extract_from_grid(
                    network_grid,
                    compact_network=True,
                )
                tables.extend(grid_result["tables"])
                spans.extend(grid_result["spans"])
                bboxes.extend(grid_result["bboxes"])
            return PageTableCoreMixin.table_result(tables, spans=spans, bboxes=bboxes)

        if options.flavor == "lattice":
            if grids:
                result = extract_from_grids(grids)
            elif grid is not None and grid.is_valid():
                result = extract_from_grid(grid)
            else:
                result = extract_fallback_heuristic()
        elif options.flavor == "network":
            ruled_result = extract_from_grids(grids) if grids else None
            result = extract_network_result()
            if ruled_result is not None and cls.should_prefer_ruled_result(
                result, ruled_result
            ):
                result = ruled_result
            else:
                stream_result = extract_stream_result()
                if cls.should_prefer_richer_result(result, stream_result):
                    result = stream_result
            result = cls.normalize_result(result)
        elif options.flavor == "stream":
            ruled_result = extract_from_grids(grids) if grids else None
            result = extract_stream_result()
            if ruled_result is not None and cls.should_prefer_ruled_result(
                result, ruled_result
            ):
                result = ruled_result
            else:
                network_result = extract_network_result()
                if network_result.get("tables") and cls.should_prefer_split_result(
                    result, network_result
                ):
                    result = network_result
            result = cls.normalize_result(result)
        elif options.flavor in ("auto", "hybrid"):
            if len(grids) > 1:
                result = extract_from_grids(grids, include_run_chars=False)
            elif grid is not None and grid.is_valid():
                lattice_result = extract_from_grid(grid, include_run_chars=False)
                network_grids = detect_network_grids(
                    visible_runs,
                    edge_tolerance=options.edge_tolerance,
                    row_tolerance=options.row_tolerance,
                    column_tolerance=options.column_tolerance,
                    columns=normalized_columns,
                )
                if (
                    options.flavor == "hybrid"
                    and network_grids
                    and len(network_grids[0].cols) > len(grid.cols)
                ):
                    result = extract_from_grid(
                        merge_grids(grid, network_grids[0]),
                        compact_network=True,
                    )
                else:
                    result = lattice_result
            else:
                network_grids = detect_network_grids(
                    visible_runs,
                    edge_tolerance=options.edge_tolerance,
                    row_tolerance=options.row_tolerance,
                    column_tolerance=options.column_tolerance,
                    columns=normalized_columns,
                )
                stream_grid = detect_stream_grid(
                    visible_runs,
                    row_tolerance=options.row_tolerance,
                    column_tolerance=options.column_tolerance,
                    columns=normalized_columns,
                )
                selected_grid = network_grids[0] if network_grids else stream_grid
                if selected_grid is not None and selected_grid.is_valid():
                    result = extract_from_grid(selected_grid)
                else:
                    result = extract_fallback_heuristic()
            result = cls.normalize_result(result)
            stream_result = cls.normalize_result(extract_stream_result())
            network_result = cls.normalize_result(extract_network_result())
            if not result.get("tables"):
                if stream_result.get("tables"):
                    result = stream_result
                elif network_result.get("tables"):
                    result = network_result
            else:
                if cls.should_prefer_richer_result(result, stream_result):
                    result = stream_result
                elif cls.should_prefer_richer_result(result, network_result):
                    result = network_result
            if cls.is_tiny_single_result(result):
                result = cls.table_result([])
        else:
            raise NotImplementedError(f"Unknown flavor specified: {options.flavor!r}")

        return result

    def extract_tables_core(self: PageTableHost, **kwargs: Any) -> TableExtractionResult:
        options = PageTableCoreMixin.table_options_from_kwargs(kwargs)
        context = PageTableCoreMixin.build_table_extraction_context(self, options)
        if not context.visible_runs:
            result = PageTableCoreMixin.table_result([])
            cache_key = table_cache_key(kwargs)
            self.tables[cache_key] = result
            return result

        if options.canonicalize and not options.detect_header:
            canonical_flavors = ("lattice", "stream", "network", "auto", "hybrid")
            canonical_results = [
                PageTableCoreMixin.extract_table_strategy(
                    context,
                    PageTableCoreMixin.canonical_table_options(options, flavor=current_flavor),
                )
                for current_flavor in canonical_flavors
            ]
            result = PageTableCoreMixin.canonicalize_results(canonical_results)
        else:
            result = PageTableCoreMixin.extract_table_strategy(context, options)

        if options.canonicalize:
            cache_key = table_cache_key(kwargs)
            self.tables[cache_key] = result
        return result


__all__ = (
    "GridExtractOptions",
    "HeuristicExtractOptions",
    "PageTableCoreMixin",
    "TableExtractionContext",
    "TableExtractionOptions",
)
