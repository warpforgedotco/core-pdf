# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Iterator
from statistics import median

from core_pdf.impl.engine.layout.models import TableGrid, TextRun
from core_pdf.impl.engine.extraction.common.ordering import cluster_runs_into_lines
from core_pdf.impl.engine.extraction.tables.grid import MIN_CELL_WIDTH
from core_pdf.impl.engine.extraction.tables.grid import connected_cell_groups
from core_pdf.impl.engine.extraction.tables.grid import intersections_to_cells
from core_pdf.impl.engine.extraction.tables.grid import join_table_edges
from core_pdf.impl.engine.extraction.tables.grid import snap
from core_pdf.impl.engine.extraction.tables.grid import snap_table_edges
from core_pdf.impl.engine.extraction.tables.grid import table_edge_intersections

DEFAULT_COLUMN_TOLERANCE: float = 8.0
DEFAULT_ROW_TOLERANCE: float = 2.0
DEFAULT_MIN_WORDS_VERTICAL: int = 3
DEFAULT_MIN_WORDS_HORIZONTAL: int = 2


def split_discontinuous_order_rows(
    rows: list[list[TextRun]],
    *,
    order_gap: int = 4,
) -> list[list[TextRun]]:
    split_rows: list[list[TextRun]] = []
    for row in rows:
        ordered = sorted(row, key=lambda run: run.order)
        current: list[TextRun] = []
        previous_order: int | None = None
        for run in ordered:
            if (
                current
                and previous_order is not None
                and run.order - previous_order > order_gap
            ):
                split_rows.append(sorted(current, key=lambda item: item.x0))
                current = []
            current.append(run)
            previous_order = run.order
        if current:
            split_rows.append(sorted(current, key=lambda item: item.x0))
    return split_rows


def uses_compressed_text_rows(runs: list[TextRun]) -> bool:
    text_runs = 0
    compressed_runs = 0
    for run in runs:
        if run.visible and run.has_text:
            text_runs += 1
            if run.height_value < 1.0:
                compressed_runs += 1
    return compressed_runs * 2 > text_runs


def detect_stream_grid(
    runs: list[TextRun],
    *,
    row_tolerance: float = DEFAULT_ROW_TOLERANCE,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    columns: list[float] | None = None,
) -> TableGrid | None:
    if not runs:
        return None

    compressed_rows = uses_compressed_text_rows(runs)
    min_row_height = max(0.5, row_tolerance) if compressed_rows else 8.0

    rows = cluster_runs_into_lines(
        runs,
        lookback=1,
        min_height=min_row_height,
        sort_by_horizontal_position=True,
    )
    if compressed_rows:
        rows = split_discontinuous_order_rows(rows)

    if len(rows) < 2:
        return None

    if columns:
        min_x = min(run.x0 for run in runs)
        max_x = max(run.x1 for run in runs)
        snapped_x_list = [min_x]
        snapped_x_list.extend(x for x in sorted(set(columns)) if min_x < x < max_x)
        snapped_x_list.append(max_x)
        if len(snapped_x_list) < 2:
            return None
    else:
        snapped_x_list = infer_stream_columns(
            rows, column_tolerance=column_tolerance, row_tolerance=row_tolerance
        )
        if len(snapped_x_list) < 2:
            virtual_grid = detect_text_edge_grid(
                [run for row in rows for run in row],
                row_tolerance=row_tolerance,
                column_tolerance=column_tolerance,
            )
            if virtual_grid is not None and virtual_grid.is_valid():
                return virtual_grid
            return None

    if not snapped_x_list:
        return None
    final_cols = sorted(snapped_x_list)

    if len(final_cols) < 2:
        return None

    filtered_cols = [final_cols[0]]
    for i in range(1, len(final_cols)):
        if final_cols[i] - filtered_cols[-1] >= MIN_CELL_WIDTH:
            filtered_cols.append(final_cols[i])

    if len(filtered_cols) < 2:
        return None

    final_rows = join_row_boundaries(rows)

    if len(final_rows) < 2:
        return None

    return TableGrid(cols=filtered_cols, rows=final_rows)


def split_rows_on_large_gaps(
    rows: list[list[TextRun]],
    *,
    row_tolerance: float = DEFAULT_ROW_TOLERANCE,
) -> list[list[list[TextRun]]]:
    if len(rows) < 2:
        return [rows] if rows else []
    row_extents = [
        (
            max(run.y1 for run in row),
            min(run.y0 for run in row),
        )
        for row in rows
        if row
    ]
    if len(row_extents) < 2:
        return [rows] if rows else []
    gaps = [
        row_extents[index][0] - row_extents[index + 1][1]
        for index in range(len(row_extents) - 1)
    ]
    positive_gaps = [gap for gap in gaps if gap > 0]
    if not positive_gaps:
        return [rows]
    typical_gap = float(median(positive_gaps))
    split_threshold = max(24.0, typical_gap * 1.75, row_tolerance * 6.0)
    groups: list[list[list[TextRun]]] = []
    current: list[list[TextRun]] = [rows[0]]
    for index, gap in enumerate(gaps):
        if gap > split_threshold:
            groups.append(current)
            current = []
        current.append(rows[index + 1])
    groups.append(current)
    return groups


def detect_stream_grids(
    runs: list[TextRun],
    *,
    row_tolerance: float = DEFAULT_ROW_TOLERANCE,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    columns: list[float] | None = None,
) -> list[TableGrid]:
    if not runs:
        return []
    compressed_rows = uses_compressed_text_rows(runs)
    min_row_height = max(0.5, row_tolerance) if compressed_rows else 8.0
    rows = cluster_runs_into_lines(
        runs,
        lookback=1,
        min_height=min_row_height,
        sort_by_horizontal_position=True,
    )
    if compressed_rows:
        rows = split_discontinuous_order_rows(rows)
    row_groups = split_rows_on_large_gaps(rows, row_tolerance=row_tolerance)
    grids: list[TableGrid] = []
    for row_group in row_groups:
        group_runs = [run for row in row_group for run in row]
        grid = detect_stream_grid(
            group_runs,
            row_tolerance=row_tolerance,
            column_tolerance=column_tolerance,
            columns=columns,
        )
        if grid is not None and grid.is_valid():
            grids.append(grid)
    return grids


def detect_text_edge_grid(
    runs: list[TextRun],
    *,
    row_tolerance: float = DEFAULT_ROW_TOLERANCE,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    min_words_vertical: int = DEFAULT_MIN_WORDS_VERTICAL,
    min_words_horizontal: int = DEFAULT_MIN_WORDS_HORIZONTAL,
) -> TableGrid | None:
    visible_runs = [run for run in runs if run.visible and run.has_text]
    if len(visible_runs) < min_words_vertical * min_words_horizontal:
        return None
    rows = cluster_runs_into_lines(
        visible_runs,
        lookback=1,
        min_height=max(0.5, row_tolerance),
        sort_by_horizontal_position=True,
    )
    rows = [
        sorted(row, key=lambda run: run.x0)
        for row in rows
        if len(row) >= min_words_horizontal
    ]
    if len(rows) < min_words_vertical:
        return None
    counts = [len(row) for row in rows]
    modal_count = max(set(counts), key=counts.count)
    if modal_count < 2 or counts.count(modal_count) < min_words_vertical:
        return None
    rows = [row for row in rows if len(row) == modal_count]
    if len(rows) < min_words_vertical:
        return None

    row_edges = words_to_horizontal_edges(rows)
    column_edges = words_to_vertical_edges(
        rows,
        column_tolerance=column_tolerance,
        min_words_vertical=min_words_vertical,
    )
    if len(row_edges) < 2 or len(column_edges) < 2:
        return None
    edges = row_edges + column_edges
    edges = snap_table_edges(edges, max(1.0, min(row_tolerance, column_tolerance)))
    edges = join_table_edges(edges, max(2.0, column_tolerance))
    intersections = table_edge_intersections(
        edges,
        tolerance=max(2.0, min(row_tolerance, column_tolerance)),
    )
    cells = intersections_to_cells(intersections)
    if len(cells) < 4:
        return None
    groups = connected_cell_groups(cells)
    if not groups:
        return None
    best = groups[0]
    cols = sorted({coord for cell in best for coord in (cell[0], cell[2])})
    rows_out = sorted(
        {coord for cell in best for coord in (cell[1], cell[3])},
        reverse=True,
    )
    if len(cols) < modal_count + 1 or len(rows_out) < 3:
        return None
    return TableGrid(cols=cols, rows=rows_out)


def words_to_horizontal_edges(
    rows: list[list[TextRun]],
) -> list[dict[str, float | str]]:
    if not rows:
        return []
    min_x = min(run.x0 for row in rows for run in row)
    max_x = max(run.x1 for row in rows for run in row)
    edges: list[dict[str, float | str]] = []
    for row in rows:
        top = max(run.y1 for run in row)
        bottom = min(run.y0 for run in row)
        edges.append(
            {
                "orientation": "h",
                "coord": top,
                "start": min_x,
                "end": max_x,
            }
        )
        edges.append(
            {
                "orientation": "h",
                "coord": bottom,
                "start": min_x,
                "end": max_x,
            }
        )
    return edges


def words_to_vertical_edges(
    rows: list[list[TextRun]],
    *,
    column_tolerance: float,
    min_words_vertical: int,
) -> list[dict[str, float | str]]:
    positions: list[tuple[float, TextRun]] = []
    for row in rows:
        for run in row:
            positions.append((run.x0, run))
            positions.append((run.x1, run))
            positions.append(((run.x0 + run.x1) * 0.5, run))
    positions.sort(key=lambda item: item[0])
    clusters: list[list[tuple[float, TextRun]]] = []
    for item in positions:
        if clusters and abs(
            item[0] - sum(value for value, ignored in clusters[-1]) / len(clusters[-1])
        ) <= max(0.5, column_tolerance):
            clusters[-1].append(item)
        else:
            clusters.append([item])

    candidates: list[tuple[float, float, float, float]] = []
    for cluster in clusters:
        row_keys = {round((run.y0 + run.y1) * 0.5, 1) for ignored, run in cluster}
        if len(row_keys) < min_words_vertical:
            continue
        cluster_runs = [run for ignored, run in cluster]
        coord = min(run.x0 for run in cluster_runs)
        candidates.append(
            (
                coord,
                min(run.y0 for run in cluster_runs),
                max(run.y1 for run in cluster_runs),
                max(run.x1 for run in cluster_runs),
            )
        )

    condensed: list[tuple[float, float, float, float]] = []
    for candidate in sorted(candidates, key=lambda item: item[0]):
        coord, y0, y1, right = candidate
        if any(
            max(y0, existing[1]) <= min(y1, existing[2])
            and abs(coord - existing[0]) <= column_tolerance
            for existing in condensed
        ):
            continue
        condensed.append(candidate)

    if not condensed:
        return []
    min_y = min(run.y0 for row in rows for run in row)
    max_y = max(run.y1 for row in rows for run in row)
    coords = [coord for coord, ignored, ignored, ignored in condensed]
    coords.append(max(run.x1 for row in rows for run in row))
    coords = sorted(coords)
    filtered_coords = [coords[0]]
    for coord in coords[1:]:
        if coord - filtered_coords[-1] >= MIN_CELL_WIDTH:
            filtered_coords.append(coord)

    edges: list[dict[str, float | str]] = []
    for coord in filtered_coords:
        edges.append(
            {
                "orientation": "v",
                "coord": coord,
                "start": min_y,
                "end": max_y,
            }
        )
    return edges


def join_row_boundaries(rows: list[list[TextRun]]) -> list[float]:
    row_extents = [
        (
            max(run.coords[TextRun.Y1] for run in row),
            min(run.coords[TextRun.Y0] for run in row),
        )
        for row in rows
        if row
    ]
    row_extents.sort(key=lambda item: (item[0] + item[1]) * 0.5, reverse=True)
    if not row_extents:
        return []
    row_mids = [(top + bottom) * 0.5 for top, bottom in row_extents]
    boundaries = [max(row_extents[0][0], row_mids[0])]
    for index in range(len(row_extents) - 1):
        boundaries.append((row_mids[index] + row_mids[index + 1]) * 0.5)
    boundaries.append(min(row_extents[-1][1], row_mids[-1]))
    return boundaries


def infer_stream_columns(
    rows: list[list[TextRun]],
    *,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    row_tolerance: float = DEFAULT_ROW_TOLERANCE,
) -> list[float]:
    populated_rows = [[run for run in row if run.has_text] for row in rows]
    populated_rows = [row for row in populated_rows if row]
    if not populated_rows:
        return []

    counts = [len(row) for row in populated_rows]
    non_single_counts = [count for count in counts if count != 1]
    if non_single_counts:
        ncols = max(set(non_single_counts), key=non_single_counts.count)
    else:
        ncols = max(set(counts), key=counts.count)
    if ncols < 2:
        return []

    modal_rows = [
        sorted(row, key=lambda run: run.x0)
        for row in populated_rows
        if len(row) == ncols
    ]
    if len(modal_rows) >= 2:
        left_edges = [
            float(median(row[col_idx].x0 for row in modal_rows))
            for col_idx in range(ncols)
        ]
        right_edges = [
            float(median(row[col_idx].x1 for row in modal_rows))
            for col_idx in range(ncols)
        ]
        boundaries = [
            (right_edges[col_idx - 1] + left_edges[col_idx]) * 0.5
            for col_idx in range(1, ncols)
        ]
        boundaries.insert(0, left_edges[0])
        boundaries.append(right_edges[-1])
        return sorted(boundaries)

    extents: list[tuple[float, float]] = []
    for row in populated_rows:
        if len(row) == ncols:
            extents.extend((run.x0, run.x1) for run in row)

    extents = merge_column_extents(sorted(extents), column_tolerance=column_tolerance)

    if not extents:
        x_positions = [run.x0 for row in populated_rows for run in row]
        return snap(x_positions, max(0.5, column_tolerance))

    for ignored in range(2):
        inner: list[tuple[float, float]] = []
        for row in populated_rows:
            for run in row:
                inside_gap = any(
                    extents[i][1] < run.x0 and run.x1 < extents[i + 1][0]
                    for i in range(len(extents) - 1)
                )
                outside = run.x1 < extents[0][0] or run.x0 > extents[-1][1]
                row_aligned = any(
                    abs(run.mid_y - other.mid_y) <= row_tolerance
                    for other in row
                    if other is not run
                )
                if (inside_gap or outside) and row_aligned:
                    inner.append((run.x0, run.x1))
        if not inner:
            break
        merged = merge_column_extents(
            sorted([*extents, *inner]), column_tolerance=column_tolerance
        )
        if merged == extents:
            break
        extents = merged

    if len(extents) < 2:
        return []

    boundaries = [
        (extents[i][0] + extents[i - 1][1]) * 0.5 for i in range(1, len(extents))
    ]
    boundaries.insert(0, min(run.x0 for row in populated_rows for run in row))
    boundaries.append(max(run.x1 for row in populated_rows for run in row))
    return sorted(boundaries)


def merge_column_extents(
    extents: list[tuple[float, float]], *, column_tolerance: float = 0.0
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for left, right in extents:
        if right - left < MIN_CELL_WIDTH * 0.25:
            continue
        if not merged:
            merged.append((left, right))
            continue
        prev_left, prev_right = merged[-1]
        if left <= prev_right or abs(left - prev_right) <= column_tolerance:
            merged[-1] = (min(prev_left, left), max(prev_right, right))
        else:
            merged.append((left, right))
    return merged


def detect_network_grids(
    runs: list[TextRun],
    *,
    row_tolerance: float = DEFAULT_ROW_TOLERANCE,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    edge_tolerance: float = 50.0,
    columns: list[float] | None = None,
) -> list[TableGrid]:
    rows = list(iter_rows(runs))
    if len(rows) < 2:
        return []

    candidate_edges: list[tuple[float, float, float]] = []
    for align in ("left", "right", "middle"):
        values: list[tuple[float, list[TextRun]]] = []
        for row in rows:
            row_runs = [run for run in row if run.text.strip()]
            if not row_runs:
                continue
            x0 = min(run.x0 for run in row_runs)
            x1 = max(run.x1 for run in row_runs)
            coord = (
                x0 if align == "left" else x1 if align == "right" else (x0 + x1) * 0.5
            )
            values.append((coord, row_runs))
        values.sort(key=lambda item: item[0])
        clusters: list[list[tuple[float, list[TextRun]]]] = []
        for value in values:
            if clusters and abs(
                value[0] - (sum(v[0] for v in clusters[-1]) / len(clusters[-1]))
            ) <= max(0.5, column_tolerance):
                clusters[-1].append(value)
            else:
                clusters.append([value])
        for cluster in clusters:
            if len(cluster) < 4:
                continue
            cluster_runs = [run for ignored, row_runs in cluster for run in row_runs]
            candidate_edges.append(
                (
                    min(run.y0 for run in cluster_runs),
                    max(run.y1 for run in cluster_runs),
                    sum(value for value, ignored in cluster) / len(cluster),
                )
            )

    if not candidate_edges:
        return []

    candidate_edges.sort(key=lambda item: (-item[1], item[2]))
    areas: list[tuple[float, float, float, float]] = []
    for bottom, top, x in candidate_edges:
        found = -1
        for idx, (area_x0, area_y0, area_x1, area_y1) in enumerate(areas):
            if top >= area_y0 - edge_tolerance and bottom <= area_y1 + edge_tolerance:
                found = idx
                break
        if found < 0:
            areas.append((x, bottom, x, top))
        else:
            area_x0, area_y0, area_x1, area_y1 = areas[found]
            areas[found] = (
                min(area_x0, x),
                min(area_y0, bottom),
                max(area_x1, x),
                max(area_y1, top),
            )

    grids: list[TableGrid] = []
    for x0, y0, x1, y1 in areas:
        area_runs = [
            run
            for run in runs
            if run.visible
            and run.text.strip()
            and run.x1 >= x0 - column_tolerance
            and run.x0 <= x1 + column_tolerance
            and run.y1 >= y0
            and run.y0 <= y1
        ]
        grids.extend(
            detect_stream_grids(
                area_runs,
                row_tolerance=row_tolerance,
                column_tolerance=column_tolerance,
                columns=columns,
            )
        )
    return grids


def iter_rows(visible_runs: list[TextRun]) -> Iterator[list[TextRun]]:
    clusters = cluster_runs_into_lines(
        visible_runs, lookback=1, sort_by_horizontal_position=True
    )
    for row in clusters:
        yield row


__all__ = (
    "DEFAULT_COLUMN_TOLERANCE",
    "DEFAULT_MIN_WORDS_HORIZONTAL",
    "DEFAULT_MIN_WORDS_VERTICAL",
    "DEFAULT_ROW_TOLERANCE",
    "detect_network_grids",
    "detect_stream_grid",
    "detect_stream_grids",
    "detect_text_edge_grid",
    "infer_stream_columns",
    "iter_rows",
    "join_row_boundaries",
    "merge_column_extents",
    "split_discontinuous_order_rows",
    "uses_compressed_text_rows",
    "words_to_horizontal_edges",
    "words_to_vertical_edges",
)
