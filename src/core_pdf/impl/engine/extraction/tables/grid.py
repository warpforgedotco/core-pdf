# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import array
import itertools

from core_pdf.impl.engine.layout.models import TableGrid
from core_pdf.impl.engine.spec.s_07_content.capture import CapturedLine

GRID_SNAP_TOLERANCE: float = 2.0
MIN_CELL_WIDTH: float = 4.0
MIN_CELL_HEIGHT: float = 4.0
DEFAULT_EDGE_JOIN_TOLERANCE: float = 3.0

TableEdge = dict[str, float | str]
IntersectionMap = dict[
    tuple[float, float],
    dict[str, list[tuple[str, float, float, float]]],
]


def snap(values: list[float] | array.array[float], tolerance: float) -> list[float]:
    if not values:
        return []

    sorted_vals = sorted(values)
    clusters: list[list[float]] = [[sorted_vals[0]]]
    last_cluster = clusters[0]

    for value in sorted_vals[1:]:
        if value - last_cluster[-1] <= tolerance:
            last_cluster.append(value)
        else:
            last_cluster = [value]
            clusters.append(last_cluster)

    return [sum(cluster) * (1.0 / len(cluster)) for cluster in clusters]


def detect_grid(
    lines: list[CapturedLine], *, line_tolerance: float = GRID_SNAP_TOLERANCE
) -> TableGrid | None:
    if not lines:
        return None
    reconstructed = detect_grid_from_edges(
        lines,
        snap_tolerance=line_tolerance,
        join_tolerance=max(DEFAULT_EDGE_JOIN_TOLERANCE, line_tolerance),
    )
    if reconstructed is not None:
        return reconstructed
    horizontals = array.array("f")
    verticals = array.array("f")
    for line in lines:
        dx = abs(line.x1 - line.x0)
        dy = abs(line.y1 - line.y0)
        if dx > dy:
            horizontals.append((line.y0 + line.y1) * 0.5)
        elif dy > dx:
            verticals.append((line.x0 + line.x1) * 0.5)
    h_clusters = snap(horizontals, line_tolerance)
    v_clusters = snap(verticals, line_tolerance)
    if len(h_clusters) < 2 or len(v_clusters) < 2:
        return None
    rows = sorted(set(h_clusters), reverse=True)
    cols = sorted(set(v_clusters))
    if len(rows) < 2 or len(cols) < 2:
        return None
    row_starts = []
    for i in range(len(rows) - 1):
        if rows[i] - rows[i + 1] >= MIN_CELL_HEIGHT:
            row_starts.append(rows[i])
    row_starts.append(rows[-1])
    col_starts = []
    for i in range(len(cols) - 1):
        if cols[i + 1] - cols[i] >= MIN_CELL_WIDTH:
            col_starts.append(cols[i])
    col_starts.append(cols[-1])
    if len(row_starts) < 2 or len(col_starts) < 2:
        return None
    return TableGrid(cols=col_starts, rows=row_starts)


def line_edges(lines: list[CapturedLine]) -> list[TableEdge]:
    edges: list[TableEdge] = []
    for line in lines:
        dx = abs(line.x1 - line.x0)
        dy = abs(line.y1 - line.y0)
        if dx > dy:
            x0, x1 = sorted((float(line.x0), float(line.x1)))
            y = (float(line.y0) + float(line.y1)) * 0.5
            if x1 - x0 >= MIN_CELL_WIDTH:
                edges.append(
                    {
                        "orientation": "h",
                        "coord": y,
                        "start": x0,
                        "end": x1,
                    }
                )
        elif dy > dx:
            y0, y1 = sorted((float(line.y0), float(line.y1)))
            x = (float(line.x0) + float(line.x1)) * 0.5
            if y1 - y0 >= MIN_CELL_HEIGHT:
                edges.append(
                    {
                        "orientation": "v",
                        "coord": x,
                        "start": y0,
                        "end": y1,
                    }
                )
    return edges


def snap_table_edges(
    edges: list[TableEdge],
    tolerance: float,
) -> list[TableEdge]:
    snapped: list[TableEdge] = []
    for orientation in ("h", "v"):
        group = [edge for edge in edges if edge["orientation"] == orientation]
        group.sort(key=lambda edge: float(edge["coord"]))
        clusters: list[list[TableEdge]] = []
        for edge in group:
            if (
                clusters
                and abs(
                    float(edge["coord"])
                    - sum(float(item["coord"]) for item in clusters[-1]) / len(clusters[-1])
                )
                <= tolerance
            ):
                clusters[-1].append(edge)
            else:
                clusters.append([edge])
        for cluster in clusters:
            coord = sum(float(edge["coord"]) for edge in cluster) / len(cluster)
            for edge in cluster:
                replacement = dict(edge)
                replacement["coord"] = coord
                snapped.append(replacement)
    return snapped


def join_table_edges(
    edges: list[TableEdge],
    tolerance: float,
) -> list[TableEdge]:
    joined: list[TableEdge] = []
    edges = sorted(
        edges,
        key=lambda edge: (
            str(edge["orientation"]),
            round(float(edge["coord"]), 4),
            float(edge["start"]),
        ),
    )
    for ignored, group_iter in itertools.groupby(
        edges,
        key=lambda edge: (str(edge["orientation"]), round(float(edge["coord"]), 4)),
    ):
        group = sorted(group_iter, key=lambda edge: float(edge["start"]))
        current = dict(group[0])
        for edge in group[1:]:
            if float(edge["start"]) <= float(current["end"]) + tolerance:
                current["end"] = max(float(current["end"]), float(edge["end"]))
            else:
                joined.append(current)
                current = dict(edge)
        joined.append(current)
    return joined


def merge_table_edges(
    lines: list[CapturedLine],
    *,
    snap_tolerance: float,
    join_tolerance: float,
) -> list[TableEdge]:
    edges = line_edges(lines)
    if not edges:
        return []
    edges = snap_table_edges(edges, snap_tolerance)
    edges = join_table_edges(edges, join_tolerance)
    return [
        edge
        for edge in edges
        if float(edge["end"]) - float(edge["start"])
        >= (MIN_CELL_WIDTH if edge["orientation"] == "h" else MIN_CELL_HEIGHT)
    ]


def table_edge_intersections(
    edges: list[TableEdge],
    *,
    tolerance: float,
) -> IntersectionMap:
    intersections: IntersectionMap = {}
    verticals = [edge for edge in edges if edge["orientation"] == "v"]
    horizontals = [edge for edge in edges if edge["orientation"] == "h"]
    for v_edge in verticals:
        x = float(v_edge["coord"])
        y0 = float(v_edge["start"])
        y1 = float(v_edge["end"])
        v_key = ("v", x, y0, y1)
        for h_edge in horizontals:
            y = float(h_edge["coord"])
            x0 = float(h_edge["start"])
            x1 = float(h_edge["end"])
            if (
                y0 <= y + tolerance
                and y1 >= y - tolerance
                and x0 <= x + tolerance
                and x1 >= x - tolerance
            ):
                h_key = ("h", y, x0, x1)
                point = (x, y)
                bucket = intersections.setdefault(point, {"v": [], "h": []})
                bucket["v"].append(v_key)
                bucket["h"].append(h_key)
    return intersections


def intersections_to_cells(
    intersections: IntersectionMap,
) -> list[tuple[float, float, float, float]]:
    def connected(p1: tuple[float, float], p2: tuple[float, float]) -> bool:
        if p1[0] == p2[0]:
            return bool(set(intersections[p1]["v"]) & set(intersections[p2]["v"]))
        if p1[1] == p2[1]:
            return bool(set(intersections[p1]["h"]) & set(intersections[p2]["h"]))
        return False

    points = sorted(intersections, key=lambda point: (point[0], -point[1]))
    cells: list[tuple[float, float, float, float]] = []
    for point in points:
        x0, top = point
        rights = [candidate for candidate in points if candidate[1] == top and candidate[0] > x0]
        belows = [candidate for candidate in points if candidate[0] == x0 and candidate[1] < top]
        for below in belows:
            if not connected(point, below):
                continue
            for right in rights:
                if not connected(point, right):
                    continue
                bottom_right = (right[0], below[1])
                if (
                    bottom_right in intersections
                    and connected(bottom_right, right)
                    and connected(bottom_right, below)
                ):
                    x1 = bottom_right[0]
                    bottom = bottom_right[1]
                    if x1 - x0 >= MIN_CELL_WIDTH and top - bottom >= MIN_CELL_HEIGHT:
                        cells.append((x0, top, x1, bottom))
                    break
            else:
                continue
            break
    return cells


def connected_cell_groups(
    cells: list[tuple[float, float, float, float]],
) -> list[list[tuple[float, float, float, float]]]:
    remaining = list(cells)
    groups: list[list[tuple[float, float, float, float]]] = []
    while remaining:
        group = [remaining.pop(0)]
        corners = set(cell_corners(group[0]))
        changed = True
        while changed:
            changed = False
            for cell in list(remaining):
                current_corners = set(cell_corners(cell))
                if corners & current_corners:
                    group.append(cell)
                    corners |= current_corners
                    remaining.remove(cell)
                    changed = True
        groups.append(group)
    groups.sort(key=lambda group: (-len(group), min(-cell[1] for cell in group)))
    return groups


def group_bbox(
    cells: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    return (
        min(cell[0] for cell in cells),
        min(cell[3] for cell in cells),
        max(cell[2] for cell in cells),
        max(cell[1] for cell in cells),
    )


def filtered_cell_groups(
    groups: list[list[tuple[float, float, float, float]]],
) -> list[list[tuple[float, float, float, float]]]:
    filtered: list[list[tuple[float, float, float, float]]] = []
    for index, group in enumerate(groups):
        bbox = group_bbox(group)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        keep = True
        if len(group) <= 8 and height <= 64.0:
            for other_index, other in enumerate(groups):
                if other_index == index or len(other) <= len(group):
                    continue
                other_bbox = group_bbox(other)
                other_width = other_bbox[2] - other_bbox[0]
                horizontal_overlap = max(
                    0.0, min(bbox[2], other_bbox[2]) - max(bbox[0], other_bbox[0])
                )
                if width <= 0.0 or other_width <= 0.0:
                    continue
                overlap_ratio = horizontal_overlap / min(width, other_width)
                vertical_gap = bbox[1] - other_bbox[3]
                if overlap_ratio >= 0.8 and 0.0 <= vertical_gap <= 80.0:
                    keep = False
                    break
        if keep:
            filtered.append(group)
    return filtered


def cell_corners(
    cell: tuple[float, float, float, float],
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    x0, top, x1, bottom = cell
    return ((x0, top), (x0, bottom), (x1, top), (x1, bottom))


def detect_grid_from_edges(
    lines: list[CapturedLine],
    *,
    snap_tolerance: float,
    join_tolerance: float,
) -> TableGrid | None:
    grids = detect_grids_from_edges(
        lines,
        snap_tolerance=snap_tolerance,
        join_tolerance=join_tolerance,
    )
    return grids[0] if grids else None


def table_grid_from_cells(
    cells: list[tuple[float, float, float, float]],
) -> TableGrid | None:
    if not cells:
        return None
    cols = sorted({coord for cell in cells for coord in (cell[0], cell[2])})
    rows = sorted({coord for cell in cells for coord in (cell[1], cell[3])}, reverse=True)
    filtered_cols = [cols[0]]
    for col in cols[1:]:
        if col - filtered_cols[-1] >= MIN_CELL_WIDTH:
            filtered_cols.append(col)
    filtered_rows = [rows[0]]
    for row in rows[1:]:
        if filtered_rows[-1] - row >= MIN_CELL_HEIGHT:
            filtered_rows.append(row)
    if len(filtered_cols) < 2 or len(filtered_rows) < 2:
        return None
    return TableGrid(cols=filtered_cols, rows=filtered_rows)


def detect_grids_from_edges(
    lines: list[CapturedLine],
    *,
    snap_tolerance: float,
    join_tolerance: float,
) -> list[TableGrid]:
    edges = merge_table_edges(
        lines,
        snap_tolerance=snap_tolerance,
        join_tolerance=join_tolerance,
    )
    if len(edges) < 4:
        return []
    intersections = table_edge_intersections(
        edges,
        tolerance=max(1.0, snap_tolerance),
    )
    cells = intersections_to_cells(intersections)
    if not cells:
        return []
    groups = connected_cell_groups(cells)
    if not groups:
        return []
    groups = filtered_cell_groups(groups)
    grids: list[TableGrid] = []
    for group in groups:
        grid = table_grid_from_cells(group)
        if grid is not None:
            grids.append(grid)
    return grids


def merge_grids(primary: TableGrid, secondary: TableGrid) -> TableGrid:
    cols = sorted(set(primary.cols + secondary.cols))
    rows = sorted(set(primary.rows + secondary.rows), reverse=True)
    filtered_cols = [cols[0]]
    for col in cols[1:]:
        if col - filtered_cols[-1] >= MIN_CELL_WIDTH:
            filtered_cols.append(col)
    filtered_rows = [rows[0]]
    for row in rows[1:]:
        if filtered_rows[-1] - row >= MIN_CELL_HEIGHT:
            filtered_rows.append(row)
    return TableGrid(cols=filtered_cols, rows=filtered_rows)


__all__ = (
    "DEFAULT_EDGE_JOIN_TOLERANCE",
    "GRID_SNAP_TOLERANCE",
    "IntersectionMap",
    "MIN_CELL_HEIGHT",
    "MIN_CELL_WIDTH",
    "TableEdge",
    "cell_corners",
    "connected_cell_groups",
    "detect_grid",
    "detect_grids_from_edges",
    "detect_grid_from_edges",
    "filtered_cell_groups",
    "group_bbox",
    "intersections_to_cells",
    "join_table_edges",
    "line_edges",
    "merge_grids",
    "merge_table_edges",
    "snap",
    "snap_table_edges",
    "table_grid_from_cells",
    "table_edge_intersections",
)
