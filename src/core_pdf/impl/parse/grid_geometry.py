# SPDX-License-Identifier: AGPL-3.0-only
"""Ruled-grid geometry shared by table extraction and OCR.

Both stages look for ruling lines, but for different reasons: tables.py builds cell
topology from a page's vector rulings, while the OCR stage detects rulings in a
raster to decide whether a scanned region is a regular table worth recognizing cell
by cell. The segment classification and connected-component work underneath is the
same, so it lives here rather than being reached for sideways.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy

from core_pdf.impl.layout.spatial import SpatialIndex
from core_pdf.impl.parse.model import CapturedPage
from core_pdf.impl.spec.s_07_content.page_program import line_coordinate_columns

AXIS_TOLERANCE = 1.5


TABLE_REGION_GAP = 22.0  # loosened to allow adjacent table regions with modest gaps to merge


class internal_DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def internal_axis_segments(
    capture: CapturedPage,
) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]]:
    page_width = float(capture.page.width)
    page_height = float(capture.page.height)
    lines = capture.grid_lines
    if not lines:
        empty = numpy.empty((0, 3), dtype=numpy.float32)
        return empty, empty

    # Read the four coordinate columns straight off the capture, then classify
    # and normalize all segments with array operations.  ``LineTable`` already
    # stores them, so no per-line Python object is built here.
    x0, y0, x1, y1 = line_coordinate_columns(lines)
    horizontal_mask = (numpy.abs(y1 - y0) <= AXIS_TOLERANCE) & (
        numpy.abs(x1 - x0) >= page_width * 0.02
    )
    vertical_mask = (
        ~horizontal_mask
        & (numpy.abs(x1 - x0) <= AXIS_TOLERANCE)
        & (numpy.abs(y1 - y0) >= page_height * 0.015)
    )
    horizontal = numpy.column_stack(
        (numpy.minimum(x0, x1), numpy.maximum(x0, x1), (y0 + y1) * 0.5)
    )[horizontal_mask].astype(numpy.float32, copy=False)
    vertical = numpy.column_stack(((x0 + x1) * 0.5, numpy.minimum(y0, y1), numpy.maximum(y0, y1)))[
        vertical_mask
    ].astype(numpy.float32, copy=False)
    return horizontal.reshape((-1, 3)), vertical.reshape((-1, 3))


def internal_grid_components(
    horizontal: numpy.ndarray[Any, Any],
    vertical: numpy.ndarray[Any, Any],
) -> tuple[tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]], ...]:
    if len(horizontal) < 2 or len(vertical) < 2:
        return ()
    v_boxes = numpy.column_stack(
        (
            vertical[:, 0] - AXIS_TOLERANCE,
            vertical[:, 1] - AXIS_TOLERANCE,
            vertical[:, 0] + AXIS_TOLERANCE,
            vertical[:, 2] + AXIS_TOLERANCE,
        )
    )
    vertical_index = SpatialIndex((i, v_boxes[i]) for i in range(len(vertical)))
    pairs: list[tuple[int, int]] = []
    for h_index, segment in enumerate(horizontal):
        h_box = (
            float(segment[0]) - AXIS_TOLERANCE,
            float(segment[2]) - AXIS_TOLERANCE,
            float(segment[1]) + AXIS_TOLERANCE,
            float(segment[2]) + AXIS_TOLERANCE,
        )
        for v_index in vertical_index.intersecting(h_box):
            pairs.append((h_index, int(v_index)))
    if not pairs:
        return ()
    disjoint = internal_DisjointSet(len(horizontal) + len(vertical))
    for h_index, v_index in pairs:
        disjoint.union(h_index, len(horizontal) + v_index)
    grouped_h: dict[int, list[int]] = defaultdict(list)
    grouped_v: dict[int, list[int]] = defaultdict(list)
    for index in sorted({h_index for h_index, internal_v_index in pairs}):
        grouped_h[disjoint.find(index)].append(index)
    for index in sorted({v_index for internal_h_index, v_index in pairs}):
        grouped_v[disjoint.find(len(horizontal) + index)].append(index)
    return tuple(
        (horizontal[grouped_h[root]], vertical[grouped_v[root]])
        for root in grouped_h.keys() & grouped_v.keys()
    )


def internal_split_grid_component(
    horizontal: numpy.ndarray[Any, Any],
    vertical: numpy.ndarray[Any, Any],
) -> tuple[tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]], ...]:
    """Split a connected ruled component into vertically separated table regions."""
    positions = numpy.unique(horizontal[:, 2])
    if len(positions) < 3:
        return ((horizontal, vertical),)
    group_breaks = numpy.empty(len(positions), dtype=numpy.bool_)
    group_breaks[0] = True
    group_breaks[1:] = numpy.diff(positions) > TABLE_REGION_GAP
    position_groups = numpy.cumsum(group_breaks) - 1
    horizontal_groups = position_groups[numpy.searchsorted(positions, horizontal[:, 2])]
    regions = []
    for group_index in range(int(position_groups[-1]) + 1):
        group_positions = positions[position_groups == group_index]
        y0, y1 = group_positions[0], group_positions[-1]
        region_horizontal = horizontal[horizontal_groups == group_index]
        region_vertical = vertical[
            (vertical[:, 1] <= y1 + AXIS_TOLERANCE) & (vertical[:, 2] >= y0 - AXIS_TOLERANCE)
        ]
        if len(region_horizontal) >= 2 and len(region_vertical) >= 2:
            regions.append((region_horizontal, region_vertical))
    return tuple(regions) or ((horizontal, vertical),)
