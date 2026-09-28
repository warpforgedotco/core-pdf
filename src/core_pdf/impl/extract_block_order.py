# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from copy import replace
from heapq import heappop, heappush
from typing import Protocol

import numpy

from core_pdf.impl.extract_contracts import ParsedBlock, ParsedLine
from core_pdf.impl.extract_layout_rules import LAYOUT_RULES
from core_pdf.impl.extract_xy_cut import row_order_indexes
from core_pdf.impl.geometry import array_bbox, horizontal_overlap_ratio, interval_overlap
from core_pdf.impl.spatial import cluster_1d, interval_overlap_pairs


class BlockReorderPass(Protocol):
    def __call__(self, blocks: list[ParsedBlock], /) -> list[ParsedBlock]: ...


def line_bbox(line: ParsedLine) -> tuple[float, float, float, float]:
    bbox = line.line.bbox
    assert bbox is not None
    return bbox


def block_bbox(lines: tuple[ParsedLine, ...]) -> tuple[float, float, float, float]:
    return array_bbox(numpy.asarray(tuple(line_bbox(line) for line in lines), dtype=numpy.float32))


def sparse_block_candidate_pairs(
    blocks: list[ParsedBlock], full_width: list[bool]
) -> list[tuple[int, int]]:
    boxes = numpy.asarray(tuple(block.bbox for block in blocks), dtype=numpy.float64)
    pairs = interval_overlap_pairs(boxes[:, 0], boxes[:, 2])
    pairs.update(interval_overlap_pairs(boxes[:, 1], boxes[:, 3]))
    full_width_indexes = [index for index, value in enumerate(full_width) if value]
    for index in full_width_indexes:
        pairs.update(
            (min(index, other), max(index, other)) for other in range(len(blocks)) if other != index
        )
    return sorted(pairs)


def full_width_blocks(blocks: list[ParsedBlock]) -> list[bool]:
    page_x0 = min(block.bbox[0] for block in blocks)
    page_x1 = max(block.bbox[2] for block in blocks)
    page_width = max(1.0, page_x1 - page_x0)
    full_width_ratio = LAYOUT_RULES.reading_order.full_width_ratio
    return [(block.bbox[2] - block.bbox[0]) / page_width >= full_width_ratio for block in blocks]


def topological_block_order_from_pairs(
    blocks: list[ParsedBlock], pairs: Iterable[tuple[int, int]], full_width: list[bool]
) -> list[ParsedBlock]:
    if len(blocks) <= 2:
        return blocks
    n = len(blocks)
    in_degree = [0] * n
    graph: dict[int, list[int]] = defaultdict(list)
    rules = LAYOUT_RULES.reading_order
    stacking = rules.stacking_tolerance
    side_by_side = rules.side_by_side_overlap
    stacked_overlap = rules.stacked_overlap_ratio

    for i, j in pairs:
        ax0, ay0, ax1, ay1 = blocks[i].bbox
        a_is_full_width = full_width[i]
        bx0, by0, bx1, by1 = blocks[j].bbox
        b_is_full_width = full_width[j]

        if a_is_full_width and not b_is_full_width and ay0 >= by1 - stacking:
            graph[i].append(j)
            in_degree[j] += 1
        elif b_is_full_width and not a_is_full_width and by0 >= ay1 - stacking:
            graph[j].append(i)
            in_degree[i] += 1
        elif not a_is_full_width and not b_is_full_width:
            if horizontal_overlap_ratio(blocks[i].bbox, blocks[j].bbox) >= stacked_overlap:
                if ay0 >= by1 - stacking:
                    graph[i].append(j)
                    in_degree[j] += 1
                elif by0 >= ay1 - stacking:
                    graph[j].append(i)
                    in_degree[i] += 1
            elif ax1 <= bx0 + stacking and interval_overlap(ay0, ay1, by0, by1) > side_by_side:
                graph[i].append(j)
                in_degree[j] += 1
            elif bx1 <= ax0 + stacking and interval_overlap(ay0, ay1, by0, by1) > side_by_side:
                graph[j].append(i)
                in_degree[i] += 1

    ready: list[tuple[float, float, int, int]] = []
    serial = 0
    for index in range(n):
        if in_degree[index] == 0:
            heappush(ready, (-blocks[index].bbox[3], blocks[index].bbox[0], serial, index))
            serial += 1
    result: list[int] = []

    while ready:
        _negative_top, _left, _serial, curr = heappop(ready)
        result.append(curr)
        for nxt in graph[curr]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                heappush(ready, (-blocks[nxt].bbox[3], blocks[nxt].bbox[0], serial, nxt))
                serial += 1

    if len(result) == n:
        return [blocks[i] for i in result]
    return blocks


def topological_block_order(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    if len(blocks) <= 2:
        return blocks
    full_width = full_width_blocks(blocks)
    pairs = sparse_block_candidate_pairs(blocks, full_width)
    return topological_block_order_from_pairs(blocks, pairs, full_width)


def interleave_columnar_blocks(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    rules = LAYOUT_RULES.reading_order
    candidates = [block for block in blocks if len(block.lines) >= rules.interleave_min_lines]
    if len(candidates) < rules.interleave_min_blocks:
        return blocks
    x0s = [block.bbox[0] for block in candidates]
    x1s = [block.bbox[2] for block in candidates]
    if (
        max(x0s) - min(x0s) > rules.interleave_left_spread
        or max(x1s) - min(x1s) > rules.interleave_right_spread
    ):
        return blocks
    merged_lines = tuple(line for block in candidates for line in block.lines)
    boxes = numpy.asarray(tuple(line_bbox(line) for line in merged_lines), dtype=numpy.float32)
    ordered = row_order_indexes(numpy.arange(len(merged_lines)), boxes)
    merged = ParsedBlock(
        lines=tuple(merged_lines[int(index)] for index in ordered),
        bbox=block_bbox(merged_lines),
    )
    candidate_ids = {id(block) for block in candidates}
    return [merged, *(block for block in blocks if id(block) not in candidate_ids)]


def column_major_prose(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    rules = LAYOUT_RULES.reading_order
    output: list[ParsedBlock] = []
    for block in blocks:
        if len(block.lines) < rules.column_major_min_lines:
            output.append(block)
            continue
        line_starts = numpy.fromiter(
            (line_bbox(line)[0] for line in block.lines), dtype=numpy.float64
        )
        starts = numpy.sort(line_starts)
        clusters = [
            float(starts[group[0]])
            for group in cluster_1d(starts, rules.column_major_column_gap, linkage="anchor")
        ]
        if len(clusters) < rules.column_major_min_columns:
            output.append(block)
            continue
        alphabetic = 0
        total = 0
        for line in block.lines:
            for character in line.line.text:
                is_alpha = character.isalpha()
                alphabetic += is_alpha
                total += is_alpha or character.isdigit()
        if alphabetic / max(1, total) < rules.column_major_min_alpha_ratio:
            output.append(block)
            continue
        cluster_values = numpy.asarray(clusters, dtype=numpy.float64)
        insertion = numpy.searchsorted(cluster_values, line_starts)
        left = numpy.maximum(0, insertion - 1)
        right = numpy.minimum(len(cluster_values) - 1, insertion)
        choose_right = numpy.abs(line_starts - cluster_values[right]) < numpy.abs(
            line_starts - cluster_values[left]
        )
        line_clusters = numpy.where(choose_right, right, left)
        transitions = numpy.count_nonzero(line_clusters[1:] != line_clusters[:-1])
        if transitions / max(1, len(line_clusters) - 1) < rules.column_major_min_transition_ratio:
            output.append(block)
            continue
        columns: list[list[ParsedLine]] = [[] for _ in clusters]
        for line, assigned in zip(block.lines, line_clusters, strict=True):
            columns[int(assigned)].append(line)
        ordered = tuple(
            line
            for column in columns
            for line in sorted(column, key=lambda item: -line_bbox(item)[1])
        )
        output.append(replace(block, lines=ordered))
    return output


def transpose_numeric_table_blocks(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    rules = LAYOUT_RULES.reading_order
    output: list[ParsedBlock] = []
    for block in blocks:
        if len(block.lines) < rules.transpose_min_lines:
            output.append(block)
            continue
        starts = sorted(line_bbox(line)[0] for line in block.lines)
        columns = cluster_1d(starts, rules.transpose_column_gap, linkage="anchor")
        if len(columns) < rules.transpose_min_columns:
            output.append(block)
            continue
        text = " ".join(line.line.text for line in block.lines)
        numeric = sum(map(str.isdigit, text))
        alphanumeric = sum(map(str.isalnum, text))
        if numeric / max(1, alphanumeric) < rules.transpose_min_digit_ratio:
            output.append(block)
            continue
        boxes = numpy.asarray(tuple(line_bbox(line) for line in block.lines))
        indexes = row_order_indexes(numpy.arange(len(block.lines)), boxes)
        ordered = tuple(block.lines[int(index)] for index in indexes)
        output.append(replace(block, lines=ordered))
    return output


def has_repeated_block_columns(blocks: tuple[ParsedBlock, ...]) -> bool:
    rules = LAYOUT_RULES.reading_order
    bounded = tuple(block.bbox for block in blocks if block.bbox is not None)
    if len(bounded) < rules.repeated_columns_min_blocks:
        return False
    top = max(box[3] for box in bounded)
    bottom = min(box[1] for box in bounded)
    cutoff = top - (top - bottom) * rules.repeated_columns_top_ratio
    starts = sorted(box[0] for box in bounded if box[3] >= cutoff)
    if len(starts) < rules.repeated_columns_min_top_blocks:
        return False
    clusters = cluster_1d(starts, rules.repeated_columns_gap, linkage="chain")
    return (
        sum(len(cluster) >= rules.repeated_columns_min_members for cluster in clusters)
        >= rules.repeated_columns_min_columns
    )


REORDER_PASSES: tuple[BlockReorderPass, ...] = (
    interleave_columnar_blocks,
    transpose_numeric_table_blocks,
    column_major_prose,
)
