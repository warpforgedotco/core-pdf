# SPDX-License-Identifier: AGPL-3.0-only
"""Block-level reading-order evidence and repair heuristics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from heapq import heappop, heappush
from itertools import combinations

import numpy

from core_pdf.impl.extract.contracts import ParsedBlock, ParsedLine, ReadingOrderEvidence
from core_pdf.impl.layout import regions as layout_regions
from core_pdf.impl.model.geometry import horizontal_overlap_ratio, interval_overlap


def internal_block_bbox(lines: tuple[ParsedLine, ...]) -> tuple[float, float, float, float]:
    boxes = numpy.asarray(tuple(line.bbox for line in lines), dtype=numpy.float32)
    return (
        float(numpy.min(boxes[:, 0])),
        float(numpy.min(boxes[:, 1])),
        float(numpy.max(boxes[:, 2])),
        float(numpy.max(boxes[:, 3])),
    )


def internal_inversion_count(values: tuple[int, ...]) -> int:
    """Count source-order inversions in O(n log n) time and O(n) memory."""
    if len(values) < 2:
        return 0
    ranks = {value: rank + 1 for rank, value in enumerate(sorted(values))}
    tree = [0] * (len(values) + 1)
    inversions = 0
    for seen, value in enumerate(values):
        rank = ranks[value]
        prefix = 0
        index = rank
        while index:
            prefix += tree[index]
            index -= index & -index
        inversions += seen - prefix
        index = rank
        while index < len(tree):
            tree[index] += 1
            index += index & -index
    return inversions


def internal_reading_order_evidence(
    blocks: tuple[ParsedBlock, ...],
) -> ReadingOrderEvidence:
    """Summarize repair strength and ambiguity for an ordered block sequence."""
    lines = tuple(line for block in blocks for line in block.lines)
    sequences = tuple(line.sequence for line in lines)
    inversions = internal_inversion_count(sequences)
    maximum = len(lines) * (len(lines) - 1) // 2
    rotations = {line.rotation % 360 for line in lines}
    mixed_rotation_block = any(
        len({line.rotation % 360 for line in block.lines}) > 1 for block in blocks
    )
    columns = {block.column_index for block in blocks if block.column_index is not None}
    repaired = inversions > 0
    ambiguous = mixed_rotation_block
    confidence = 0.5 if ambiguous else (0.85 if len(rotations) > 1 else 1.0)
    return ReadingOrderEvidence(
        line_count=len(lines),
        source_inversions=inversions,
        source_inversion_ratio=inversions / maximum if maximum else 0.0,
        column_count=max(1, len(columns)) if lines else 0,
        rotation_count=len(rotations),
        repaired=repaired,
        ambiguous=ambiguous,
        confidence=confidence,
        strategy="geometric-repair" if repaired else "source-stable",
    )


def internal_interval_overlap_pairs(
    starts: numpy.ndarray, ends: numpy.ndarray
) -> set[tuple[int, int]]:
    """Return index pairs whose open intervals overlap, using a sweep line."""
    order = numpy.argsort(starts, kind="stable")
    active: set[int] = set()
    ending: list[tuple[float, int]] = []
    pairs: set[tuple[int, int]] = set()
    for raw_index in order:
        index = int(raw_index)
        start = float(starts[index])
        while ending and ending[0][0] <= start:
            _end, expired = heappop(ending)
            active.discard(expired)
        for other in active:
            pairs.add((other, index) if other < index else (index, other))
        active.add(index)
        heappush(ending, (float(ends[index]), index))
    return pairs


def internal_sparse_block_candidate_pairs(
    blocks: list[ParsedBlock], full_width: list[bool]
) -> list[tuple[int, int]]:
    boxes = numpy.asarray(tuple(block.bbox for block in blocks), dtype=numpy.float64)
    pairs = internal_interval_overlap_pairs(boxes[:, 0], boxes[:, 2])
    pairs.update(internal_interval_overlap_pairs(boxes[:, 1], boxes[:, 3]))
    full_width_indexes = [index for index, value in enumerate(full_width) if value]
    for index in full_width_indexes:
        pairs.update(
            (min(index, other), max(index, other)) for other in range(len(blocks)) if other != index
        )
    return sorted(pairs)


def internal_topological_block_order_from_pairs(
    blocks: list[ParsedBlock], pairs: Iterable[tuple[int, int]]
) -> list[ParsedBlock]:
    """Sort blocks into topological reading order using a spatial predecessor DAG.

    Full-width header blocks enforce strict vertical precedence over all child columns,
    and column blocks are ordered left-to-right while preserving top-down sequence inside
    each column.
    """
    if len(blocks) <= 2:
        return blocks
    page_x0 = min(block.bbox[0] for block in blocks)
    page_x1 = max(block.bbox[2] for block in blocks)
    page_width = max(1.0, page_x1 - page_x0)

    n = len(blocks)
    full_width = [(block.bbox[2] - block.bbox[0]) / page_width >= 0.70 for block in blocks]
    in_degree = [0] * n
    graph: dict[int, list[int]] = defaultdict(list)

    for i, j in pairs:
        ax0, ay0, ax1, ay1 = blocks[i].bbox
        a_is_full_width = full_width[i]
        bx0, by0, bx1, by1 = blocks[j].bbox
        b_is_full_width = full_width[j]

        # Rule 1: Full-width header preceding child blocks below it
        if a_is_full_width and not b_is_full_width and ay0 >= by1 - 2.0:
            graph[i].append(j)
            in_degree[j] += 1
        elif b_is_full_width and not a_is_full_width and by0 >= ay1 - 2.0:
            graph[j].append(i)
            in_degree[i] += 1
        elif not a_is_full_width and not b_is_full_width:
            if horizontal_overlap_ratio(blocks[i].bbox, blocks[j].bbox) >= 0.45:
                if ay0 >= by1 - 2.0:
                    graph[i].append(j)
                    in_degree[j] += 1
                elif by0 >= ay1 - 2.0:
                    graph[j].append(i)
                    in_degree[i] += 1
            elif ax1 <= bx0 + 2.0 and interval_overlap(ay0, ay1, by0, by1) > 4.0:
                # Column A is strictly left of Column B with vertical overlap
                graph[i].append(j)
                in_degree[j] += 1
            elif bx1 <= ax0 + 2.0 and interval_overlap(ay0, ay1, by0, by1) > 4.0:
                # Column B is strictly left of Column A with vertical overlap
                graph[j].append(i)
                in_degree[i] += 1

    # Kahn's algorithm with the same stable top-down, left-to-right priority as
    # the former repeatedly sorted list, but O(log N) queue operations.
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


def internal_topological_block_order_quadratic(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    """Reference implementation used for small inputs and equivalence tests."""
    return internal_topological_block_order_from_pairs(blocks, combinations(range(len(blocks)), 2))


def internal_topological_block_order(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    if len(blocks) < 64:
        return internal_topological_block_order_quadratic(blocks)
    page_x0 = min(block.bbox[0] for block in blocks)
    page_x1 = max(block.bbox[2] for block in blocks)
    page_width = max(1.0, page_x1 - page_x0)
    full_width = [(block.bbox[2] - block.bbox[0]) / page_width >= 0.70 for block in blocks]
    pairs = internal_sparse_block_candidate_pairs(blocks, full_width)
    return internal_topological_block_order_from_pairs(blocks, pairs)


def internal_interleave_columnar_blocks(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    """Interleave line fragments when a scanned table was split into columns."""
    candidates = [block for block in blocks if len(block.lines) >= 20]
    if len(candidates) < 3:
        return blocks
    x0s = [block.bbox[0] for block in candidates]
    x1s = [block.bbox[2] for block in candidates]
    if max(x0s) - min(x0s) > 20.0 or max(x1s) - min(x1s) > 30.0:
        return blocks
    merged_lines = tuple(line for block in candidates for line in block.lines)
    boxes = numpy.asarray(tuple(line.bbox for line in merged_lines), dtype=numpy.float32)
    ordered = layout_regions.internal_row_order_indexes(numpy.arange(len(merged_lines)), boxes)
    merged = ParsedBlock(
        lines=tuple(merged_lines[int(index)] for index in ordered),
        bbox=internal_block_bbox(merged_lines),
    )
    candidate_ids = {id(block) for block in candidates}
    return [merged, *(block for block in blocks if id(block) not in candidate_ids)]


def internal_column_major_prose(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    """Use column-major order for wide prose blocks from magazine-style pages."""
    output: list[ParsedBlock] = []
    for block in blocks:
        if len(block.lines) < 80:
            output.append(block)
            continue
        alphabetic = 0
        total = 0
        for line in block.lines:
            for character in line.text:
                is_alpha = character.isalpha()
                alphabetic += is_alpha
                total += is_alpha or character.isdigit()
        line_starts = numpy.fromiter((line.bbox[0] for line in block.lines), dtype=numpy.float64)
        starts = numpy.sort(line_starts)
        clusters: list[float] = []
        for start in starts:
            if not clusters or start - clusters[-1] > 40.0:
                clusters.append(float(start))
        if len(clusters) < 3 or alphabetic / max(1, total) < 0.45:
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
        transitions = sum(
            left != right for left, right in zip(line_clusters, line_clusters[1:], strict=False)
        )
        if transitions / max(1, len(line_clusters) - 1) < 0.25:
            output.append(block)
            continue
        # Stable column-major ordering from the nearest-column assignment.
        # Re-deriving membership with a fixed window emitted lines close to
        # two columns twice and silently dropped lines close to none.
        columns: list[list[ParsedLine]] = [[] for internal_cluster in clusters]
        for line, assigned in zip(block.lines, line_clusters, strict=True):
            columns[int(assigned)].append(line)
        ordered = tuple(
            line for column in columns for line in sorted(column, key=lambda item: -item.bbox[1])
        )
        output.append(replace(block, lines=ordered))
    return output


def internal_transpose_numeric_table_blocks(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    """Transpose vectorized table columns into row-wise reading order."""
    output: list[ParsedBlock] = []
    for block in blocks:
        if len(block.lines) < 300:
            output.append(block)
            continue
        text = " ".join(line.text for line in block.lines)
        numeric = sum(character.isdigit() for character in text)
        alphanumeric = sum(character.isalnum() for character in text)
        starts = sorted(line.bbox[0] for line in block.lines)
        columns: list[float] = []
        for start in starts:
            if not columns or start - columns[-1] > 8.0:
                columns.append(start)
        if numeric / max(1, alphanumeric) < 0.25 or len(columns) < 20:
            output.append(block)
            continue
        ordered = tuple(sorted(block.lines, key=lambda line: (line.bbox[0], line.bbox[1])))
        output.append(replace(block, lines=ordered))
    return output


def internal_has_repeated_block_columns(blocks: tuple[ParsedBlock, ...]) -> bool:
    """Identify pages whose blocks form a repeated multi-column grid."""
    bounded = tuple(block.bbox for block in blocks if block.bbox is not None)
    if len(bounded) < 6:
        return False
    top = max(box[3] for box in bounded)
    bottom = min(box[1] for box in bounded)
    cutoff = top - (top - bottom) * 0.55
    starts = sorted(box[0] for box in bounded if box[3] >= cutoff)
    if len(starts) < 6:
        return False
    clusters: list[list[float]] = []
    for start in starts:
        if clusters and start - clusters[-1][-1] <= 16.0:
            clusters[-1].append(start)
        else:
            clusters.append([start])
    return sum(len(cluster) >= 3 for cluster in clusters) >= 3
