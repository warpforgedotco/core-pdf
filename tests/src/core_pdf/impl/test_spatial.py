import math

import numpy

from core_pdf.impl.spatial import (
    BoxIndex,
    DisjointSet,
    band_rows,
    cluster_1d,
    interval_overlap_pairs,
)


def test_chain_linkage_compares_each_value_with_the_previous_member() -> None:
    assert cluster_1d([0.0, 10.0, 20.0, 31.0], 10.0, linkage="chain") == [[0, 1, 2], [3]]


def test_anchor_linkage_compares_each_value_with_the_first_member() -> None:
    assert cluster_1d([0.0, 10.0, 20.0, 31.0], 10.0, linkage="anchor") == [[0, 1], [2], [3]]


def test_mean_linkage_compares_each_value_with_the_running_mean() -> None:
    assert cluster_1d([0.0, 10.0, 14.0, 30.0], 10.0, linkage="mean") == [[0, 1, 2], [3]]


def test_non_finite_gaps_keep_each_linkage_comparison() -> None:
    values = [0.0, math.nan, 1.0]
    assert cluster_1d(values, 5.0, linkage="anchor") == [[0, 1, 2]]
    assert cluster_1d(values, 5.0, linkage="chain") == [[0], [1], [2]]
    assert cluster_1d(values, 5.0, linkage="mean") == [[0], [1], [2]]


def test_bands_group_descending_centers_within_the_tolerance_of_the_anchor() -> None:
    centers = [100.0, 97.0, 94.0, 80.0]
    assert band_rows(centers, 4.0, range(4)) == [[0, 1], [2], [3]]
    assert band_rows(centers, 4.0, [3, 0, 1, 2], linkage="window") == [[3], [0, 1], [2]]
    assert band_rows([math.nan, 1.0], 4.0, range(2), linkage="anchor") == [[0, 1]]
    assert band_rows([math.nan, 1.0], 4.0, range(2), linkage="window") == [[0], [1]]


def test_a_box_index_keeps_the_dtype_of_the_boxes_it_indexes() -> None:
    boxes = numpy.asarray([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=numpy.float32)
    index = BoxIndex.from_array(boxes)
    assert index.areas.dtype == numpy.float32
    assert index.intersection_areas((0.0, 0.0, 5.0, 5.0)).dtype == numpy.float32
    assert BoxIndex.from_boxes([(0.0, 0.0, 1.0, 1.0)]).boxes.dtype == numpy.float64


def test_box_index_overlap_ratios() -> None:
    index = BoxIndex.from_boxes([(0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 30.0, 10.0)])
    assert index.overlap_min((5.0, 0.0, 25.0, 10.0)).tolist() == [0.5, 0.5]
    assert index.overlap_of((5.0, 0.0, 25.0, 10.0)).tolist() == [0.25, 0.25]
    assert index.overlap_of((5.0, 5.0, 5.0, 5.0)).tolist() == [0.0, 0.0]
    assert index.matching_overlap_min((0.0, 0.0, 12.0, 10.0), 0.9).tolist() == [0]


def test_pairwise_intersections_are_chunked_by_query_rows() -> None:
    index = BoxIndex.from_boxes([(0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)])
    queries = numpy.asarray([[0, 0, 5, 5], [10, 0, 20, 10], [0, 0, 20, 10]], dtype=numpy.float64)
    chunks = list(index.pairwise_intersection(queries, 2))
    assert [start for start, _ in chunks] == [0, 2]
    assert numpy.vstack([areas for _, areas in chunks]).tolist() == [
        [25.0, 0.0],
        [0.0, 50.0],
        [100.0, 100.0],
    ]


def test_disjoint_sets_and_interval_pairs() -> None:
    disjoint = DisjointSet(4)
    disjoint.union(0, 2)
    disjoint.union(3, 2)
    assert disjoint.find(0) == disjoint.find(2) == disjoint.find(3) != disjoint.find(1)
    starts = numpy.asarray([0.0, 5.0, 10.0])
    ends = numpy.asarray([6.0, 10.0, 12.0])
    assert interval_overlap_pairs(starts, ends) == {(0, 1)}
