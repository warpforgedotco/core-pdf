# SPDX-License-Identifier: AGPL-3.0-only
"""A mark lying in the gutter must not hide the column boundary.

Reading order is found by projecting the boxes and looking for a column of
empty space. Anything overlapping that space hides it, which on a recognised
page happens constantly: a centred page number, a speck of noise, a stray
accent. The columns then read interleaved for the rest of the page.

The boundary search is exercised directly here. Driving it through a whole
document instead would mostly test which cut the recursion happens to reach
first, and a synthetic page laid out to force that path stops resembling the
two-column pages this protects.
"""

from __future__ import annotations

import numpy

from core_pdf.impl.layout.regions import (
    internal_best_projection_gap,
    internal_column_gap_minimum,
    internal_gutter_tolerating_contained_boxes,
)


def two_columns(mark: tuple[float, float] | None = None) -> numpy.ndarray:
    """Two columns split by a narrow gutter, optionally with a mark inside it.

    The gutter runs from 93 to 120. It is deliberately tight, so a mark in the
    middle leaves clear space on either side narrower than the smallest gap
    that counts as a boundary -- exactly what a wide gutter would hide.
    """
    boxes = []
    for index in range(8):
        top = 700.0 - index * 13.0
        boxes.append((60.0, top - 11.0, 93.0, top))
        boxes.append((120.0, top - 11.0, 160.0, top))
    if mark is not None:
        boxes.append((mark[0], 550.0, mark[1], 559.0))
    return numpy.asarray(boxes, dtype=numpy.float32)


def test_a_clean_gutter_is_found_by_the_plain_projection() -> None:
    boxes = two_columns()
    minimum = internal_column_gap_minimum(boxes)
    assert internal_best_projection_gap(boxes, 0, minimum) is not None


def test_a_mark_in_the_gutter_hides_it_from_the_plain_projection() -> None:
    # The premise of the whole exercise: one mark and the boundary vanishes.
    boxes = two_columns(mark=(102.0, 112.0))
    minimum = internal_column_gap_minimum(boxes)
    assert internal_best_projection_gap(boxes, 0, minimum) is None


def test_the_boundary_is_still_found_around_a_mark_inside_it() -> None:
    boxes = two_columns(mark=(102.0, 112.0))
    cut = internal_gutter_tolerating_contained_boxes(boxes, internal_column_gap_minimum(boxes))
    assert cut is not None
    # The cut falls in the gutter, so the columns land on opposite sides.
    assert 93.0 < cut < 120.0


def test_a_box_reaching_into_both_columns_still_joins_them() -> None:
    # A heading spanning the columns is not gutter furniture, and must not be
    # ignored the way a contained mark is.
    boxes = two_columns()
    spanning = numpy.asarray([(70.0, 550.0, 150.0, 559.0)], dtype=numpy.float32)
    boxes = numpy.concatenate((boxes, spanning))
    assert (
        internal_gutter_tolerating_contained_boxes(boxes, internal_column_gap_minimum(boxes))
        is None
    )


def test_a_single_column_is_not_split_down_the_middle() -> None:
    boxes = numpy.asarray(
        [(60.0, 700.0 - index * 13.0, 400.0, 711.0 - index * 13.0) for index in range(12)],
        dtype=numpy.float32,
    )
    assert (
        internal_gutter_tolerating_contained_boxes(boxes, internal_column_gap_minimum(boxes))
        is None
    )


def three_columns() -> numpy.ndarray:
    """Three columns, so there are two gutters rather than one.

    Columns occupy 60-140, 160-240 and 260-340, leaving gutters at 140-160 and
    240-260.  Nothing reaches across from the first column to the third, which
    is what lets a span drawn between the two gutters look empty from the
    outside while actually containing the whole middle column.
    """
    boxes = []
    for index in range(9):
        top = 700.0 - index * 13.0
        for left, right in ((60.0, 140.0), (160.0, 240.0), (260.0, 340.0)):
            boxes.append((left, top - 11.0, right, top))
    return numpy.asarray(boxes, dtype=numpy.float32)


def test_two_gutters_do_not_merge_into_a_cut_through_the_column_between_them() -> None:
    boxes = three_columns()
    cut = internal_gutter_tolerating_contained_boxes(boxes, internal_column_gap_minimum(boxes))
    assert cut is not None
    # Landing between 160 and 240 would split the middle column down the middle
    # and interleave it with its neighbour for the rest of the page.
    assert not 160.0 <= cut <= 240.0
    assert 140.0 <= cut <= 160.0 or 240.0 <= cut <= 260.0


def test_a_gutter_survives_the_coarse_sampling_of_a_wide_region() -> None:
    """A real gutter must not be lost to the resolution it is measured at.

    The boundary is searched at a fixed number of sampled positions, so the
    wider the region the further apart they fall and the more a gutter's
    measured width understates it.  This gutter is comfortably above the floor
    but only a little wider than the spacing of the samples across a 500pt
    region.
    """
    boxes = []
    for index in range(10):
        top = 700.0 - index * 13.0
        boxes.append((60.0, top - 11.0, 300.0, top))
        boxes.append((313.0, top - 11.0, 560.0, top))
    region = numpy.asarray(boxes, dtype=numpy.float32)

    cut = internal_gutter_tolerating_contained_boxes(region, internal_column_gap_minimum(region))

    assert cut is not None
    assert 300.0 <= cut <= 313.0
