# SPDX-License-Identifier: AGPL-3.0-only

"""Inside a knockout group, an element that misses everything painted so far
composites the same whether or not it goes through an elementary group, so it
skips one. These pin the decision, not the arithmetic: which items are
eligible, and that a genuine overlap is never skipped."""

from types import SimpleNamespace
from typing import Any, cast

import numpy

from core_pdf.impl.render.model import PathPaintItem, PathPaintKind, RasterGroup
from core_pdf.impl.render.target import RasterTarget


def make_target() -> RasterTarget:
    target = RasterTarget.__new__(RasterTarget)
    # Only clipped_pixel_box is reached; the rest of ClipState is irrelevant
    # to the decision under test.
    target.clip = cast(Any, SimpleNamespace)(
        clipped_pixel_box=lambda bbox: (
            None,
            (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
        )
    )
    return target


def fill_item(bbox, *, edge_array=True, kind=PathPaintKind.FILL, pattern=None, blend=None):
    return PathPaintItem(
        kind,
        0,
        bbox,
        None,
        (0, 0, 0, 255),
        None,
        None,
        None,
        0.0,
        0,
        0,
        None,
        "nonzero",
        blend,
        None,
        edge_array=numpy.zeros((1, 4)) if edge_array else None,
        fill_pattern=pattern,
    )


def knockout_group() -> RasterGroup:
    return RasterGroup(bytearray(4), knockout=True, painted_boxes=[])


def test_a_group_that_does_not_knock_out_never_skips() -> None:
    target = make_target()
    plain = RasterGroup(bytearray(4))
    assert plain.painted_boxes is None
    assert target.knockout_needs_group(fill_item((0, 0, 10, 10)), plain) is True


def test_disjoint_fills_skip_and_are_remembered() -> None:
    target, group = make_target(), knockout_group()
    assert target.knockout_needs_group(fill_item((0, 0, 10, 10)), group) is False
    assert target.knockout_needs_group(fill_item((20, 0, 30, 10)), group) is False
    assert group.painted_boxes == [(0, 0, 10, 10), (20, 0, 30, 10)]


def test_an_overlapping_fill_takes_the_group() -> None:
    target, group = make_target(), knockout_group()
    assert target.knockout_needs_group(fill_item((0, 0, 10, 10)), group) is False
    # Shares a pixel column with the first.
    assert target.knockout_needs_group(fill_item((9, 0, 20, 10)), group) is True
    # Rejected items are not recorded; only what actually painted directly is.
    assert group.painted_boxes == [(0, 0, 10, 10)]


def test_boxes_that_merely_touch_are_disjoint() -> None:
    """The box is exactly what fill_path paints into, so abutting is a miss."""
    target, group = make_target(), knockout_group()
    assert target.knockout_needs_group(fill_item((0, 0, 10, 10)), group) is False
    assert target.knockout_needs_group(fill_item((10, 0, 20, 10)), group) is False


def test_separation_in_either_axis_is_enough() -> None:
    target, group = make_target(), knockout_group()
    assert target.knockout_needs_group(fill_item((0, 0, 10, 10)), group) is False
    assert target.knockout_needs_group(fill_item((0, 10, 10, 20)), group) is False


def test_only_plain_edge_array_fills_are_eligible() -> None:
    target = make_target()
    for item in (
        fill_item((0, 0, 10, 10), edge_array=False),
        fill_item((0, 0, 10, 10), kind=PathPaintKind.STROKE),
        fill_item((0, 0, 10, 10), kind=PathPaintKind.FILL_STROKE),
        fill_item((0, 0, 10, 10), pattern=object()),
        fill_item((0, 0, 10, 10), blend="Multiply"),
        fill_item(None),
    ):
        assert target.knockout_needs_group(item, knockout_group()) is True


def test_the_scan_gives_up_past_its_limit() -> None:
    target, group = make_target(), knockout_group()
    for index in range(RasterTarget.KNOCKOUT_DISJOINT_LIMIT):
        assert (
            target.knockout_needs_group(fill_item((index * 2, 0, index * 2 + 1, 1)), group) is False
        )
    assert target.knockout_needs_group(fill_item((10_000, 0, 10_001, 1)), group) is True
