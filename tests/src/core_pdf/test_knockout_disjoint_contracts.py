# SPDX-License-Identifier: AGPL-3.0-only

"""Inside a knockout group, an element that misses everything painted so far
composites the same whether or not it goes through an elementary group, so it
skips one. These pin the decision, not the arithmetic: which items are
eligible, what gets recorded as painted, and that a genuine overlap is never
skipped."""

from types import SimpleNamespace
from typing import Any, cast

import numpy
import pytest

from core_pdf.impl.render.model import (
    DisplayListItem,
    ImagePaintItem,
    PathPaintItem,
    PathPaintKind,
    RasterGroup,
)
from core_pdf.impl.render.target import RasterTarget
from tests.src.core_pdf.test_pattern_rendering import make_target as make_real_target


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
    target.width = 10_000
    target.height = 10_000
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


def needs_group(target: RasterTarget, group: RasterGroup, item: Any) -> bool:
    """The decision paint_item makes, with the recording it does on the way."""
    target.buffer_stack = [group]
    boxes = group.painted_boxes
    assert boxes is not None
    return not target.knockout_paint_is_disjoint(item, boxes)


def test_a_group_that_does_not_knock_out_records_nothing() -> None:
    # The list is what paint_item tests before it reaches the decision at all,
    # and only a knockout group is given one.
    assert RasterGroup(bytearray(4)).painted_boxes is None


def test_disjoint_fills_skip_and_are_remembered() -> None:
    target, group = make_target(), knockout_group()
    assert needs_group(target, group, fill_item((0, 0, 10, 10))) is False
    assert needs_group(target, group, fill_item((20, 0, 30, 10))) is False
    assert group.painted_boxes == [(0, 0, 10, 10), (20, 0, 30, 10)]


def test_an_overlapping_fill_takes_the_group() -> None:
    target, group = make_target(), knockout_group()
    assert needs_group(target, group, fill_item((0, 0, 10, 10))) is False
    # Shares a pixel column with the first.
    assert needs_group(target, group, fill_item((9, 0, 20, 10))) is True
    # It still painted, through its group, so it is recorded like any other.
    assert group.painted_boxes == [(0, 0, 10, 10), (9, 0, 20, 10)]


def test_boxes_that_merely_touch_are_disjoint() -> None:
    """The box is exactly what fill_path paints into, so abutting is a miss."""
    target, group = make_target(), knockout_group()
    assert needs_group(target, group, fill_item((0, 0, 10, 10))) is False
    assert needs_group(target, group, fill_item((10, 0, 20, 10))) is False


def test_separation_in_either_axis_is_enough() -> None:
    target, group = make_target(), knockout_group()
    assert needs_group(target, group, fill_item((0, 0, 10, 10))) is False
    assert needs_group(target, group, fill_item((0, 10, 10, 20))) is False


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
        assert needs_group(target, knockout_group(), item) is True


@pytest.mark.parametrize(
    "item",
    [
        fill_item((0, 0, 10, 10), edge_array=False),
        fill_item((0, 0, 10, 10), kind=PathPaintKind.STROKE),
        fill_item(None),
    ],
)
def test_an_unbounded_element_occupies_the_whole_page(item: Any) -> None:
    """A stroke reaches beyond its bbox and a fill without an edge array
    derives its own, so neither says where it painted. Something painted, so
    nothing after it can be proved to miss it."""
    target, group = make_target(), knockout_group()
    assert needs_group(target, group, item) is True
    assert group.painted_boxes == [(0, 0, target.width, target.height)]
    # A fill nowhere near it is no longer eligible.
    assert needs_group(target, group, fill_item((20, 20, 30, 30))) is True


def test_a_bounded_but_ineligible_fill_records_only_its_own_box() -> None:
    """A pattern fill cannot skip the group, but it does paint inside its box
    and nowhere else, so it does not have to retire the test."""
    target, group = make_target(), knockout_group()
    assert needs_group(target, group, fill_item((0, 0, 10, 10), pattern=object())) is True
    assert group.painted_boxes == [(0, 0, 10, 10)]
    assert needs_group(target, group, fill_item((20, 0, 30, 10))) is False


def test_a_fill_that_lands_nowhere_records_nothing() -> None:
    target, group = make_target(), knockout_group()
    target.clip = cast(Any, SimpleNamespace)(clipped_pixel_box=lambda bbox: None)
    assert needs_group(target, group, fill_item((0, 0, 10, 10))) is False
    assert group.painted_boxes == []


def test_the_scan_gives_up_past_its_limit() -> None:
    target, group = make_target(), knockout_group()
    for index in range(RasterTarget.KNOCKOUT_DISJOINT_LIMIT):
        assert needs_group(target, group, fill_item((index * 2, 0, index * 2 + 1, 1))) is False
    # The one that fills the list is still genuinely disjoint, but recording it
    # trips the limit, and past there the page stands in for the boxes.
    assert needs_group(target, group, fill_item((5_000, 5_000, 5_001, 5_001))) is False
    assert group.painted_boxes == [(0, 0, target.width, target.height)]
    assert needs_group(target, group, fill_item((9_000, 9_000, 9_001, 9_001))) is True


# The unit tests above drive the decision directly. These drive paint_item, so
# they also cover what the elementary-group path leaves behind: an element that
# took a group still painted, and occupancy has to survive push and pop.


def grouped(target: RasterTarget, items: list[Any], monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Whether each item went through an elementary group, painting for real."""
    used: list[bool] = []
    original = RasterTarget.push_elementary_group

    def counting(self: RasterTarget, **kwargs: Any) -> None:
        used[-1] = True
        original(self, **kwargs)

    monkeypatch.setattr(RasterTarget, "push_elementary_group", counting)
    for item in items:
        used.append(False)
        target.paint_item(item)
    return used


def knockout_target() -> RasterTarget:
    target = make_real_target(width=40, height=40)
    target.push_group(bytearray(40 * 40 * 4), None, None, isolated=True, knockout=True)
    return target


def test_painting_three_disjoint_fills_takes_no_group(monkeypatch: pytest.MonkeyPatch) -> None:
    target = knockout_target()
    items = [fill_item((0, 0, 5, 5)), fill_item((10, 0, 15, 5)), fill_item((20, 0, 25, 5))]
    assert grouped(target, items, monkeypatch) == [False, False, False]


def test_painting_a_fill_back_over_the_first_takes_a_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = knockout_target()
    items = [fill_item((0, 0, 5, 5)), fill_item((10, 0, 15, 5)), fill_item((1, 1, 4, 4))]
    assert grouped(target, items, monkeypatch) == [False, False, True]


def test_a_fill_over_an_image_takes_a_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """An image composites in through an elementary group and leaves pixels
    behind, but nothing the item carries bounds them: blit_image derives its
    own quad. A later fill must not be told the backdrop is untouched."""
    target = knockout_target()
    image = ImagePaintItem("image", 0, (0, 0, 5, 5), None, None, None, None, None, None, None, {})
    items = [image, fill_item((0, 0, 5, 5))]
    assert grouped(target, items, monkeypatch) == [True, True]


def test_a_fill_over_a_stroke_takes_a_group(monkeypatch: pytest.MonkeyPatch) -> None:
    target = knockout_target()
    items = [fill_item((0, 0, 5, 5), kind=PathPaintKind.STROKE), fill_item((20, 0, 25, 5))]
    assert grouped(target, items, monkeypatch) == [True, True]


def test_a_fill_after_a_nested_group_takes_a_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nested group composites straight into the knockout parent without
    passing through paint_item at all."""
    target = knockout_target()
    used: list[bool] = []
    original = RasterTarget.push_elementary_group

    def counting(self: RasterTarget, **kwargs: Any) -> None:
        used[-1] = True
        original(self, **kwargs)

    monkeypatch.setattr(RasterTarget, "push_elementary_group", counting)
    target.paint_display_item(DisplayListItem("group-begin", 0))
    target.paint_display_item(DisplayListItem("group-end", 1))
    used.append(False)
    target.paint_item(fill_item((0, 0, 5, 5)))
    assert used == [True]
