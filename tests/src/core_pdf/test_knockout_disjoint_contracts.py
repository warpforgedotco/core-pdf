# SPDX-License-Identifier: AGPL-3.0-only


from typing import Any

import numpy
import pytest

from core_pdf.impl.capture_records import CapturedPath
from core_pdf.impl.render_model import (
    PathPaintItem,
    PathPaintKind,
    RasterGroup,
    display_item,
)
from core_pdf.impl.render_target import RasterTarget
from tests.src.core_pdf import raster_support


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


def pixel_view() -> numpy.ndarray:
    return numpy.zeros((1, 1, 4), dtype=numpy.uint8)


def knockout_group() -> RasterGroup:
    return RasterGroup(bytearray(4), view=pixel_view(), knockout=True, painted_boxes=[])


def knockout_target(width: int = 40, height: int = 40) -> RasterTarget:
    target = raster_support.make_target(width=width, height=height)
    target.push_group(bytearray(width * height * 4), None, None, isolated=True, knockout=True)
    return target


def painted_boxes(target: RasterTarget) -> list[Any] | None:
    return target.buffer_stack[-1].painted_boxes


class TookGroup(Exception):
    pass


def needs_group(target: RasterTarget, item: Any, monkeypatch: pytest.MonkeyPatch) -> bool:
    def refuse(self: RasterTarget, *args: Any, **kwargs: Any) -> None:
        raise TookGroup

    with monkeypatch.context() as patch:
        patch.setattr(RasterTarget, "push_scratch_group", refuse)
        try:
            target.paint_item(item)
        except TookGroup:
            return True
    return False


def test_a_group_that_does_not_knock_out_records_nothing() -> None:
    assert RasterGroup(bytearray(4), view=pixel_view()).painted_boxes is None


def test_disjoint_fills_skip_and_are_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    target = knockout_target()
    assert needs_group(target, fill_item((0, 0, 10, 10)), monkeypatch) is False
    assert needs_group(target, fill_item((20, 0, 30, 10)), monkeypatch) is False
    # Device rows run downward: page y 0 to 10 are the bottom rows of the 40-high target.
    assert painted_boxes(target) == [(0, 30, 10, 40), (20, 30, 30, 40)]


def test_an_overlapping_fill_takes_the_group(monkeypatch: pytest.MonkeyPatch) -> None:
    target = knockout_target()
    assert needs_group(target, fill_item((0, 0, 10, 10)), monkeypatch) is False
    assert needs_group(target, fill_item((9, 0, 20, 10)), monkeypatch) is True


def test_boxes_that_merely_touch_are_disjoint(monkeypatch: pytest.MonkeyPatch) -> None:
    target = knockout_target()
    assert needs_group(target, fill_item((0, 0, 10, 10)), monkeypatch) is False
    assert needs_group(target, fill_item((10, 0, 20, 10)), monkeypatch) is False


def test_separation_in_either_axis_is_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    target = knockout_target()
    assert needs_group(target, fill_item((0, 0, 10, 10)), monkeypatch) is False
    assert needs_group(target, fill_item((0, 10, 10, 20)), monkeypatch) is False


def test_only_plain_edge_array_fills_are_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    for item in (
        fill_item((0, 0, 10, 10), edge_array=False),
        fill_item((0, 0, 10, 10), kind=PathPaintKind.STROKE),
        fill_item((0, 0, 10, 10), kind=PathPaintKind.FILL_STROKE),
        fill_item((0, 0, 10, 10), pattern=object()),
        fill_item((0, 0, 10, 10), blend="Multiply"),
        fill_item(None),
    ):
        target = knockout_target()
        assert needs_group(target, item, monkeypatch) is True
        assert painted_boxes(target) == []


def test_a_fill_that_lands_nowhere_records_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    target = knockout_target()
    assert needs_group(target, fill_item((100, 100, 110, 110)), monkeypatch) is False
    assert painted_boxes(target) == []


def test_the_scan_gives_up_past_its_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    limit = RasterTarget.KNOCKOUT_DISJOINT_LIMIT
    width = limit * 2 + 8
    target = knockout_target(width, 8)
    for index in range(limit):
        item = fill_item((index * 2, 0, index * 2 + 1, 1))
        assert needs_group(target, item, monkeypatch) is False
    last = limit * 2
    assert needs_group(target, fill_item((last, 4, last + 1, 5)), monkeypatch) is False
    assert painted_boxes(target) == [(0, 0, width, 8)]
    assert needs_group(target, fill_item((last + 2, 6, last + 3, 7)), monkeypatch) is True


def grouped(target: RasterTarget, items: list[Any], monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    used: list[bool] = []
    original = RasterTarget.push_scratch_group

    def counting(self: RasterTarget, *args: Any, **kwargs: Any) -> None:
        used[-1] = True
        original(self, *args, **kwargs)

    monkeypatch.setattr(RasterTarget, "push_scratch_group", counting)
    for item in items:
        used.append(False)
        target.paint_item(item)
    return used


def stroke_item(x0: float, y0: float, x1: float, y1: float) -> PathPaintItem:
    path = CapturedPath()
    path.move_to(x0, y0)
    path.line_to(x1, y1)
    return PathPaintItem(
        PathPaintKind.STROKE,
        0,
        (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
        path,
        None,
        None,
        (0, 0, 0, 255),
        None,
        2.0,
        0,
        0,
        None,
        "nonzero",
        None,
        None,
    )


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


def test_a_stroke_records_the_pixels_it_painted(monkeypatch: pytest.MonkeyPatch) -> None:
    target = knockout_target()
    stroke = stroke_item(2, 2, 30, 2)
    assert grouped(target, [stroke], monkeypatch) == [True]
    boxes = target.buffer_stack[-1].painted_boxes
    assert boxes is not None
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert (x1 - x0, y1 - y0) != (target.width, target.height)


def test_a_fill_over_a_stroke_takes_a_group(monkeypatch: pytest.MonkeyPatch) -> None:
    target = knockout_target()
    items = [stroke_item(2, 2, 30, 2), fill_item((10, 1, 15, 3))]
    assert grouped(target, items, monkeypatch) == [True, True]


def test_a_fill_clear_of_a_stroke_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    target = knockout_target()
    items = [stroke_item(2, 2, 30, 2), fill_item((0, 30, 5, 35))]
    assert grouped(target, items, monkeypatch) == [True, False]


def nested_group(target: RasterTarget, items: list[Any]) -> None:
    target.paint_display_item(display_item("group-begin", 0))
    for item in items:
        target.paint_item(item)
    target.paint_display_item(display_item("group-end", 1))


def test_a_fill_over_a_nested_group_takes_a_group(monkeypatch: pytest.MonkeyPatch) -> None:
    target = knockout_target()
    nested_group(target, [stroke_item(2, 2, 30, 2)])
    assert grouped(target, [fill_item((10, 1, 15, 3))], monkeypatch) == [True]


def test_a_fill_clear_of_a_nested_group_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    target = knockout_target()
    nested_group(target, [stroke_item(2, 2, 30, 2)])
    assert grouped(target, [fill_item((0, 30, 5, 35))], monkeypatch) == [False]


def test_an_empty_nested_group_records_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    target = knockout_target()
    nested_group(target, [])
    assert target.buffer_stack[-1].painted_boxes == []
