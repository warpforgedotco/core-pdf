# SPDX-License-Identifier: AGPL-3.0-only

import pytest

from core_pdf_cythonized import SlotLayout


class Point:
    __slots__ = ("x", "y", "label")

    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label


def test_build_sets_the_named_slots_without_init():
    layout = SlotLayout(Point, ("label", "x", "y"))
    point = layout.build(("a", 1.5, 2))
    assert type(point) is Point
    assert (point.x, point.y, point.label) == (1.5, 2, "a")


def test_build_needs_one_value_per_slot():
    with pytest.raises(ValueError):
        SlotLayout(Point, ("x", "y")).build((1,))
