# SPDX-License-Identifier: AGPL-3.0-only
"""A full-width element divides the page it is set across.

Elements are ordered by the same recursive cut that orders lines, and that cut
finds columns by looking for a vertical gap running the height of the region.
A table or figure set across both columns puts a box over that gap. Ordering
the elements without saying the box is an obstacle leaves no column split
available, so a two-column page falls back to row order and reads its columns
interleaved -- one line from the left, one from the right, for the whole page.

The order is exercised directly. Driving it through a document would mostly
test which cut the recursion reaches first, and a synthetic page laid out to
force that path stops resembling the pages this protects.
"""

from __future__ import annotations

from core_pdf.impl._impl.extract.block_layout import layout_element_order


def two_columns_around(banner: tuple[float, float, float, float] | None) -> list[tuple]:
    """Two columns of blocks, optionally with a full-width element between them.

    The columns occupy 60-290 and 310-540, leaving a gutter at 290-310.  Boxes
    are listed top-down within each column and the left column first, so the
    identity order is already the reading order.
    """
    boxes: list[tuple[float, float, float, float]] = []
    for top in (700.0, 660.0, 620.0):
        boxes.append((60.0, top - 30.0, 290.0, top))
    for top in (700.0, 660.0, 620.0):
        boxes.append((310.0, top - 30.0, 540.0, top))
    if banner is not None:
        boxes.append(banner)
    return boxes


def test_columns_read_down_before_across_without_a_banner() -> None:
    order = layout_element_order(tuple(two_columns_around(None)))

    assert order == (0, 1, 2, 3, 4, 5)


def test_a_full_width_element_does_not_interleave_the_columns_around_it() -> None:
    # The banner sits below both columns and spans the full text width, so its
    # box lies across the gutter the column split depends on.
    banner = (60.0, 520.0, 540.0, 580.0)
    order = layout_element_order(tuple(two_columns_around(banner)))

    left, right = order.index(0), order.index(3)
    assert order[:3] == (0, 1, 2), "left column must stay together and come first"
    assert order[3:6] == (3, 4, 5), "right column must stay together"
    assert left < right
    # The banner is below both columns, so it reads last.
    assert order[-1] == 6


def test_a_full_width_element_separates_the_columns_above_it_from_those_below() -> None:
    """Text above a banner is read before text below it, column by column."""
    boxes: list[tuple[float, float, float, float]] = []
    for top in (700.0, 660.0):  # 0, 1  upper left column
        boxes.append((60.0, top - 30.0, 290.0, top))
    for top in (700.0, 660.0):  # 2, 3  upper right column
        boxes.append((310.0, top - 30.0, 540.0, top))
    boxes.append((60.0, 560.0, 540.0, 620.0))  # 4  banner across both columns
    for top in (520.0, 480.0):  # 5, 6  lower left column
        boxes.append((60.0, top - 30.0, 290.0, top))
    for top in (520.0, 480.0):  # 7, 8  lower right column
        boxes.append((310.0, top - 30.0, 540.0, top))

    order = layout_element_order(tuple(boxes))

    assert order.index(4) > max(order.index(index) for index in (0, 1, 2, 3))
    assert order.index(4) < min(order.index(index) for index in (5, 6, 7, 8))
    # Within each half the columns stay whole rather than interleaving.
    assert order[:4] == (0, 1, 2, 3)
    assert order[5:] == (5, 6, 7, 8)
