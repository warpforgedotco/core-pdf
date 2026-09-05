# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import cast

import pytest

from core_pdf.impl._impl.model.page_selection import PageSelection, resolve_page_selection


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (None, [0, 1, 2, 3, 4]),
        (3, [2]),
        (range(1, 6, 2), [0, 2, 4]),
        (range(5, 0, -2), [4, 2, 0]),
        (range(1, 7, 2), [0, 2, 4]),
        (range(5, -1, -3), [4, 1]),
        (" 4, 2-4, 2,, 1 ", [3, 1, 2, 0]),
        ("5-3", [4, 3, 2]),
        ([3, 1, 3, 2], [2, 0, 1]),
        ((2, 2, 1), [1, 0]),
    ],
)
def test_selection_preserves_order_and_deduplicates_zero_based_indexes(
    selection: PageSelection | None, expected: list[int]
) -> None:
    assert resolve_page_selection(selection, 5) == expected


@pytest.mark.parametrize(
    ("selection", "page_count"),
    [(None, 0), ("", 5), (", , ", 5), ([], 5), (range(2, 2), 5)],
)
def test_empty_selection_is_rejected(selection: PageSelection | None, page_count: int) -> None:
    with pytest.raises(ValueError, match="invalid page selection"):
        resolve_page_selection(selection, page_count)


@pytest.mark.parametrize("selection", ["-2", "2-", "a", "1-a", "1-2-3"])
def test_malformed_selection_is_rejected(selection: str) -> None:
    with pytest.raises(ValueError, match="invalid page selection"):
        resolve_page_selection(selection, 5)


@pytest.mark.parametrize("selection", [0, -1, 6, "1,6", [1, 0]])
def test_out_of_range_selection_is_rejected(selection: PageSelection) -> None:
    with pytest.raises(IndexError, match="page selection out of range"):
        resolve_page_selection(selection, 5)


@pytest.mark.parametrize(
    "selection",
    [True, 1.5, {1: 2}, object(), b"\x01", bytearray(b"\x01"), memoryview(b"\x01")],
)
def test_unsupported_selection_type_is_rejected(selection: object) -> None:
    with pytest.raises(TypeError, match="invalid page selection"):
        resolve_page_selection(cast(PageSelection, selection), 5)


@pytest.mark.parametrize(
    "selection", [["invalid"], [None], ["1"], [1.9], [True], [1, False], [float("inf")]]
)
def test_invalid_sequence_item_is_rejected(selection: object) -> None:
    with pytest.raises(ValueError, match="invalid page selection"):
        resolve_page_selection(cast(PageSelection, selection), 5)


@pytest.mark.parametrize("selection", ["99,invalid", "0,1-", [99, "1"]])
def test_invalid_syntax_or_sequence_type_precedes_bounds_errors(selection: object) -> None:
    with pytest.raises(ValueError, match="invalid page selection"):
        resolve_page_selection(cast(PageSelection, selection), 5)


@pytest.mark.parametrize(
    ("selection", "invalid_page"),
    [
        (range(1, 10**100), 6),
        (range(5, 10**100, 3), 8),
        (range(5, -(10**100), -2), -1),
        (range(3, -(10**100), -3), 0),
        (f"1-{10**100}", 6),
        (f"{10**100}-1", 10**100),
        ("5-0", 0),
    ],
)
def test_range_bounds_report_first_invalid_page_without_expanding(
    selection: PageSelection, invalid_page: int
) -> None:
    # Range objects and their string forms remain compact; materializing these
    # invalid selections would be impossible.
    with pytest.raises(IndexError, match=f"^page selection out of range: {invalid_page}$"):
        resolve_page_selection(selection, 5)


def test_all_segment_bounds_are_checked_before_expanding_valid_ranges() -> None:
    page_count = 10**100
    with pytest.raises(IndexError, match="^page selection out of range: 0$"):
        resolve_page_selection(f"1-{page_count},0", page_count)


def test_syntax_errors_after_large_ranges_precede_bounds_errors() -> None:
    with pytest.raises(ValueError, match="invalid page selection"):
        resolve_page_selection(f"1-{10**100},invalid", 5)
