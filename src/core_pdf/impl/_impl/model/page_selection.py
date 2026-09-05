# SPDX-License-Identifier: AGPL-3.0-only
"""Page-selection values and normalization shared by PDF and structured documents."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

PageSelection: TypeAlias = int | str | range | Sequence[int]


def resolve_page_selection(pages: PageSelection | None, page_count: int) -> list[int]:
    """Return normalized, deduplicated 0-based indexes for a 1-based page selection."""
    segments: list[int | range]
    match pages:
        case None:
            segments = [range(1, page_count + 1)]
        case int() if type(pages) is int:
            segments = [pages]
        case range() as page_range:
            segments = [page_range]
        case str() as page_spec:
            segments = []
            for part in page_spec.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    parts = part.split("-", 1)
                    if not parts[0].strip() or not parts[1].strip():
                        raise ValueError(f"invalid page selection: {pages!r}")
                    try:
                        start = int(parts[0])
                        end = int(parts[1])
                    except ValueError as exc:
                        raise ValueError(f"invalid page selection: {pages!r}") from exc
                    step = 1 if end >= start else -1
                    segments.append(range(start, end + step, step))
                else:
                    try:
                        segments.append(int(part))
                    except ValueError as exc:
                        raise ValueError(f"invalid page selection: {pages!r}") from exc
        case bytes() | bytearray() | memoryview():
            raise TypeError(f"invalid page selection: {pages!r}")
        case Sequence() as page_sequence:
            segments = []
            for page_number in page_sequence:
                if type(page_number) is not int:
                    raise ValueError(f"invalid page selection: {pages!r}")
                segments.append(page_number)
        case _:
            raise TypeError(f"invalid page selection: {pages!r}")

    # Finish syntax/type validation before bounds checks, and validate every
    # compact segment before allocating any expanded range.
    has_pages = False
    for segment in segments:
        if isinstance(segment, range):
            if not segment:
                continue
            first, last, step = segment.start, segment[-1], segment.step
        else:
            first = last = segment
            step = 1
        has_pages = True
        if first < 1 or first > page_count:
            invalid_page = first
        elif last > page_count:
            invalid_page = first + ((page_count - first) // step + 1) * step
        elif last < 1:
            invalid_page = first + ((first - 1) // -step + 1) * step
        else:
            continue
        raise IndexError(f"page selection out of range: {invalid_page}")
    if not has_pages:
        raise ValueError(f"invalid page selection: {pages!r}")

    normalized: list[int] = []
    seen: set[int] = set()
    for segment in segments:
        for page_number in segment if isinstance(segment, range) else (segment,):
            page_index = page_number - 1
            if page_index not in seen:
                normalized.append(page_index)
                seen.add(page_index)
    return normalized


__all__ = ("PageSelection", "resolve_page_selection")
