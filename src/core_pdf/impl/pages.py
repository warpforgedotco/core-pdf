# SPDX-License-Identifier: AGPL-3.0-only
"""Shared page-selection resolution used across the engine and the structured IR."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias, cast

PageSelection: TypeAlias = int | str | range | Sequence[int]


def resolve_page_selection(pages: PageSelection | None, page_count: int) -> list[int]:
    """Return normalized, deduplicated 0-based indexes for a 1-based page selection."""
    match pages:
        case None:
            selected = list(range(page_count))
        case int() if type(pages) is int:
            selected = [pages - 1]
        case range() as page_range:
            selected = [page_number - 1 for page_number in page_range]
        case str() as page_spec:
            selected = []
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
                    selected.extend(range(start - 1, end - 1 + step, step))
                else:
                    try:
                        selected.append(int(part) - 1)
                    except ValueError as exc:
                        raise ValueError(f"invalid page selection: {pages!r}") from exc
        case Sequence() as page_sequence:
            try:
                selected = [int(cast(Any, page_number)) - 1 for page_number in page_sequence]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid page selection: {pages!r}") from exc
        case _:
            raise TypeError(f"invalid page selection: {pages!r}")

    if not selected:
        raise ValueError(f"invalid page selection: {pages!r}")

    normalized: list[int] = []
    seen: set[int] = set()
    for page_index in selected:
        if page_index < 0 or page_index >= page_count:
            raise IndexError(f"page selection out of range: {page_index + 1}")
        if page_index not in seen:
            normalized.append(page_index)
            seen.add(page_index)
    return normalized


__all__ = ("PageSelection", "resolve_page_selection")
