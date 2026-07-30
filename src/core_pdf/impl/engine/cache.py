# SPDX-License-Identifier: AGPL-3.0-only
"""Small per-page caches shared by the PDF and parse layers."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TypeAlias, TypeVar

CacheKey: TypeAlias = str | tuple[Hashable, ...]
internal_T = TypeVar("internal_T")


class ExtractionCache(dict[CacheKey, object]):
    """A page-owned cache; synchronization is provided by the page lock."""

    def get_as(self, key: CacheKey, expected_type: type[internal_T]) -> internal_T | None:
        value = self.get(key)
        return value if isinstance(value, expected_type) else None


__all__ = ("CacheKey", "ExtractionCache")
