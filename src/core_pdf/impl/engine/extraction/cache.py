# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Hashable, MutableMapping
from typing import TypeAlias, TypeVar

ExtractionCacheKey: TypeAlias = str | tuple[Hashable, ...]
ExtractionCacheMapping: TypeAlias = MutableMapping[ExtractionCacheKey, object]

_T = TypeVar("_T")


class ExtractionCache(dict[ExtractionCacheKey, object]):
    def get_as(self, key: ExtractionCacheKey, expected_type: type[_T]) -> _T | None:
        value = self.get(key)
        return value if isinstance(value, expected_type) else None
