# SPDX-License-Identifier: AGPL-3.0-only
"""Compiled PDF dictionary lookup and inheritance kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_objects.object_cache import (
        CachedPdfObject,
        InheritedValueMap,
        InheritedValuesCache,
    )
    from core_pdf.impl.types import PdfDict


def lookup_dict_key_default(value: object, key: str, default: object = None) -> object:
    if not isinstance(value, dict):
        return default
    return value.get(key, default)


def lookup_dict_key(value: object, key: str) -> object:
    return lookup_dict_key_default(value, key, None)


def collect_inherited_values(
    node: PdfDict,
    keys: tuple[str, ...],
    resolve_ref: Callable[[object], object],
    cache: InheritedValuesCache | None = None,
) -> InheritedValueMap:
    values: InheritedValueMap = {}
    current: object = node
    ancestors: list[tuple[int, PdfDict]] = []
    cached_values: InheritedValueMap | None = None
    seen: set[int] = set()
    while isinstance(current, dict):
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)
        if cache is not None:
            cached_values = cache.get(marker)
            if cached_values is not None:
                for key, value in cached_values.items():
                    if key not in values:
                        values[key] = value
                break

        current_dict = cast("PdfDict", current)
        ancestors.append((marker, current_dict))
        for key in keys:
            if key not in values:
                inherited_value = lookup_dict_key(current_dict, key)
                if inherited_value is not None:
                    values[key] = cast("CachedPdfObject", inherited_value)

        parent = lookup_dict_key(current, "Parent")
        current = resolve_ref(parent) if parent is not None else None
        if not isinstance(current, dict):
            current = None

    if cache is not None:
        running_values: InheritedValueMap = cached_values if cached_values is not None else {}
        for marker, node_dict in reversed(ancestors):
            merged: InheritedValueMap = running_values.copy()
            for key in keys:
                inherited_value = lookup_dict_key(node_dict, key)
                if inherited_value is not None:
                    merged[key] = cast("CachedPdfObject", inherited_value)
            cache[marker] = merged
            running_values = merged

    return values


__all__ = (
    "collect_inherited_values",
    "lookup_dict_key",
    "lookup_dict_key_default",
)
