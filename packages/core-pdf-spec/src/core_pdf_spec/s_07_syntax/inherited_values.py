# SPDX-License-Identifier: AGPL-3.0-only
"""Inherited PDF dictionary value collection."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from core_pdf_spec.s_07_syntax.types import (
    CachedPdfObject,
    InheritedValueMap,
    PdfDict,
)
from core_pdf_spec.types import PdfReference


def inherited_dictionary_value(
    node: PdfDict,
    key: str,
    parent_value: object,
    resolve: Callable[[object], object],
) -> object:
    """Select a non-null entry while retaining its original reference identity.

    ISO 32000-1, 7.3.9–7.3.10: null dictionary entries and references to
    nonexistent objects have the same effect as an omitted entry. Inspect
    only the selected scalar chain; a resource dictionary remains shallow.
    """
    value = node.get(key)
    resolved: object = value
    seen: set[tuple[int, int]] = set()
    while isinstance(resolved, PdfReference):
        marker = (resolved.object_number, resolved.generation_number)
        if marker in seen:
            break
        seen.add(marker)
        resolved = resolve(resolved)
    return parent_value if resolved is None else value


def collect_inherited_values(
    node: PdfDict,
    keys: tuple[str, ...],
    resolve_ref: Callable[[object], object],
) -> InheritedValueMap:
    values: InheritedValueMap = {}
    current: object = node
    seen: set[int] = set()
    while current is not None:
        if not isinstance(current, dict):
            raise ValueError("invalid inherited dictionary parent")
        marker = id(current)
        if marker in seen:
            raise ValueError("inherited dictionary cycle detected")
        seen.add(marker)

        current_dict = cast("PdfDict", current)
        for key in keys:
            if key in values:
                continue
            value = inherited_dictionary_value(current_dict, key, None, resolve_ref)
            if value is not None:
                values[key] = cast(CachedPdfObject, value)

        parent = current_dict.get("Parent")
        current = resolve_ref(parent) if parent is not None else None

    return values


__all__ = (
    "collect_inherited_values",
    "inherited_dictionary_value",
)
