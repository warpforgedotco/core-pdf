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


def inherited_dictionary_values(node: PdfDict, keys: tuple[str, ...]) -> InheritedValueMap:
    """Collect present inheritable entries, retaining reference identity."""
    return {key: cast(CachedPdfObject, node[key]) for key in keys if node.get(key) is not None}


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
        for key, value in inherited_dictionary_values(current_dict, keys).items():
            values.setdefault(key, value)

        parent = current_dict.get("Parent")
        current = resolve_ref(parent) if parent is not None else None

    return values


__all__ = (
    "collect_inherited_values",
    "inherited_dictionary_values",
)
