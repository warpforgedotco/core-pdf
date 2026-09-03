# SPDX-License-Identifier: AGPL-3.0-only
"""Inherited PDF dictionary value collection."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from core_pdf.impl.spec.s_07_syntax.types import (
    CachedPdfObject,
    InheritedValueMap,
    PdfDict,
)


def collect_inherited_values(
    node: PdfDict,
    keys: tuple[str, ...],
    resolve_ref: Callable[[object], object],
) -> InheritedValueMap:
    values: InheritedValueMap = {}
    current: object = node
    seen: set[int] = set()
    while isinstance(current, dict):
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)

        current_dict = cast("PdfDict", current)
        for key in keys:
            if key not in values:
                inherited_value = current_dict.get(key)
                if inherited_value is not None:
                    values[key] = cast("CachedPdfObject", inherited_value)

        parent = current_dict.get("Parent")
        current = resolve_ref(parent) if parent is not None else None

    return values


__all__ = ("collect_inherited_values",)
