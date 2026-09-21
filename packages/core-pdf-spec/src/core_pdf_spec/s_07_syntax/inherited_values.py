# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from core_pdf_spec.s_07_syntax.resolution import resolve_reference_chain
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
    value = node.get(key)
    resolved = resolve_reference_chain(value, resolve)
    return parent_value if resolved is None else value


def collect_inherited_values(
    node: PdfDict,
    keys: tuple[str, ...],
    resolve_ref: Callable[[object], object],
    *,
    stop_at_malformed_parent: bool = False,
) -> InheritedValueMap:
    values: InheritedValueMap = {}
    current: object = node
    seen: dict[int, PdfDict] = {}
    references: set[tuple[int, int]] = set()
    while current is not None:
        if not isinstance(current, dict):
            if stop_at_malformed_parent:
                break
            raise ValueError("invalid inherited dictionary parent")
        marker = id(current)
        if marker in seen:
            if stop_at_malformed_parent:
                break
            raise ValueError("inherited dictionary cycle detected")
        seen[marker] = cast(PdfDict, current)

        current_dict = cast("PdfDict", current)
        for key in keys:
            if key in values:
                continue
            value = inherited_dictionary_value(current_dict, key, None, resolve_ref)
            if value is not None:
                values[key] = cast(CachedPdfObject, value)

        parent = current_dict.get("Parent")
        if isinstance(parent, PdfReference):
            reference = (parent.object_number, parent.generation_number)
            if reference in references:
                if stop_at_malformed_parent:
                    break
                raise ValueError("inherited dictionary cycle detected")
            references.add(reference)
        current = resolve_ref(parent) if parent is not None else None

    return values


__all__ = (
    "collect_inherited_values",
    "inherited_dictionary_value",
)
