"""AcroForm field names, inheritable entries, and child-array semantics."""

from __future__ import annotations

from typing import cast

from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfObject


def qualified_field_name(parent: str, partial_name: str | None) -> str:
    if parent and partial_name:
        return f"{parent}.{partial_name}"
    return partial_name or parent


def inherited_field_value(node: PdfDict, key: str, parent_value: PdfObject) -> PdfObject:
    """A null entry is equivalent to an absent dictionary entry in PDF."""
    value = node.get(key)
    return parent_value if value is None else value


def field_children(value: object) -> list[PdfObject]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("invalid AcroForm Kids array")
    return cast(list[PdfObject], value)
