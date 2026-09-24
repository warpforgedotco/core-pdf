from __future__ import annotations

from core_pdf_spec.s_07_syntax.types import PdfObject


def qualified_field_name(parent: str, partial_name: str | None) -> str:
    if parent and partial_name:
        return f"{parent}.{partial_name}"
    return partial_name or parent


def field_children(value: object) -> list[PdfObject]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("invalid AcroForm Kids array")
    return value


__all__ = (
    "field_children",
    "qualified_field_name",
)
