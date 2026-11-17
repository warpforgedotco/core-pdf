"""ISO 32000-2 12.7: interactive form fields: Kids and fully qualified field names."""

from __future__ import annotations

from collections.abc import Callable

from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_07_syntax.types import PdfObject

# Q's values (Table 228); 0 is the default.
QUADDING_NAMES = ("left", "center", "right")
# The field types whose text is variable text (12.7.4.3).
VARIABLE_TEXT_FIELD_TYPES = frozenset({"Tx", "Ch"})
MAX_FIELD_DEPTH = 50


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


@cites(
    "ISO 32000-2",
    "12.7.4.3",
    introduced="1.2",
    rule="Q, inheritable from the field tree and the AcroForm dictionary, quads variable text",
)
def field_quadding(node: object, form: object, resolve: Callable[[object], object]) -> str:
    """How a variable text field justifies its text: the Q of the field or of the nearest
    Parent with one, else the interactive form dictionary's, else 0 (Table 228), as "left",
    "center" or "right". A Q other than 0, 1 or 2 is taken as the default."""
    seen: set[int] = set()
    current = resolve(node)
    for _ in range(MAX_FIELD_DEPTH):
        if not isinstance(current, dict) or id(current) in seen:
            break
        seen.add(id(current))
        value = current.get("Q")
        if value is not None:
            return quadding_name(resolve(value))
        current = resolve(current.get("Parent"))
    form = resolve(form)
    if isinstance(form, dict) and (value := form.get("Q")) is not None:
        return quadding_name(resolve(value))
    return QUADDING_NAMES[0]


def quadding_name(value: object) -> str:
    if type(value) is int and 0 <= value < len(QUADDING_NAMES):
        return QUADDING_NAMES[value]
    return QUADDING_NAMES[0]


__all__ = (
    "QUADDING_NAMES",
    "VARIABLE_TEXT_FIELD_TYPES",
    "field_children",
    "field_quadding",
    "qualified_field_name",
)
