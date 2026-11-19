"""ISO 32000-2 12.7: interactive form fields: Kids and fully qualified field names."""

from __future__ import annotations

from collections.abc import Callable

from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_07_syntax.types import PdfObject

# Q's values (Table 228); 0 is the default.
QUADDING_NAMES = ("left", "center", "right")
# The field types whose text is variable text (12.7.4.3).
VARIABLE_TEXT_FIELD_TYPES = frozenset({"Tx", "Ch"})


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
def field_quadding(quadding: object, form: object, resolve: Callable[[object], object]) -> str:
    """How a variable text field justifies its text, as "left", "center" or "right": quadding,
    the Q of the field or of its nearest ancestor field, which the field tree's traversal
    hands down as it does FT and V; else the interactive form dictionary's Q; else 0
    (Table 228). A Q other than 0, 1 or 2 is taken as the default."""
    if quadding is None:
        form = resolve(form)
        if isinstance(form, dict):
            quadding = form.get("Q")
    return QUADDING_NAMES[0] if quadding is None else quadding_name(resolve(quadding))


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
