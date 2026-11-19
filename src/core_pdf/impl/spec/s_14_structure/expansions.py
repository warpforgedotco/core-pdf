"""ISO 32000-2 14.9.5: the expansion text of abbreviations and acronyms."""

from __future__ import annotations

from collections.abc import Callable

from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_07_syntax.types import PdfDict


@cites(
    "ISO 32000-2",
    "14.9.5",
    introduced="1.5",
    rule="E of a structure element or Span property list expands the abbreviation it holds",
)
def expansion_text(props: PdfDict, resolve_str: Callable[[object], str | None]) -> str | None:
    """The E entry of a structure element or of a Span marked-content property list: the
    expanded form of the abbreviation or acronym its content shows, which, unlike
    ActualText, does not replace that content."""
    value = props.get("E")
    return None if value is None else resolve_str(value)
