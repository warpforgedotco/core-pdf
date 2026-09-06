"""Role-map and revisioned attribute dictionary semantics from PDF chapter 14."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import cast

from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict


def parse_role_map(
    entries: Iterable[tuple[str, object]], resolve_name: Callable[[object], str | None]
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name, value in entries:
        mapped = resolve_name(value)
        if mapped is None:
            raise ValueError("invalid role map entry")
        mapping[name] = mapped
    return mapping


@dataclass(frozen=True, slots=True)
class StructureAttribute:
    value: PdfDict | PdfStream
    revision: int
    explicit_revision: bool


def attribute_entries(
    entries: Sequence[object],
    *,
    resolve: Callable[[object], object],
    resolve_revision: Callable[[object], int | None],
) -> Iterator[StructureAttribute]:
    """Yield all attribute objects and their optional revision numbers in order.

    Revision selection and conversion to an output dictionary belong to the
    consumer. An omitted revision number denotes revision zero.
    """
    index = 0
    while index < len(entries):
        attribute = resolve(entries[index])
        if not isinstance(attribute, (dict, PdfStream)):
            raise ValueError("invalid structure attribute entry")
        index += 1
        revision = 0
        explicit_revision = False
        if index < len(entries):
            candidate = resolve(entries[index])
            if not isinstance(candidate, (dict, PdfStream)):
                parsed = resolve_revision(candidate)
                if parsed is None:
                    raise ValueError("invalid structure attribute revision")
                revision = parsed
                explicit_revision = True
                index += 1
        yield StructureAttribute(cast(PdfDict | PdfStream, attribute), revision, explicit_revision)


def marked_content_id(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("invalid structure content mcid")
    return value
