from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core_pdf import PdfDocument
from core_pdf.impl.types import PdfReference


class _Cycle(Exception):
    pass


def metadata_retyped_objects(document: PdfDocument) -> frozenset[int]:
    """Return the objects whose /Type pdfplumber's metadata decoding turns into a str.

    pdfplumber decodes each /Info value with resolve_and_decode, which resolves references
    and writes every decoded value back into the dictionary it came from. pdfminer caches
    indirect objects, so the dictionaries it reaches through /Info are the ones the page
    tree reads: a /Type it rewrote is a str, no longer the name create_pages looks for, so
    pdfminer skips that page or page tree node. A reference cycle recurses until
    RecursionError, which pdfplumber logs and moves on from, with the keys before the cycle
    already rewritten.
    """
    resolver = document.resolver
    retyped: set[int] = set()
    active: set[int] = set()

    def decode(value: object) -> None:
        if isinstance(value, PdfReference):
            try:
                resolved = resolver.resolve(value)
            except Exception:
                return
            if isinstance(resolved, dict):
                decode_indirect(value.object_number, resolved)
            elif isinstance(resolved, list):
                decode(resolved)
        elif isinstance(value, list):
            for item in value:
                decode(item)
        elif isinstance(value, dict):
            for item in value.values():
                decode(item)

    def decode_indirect(object_number: int, dictionary: Mapping[Any, object]) -> None:
        if object_number in active:
            raise _Cycle
        active.add(object_number)
        try:
            for key, item in dictionary.items():
                decode(item)
                if key == "Type":
                    retyped.add(object_number)
        finally:
            active.discard(object_number)

    info = document.trailer_dict.get("Info")
    try:
        entries = resolver.resolve(info)
    except Exception:
        return frozenset()
    if not isinstance(entries, dict):
        return frozenset()
    for value in entries.values():
        try:
            decode(value)
        except _Cycle:
            continue
    return frozenset(retyped)
