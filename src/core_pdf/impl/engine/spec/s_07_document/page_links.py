# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Protocol

from core_pdf.impl.engine.spec.s_07_objects.coercion import parse_float_strict
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_09_fonts.encoding import decode_pdf_text_string
from core_pdf.impl.objects import MISSING, PdfName, PdfString
from core_pdf.impl.types import PdfDict


PDFKEY_SUBTYPE = PdfName.of("Subtype")
PDFKEY_ANNOTS = PdfName.of("Annots")
PDFKEY_RECT = PdfName.of("Rect")
PDFKEY_A = PdfName.of("A")
PDFKEY_S = PdfName.of("S")
PDFKEY_URI = PdfName.of("URI")
PDFKEY_D = PdfName.of("D")


class LinkResolver(Protocol):
    def resolve_str(self, value: object) -> str | None: ...


def lookup_pdf_key(value: object, key: str, pdf_key: PdfName) -> object:
    if not isinstance(value, dict):
        return None
    found = value.get(pdf_key, MISSING)
    if found is not MISSING:
        return found
    found = value.get(key, MISSING)
    if found is not MISSING:
        return found
    return lookup_dict_key(value, key)


def pdf_name_direct(value: object) -> str | None:
    if isinstance(value, PdfName):
        return value.value
    if isinstance(value, str):
        return value.lstrip("/")
    return None


def pdf_box_direct(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return (
            parse_float_strict(value[0]),
            parse_float_strict(value[1]),
            parse_float_strict(value[2]),
            parse_float_strict(value[3]),
        )
    except ValueError:
        return None


def pdf_string_direct(value: object) -> str | None:
    if isinstance(value, PdfString):
        return decode_pdf_text_string(value.data)
    if isinstance(value, bytes):
        return decode_pdf_text_string(value)
    if isinstance(value, str):
        return value
    return None


def link_target_direct(action: PdfDict, link_type: str | None) -> str | None:
    if link_type == "URI":
        return pdf_string_direct(lookup_pdf_key(action, "URI", PDFKEY_URI))
    if link_type == "GoTo":
        return pdf_string_direct(lookup_pdf_key(action, "D", PDFKEY_D))
    return None


def link_target_resolved(
    resolver: LinkResolver, action: PdfDict, link_type: str | None
) -> str | None:
    key = "URI" if link_type == "URI" else "D" if link_type == "GoTo" else None
    pdf_key = PDFKEY_URI if key == "URI" else PDFKEY_D if key == "D" else None
    if key is None or pdf_key is None:
        return None
    return resolver.resolve_str(lookup_pdf_key(action, key, pdf_key))


__all__ = (
    "PDFKEY_A",
    "PDFKEY_ANNOTS",
    "PDFKEY_RECT",
    "PDFKEY_S",
    "PDFKEY_SUBTYPE",
    "link_target_direct",
    "link_target_resolved",
    "lookup_pdf_key",
    "pdf_box_direct",
    "pdf_name_direct",
)
