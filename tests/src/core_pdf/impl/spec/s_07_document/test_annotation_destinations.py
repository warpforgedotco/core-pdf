# SPDX-License-Identifier: AGPL-3.0-only
"""Annotation destinations must reach the structured document as values.

A URI action normally stores its target behind an indirect reference.  If that
reference survives into the structured IR, JSON serialization falls back to
``str()`` on the reference object, so the URL is lost and -- before
``PdfReference`` had a ``__repr__`` -- the output carried a live memory address
and differed between runs of the same file.
"""

from __future__ import annotations

import json

from core_pdf.impl.types import PdfReference
from tests.helpers.pdf_bytes import one_page_pdf, open_pdf

URL = b"https://example.invalid/target#frag"


def link_pdf() -> bytes:
    """A link whose /URI is an indirect reference to the string object."""
    content = b"BT /F1 12 Tf 50 700 Td (Body) Tj ET\n"
    return one_page_pdf(
        content,
        page_extra=b"/Annots [6 0 R]",
        extra_objects=[
            b"<< /Type /Annot /Subtype /Link /Rect [50 690 200 710] /A << /S /URI /URI 7 0 R >> >>",
            b"(" + URL + b")",
        ],
    )


def destinations(pdf_bytes: bytes) -> list[object]:
    with open_pdf(pdf_bytes) as document:
        payload = json.loads(document.extract().to_json())
    page_id = payload["pages"][0]["id"]
    return [
        annotation["destination"]
        for annotation in payload["annotations"]
        if annotation["page_id"] == page_id
    ]


def test_indirect_uri_reaches_structured_output_as_the_url() -> None:
    assert destinations(link_pdf()) == [
        {"S": "URI", "URI": URL.decode()},
    ]


def test_structured_output_is_identical_across_runs() -> None:
    pdf_bytes = link_pdf()

    # Distinct PdfReference objects live at distinct addresses, so a repr that
    # leaked one made the same file serialize differently each time.
    assert destinations(pdf_bytes) == destinations(pdf_bytes)


def test_pdf_reference_renders_as_pdf_syntax_not_an_address() -> None:
    reference = PdfReference(7, 0)

    assert str(reference) == "7 0 R"
    assert repr(reference) == "PdfReference(7, 0)"
    assert "object at 0x" not in str(reference)
