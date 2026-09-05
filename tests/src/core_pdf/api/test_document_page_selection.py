# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import cast

import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.model.page_selection import PageSelection
from tests.helpers.pdf_bytes import HELVETICA, assemble_pdf, stream_obj


def internal_selection_pdf() -> bytes:
    # Verified before pinning selection expectations with Poppler 26.07.0:
    # pdfinfo reports three pages and AcroForm; pdftotext -f N -l N reports
    # PageOne/Value1, PageTwo/Value2, and PageThree/Value3 for pages 1, 2, 3.
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [6 0 R 10 0 R 14 0 R] "
        b"/DA (/F1 12 Tf 0 g) /DR << /Font << /F1 3 0 R >> >> >> >>",
        b"<< /Type /Pages /Count 3 /Kids [4 0 R 8 0 R 12 0 R] >>",
        HELVETICA,
    ]
    for index, label in enumerate(("PageOne", "PageTwo", "PageThree")):
        page = 4 + index * 4
        objects.extend(
            [
                (
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
                    "/Resources << /Font << /F1 3 0 R >> "
                    f"/XObject << /Image {page + 3} 0 R >> >> "
                    f"/Contents {page + 1} 0 R /Annots [{page + 2} 0 R] >>"
                ).encode(),
                stream_obj(
                    f"BT /F1 12 Tf 20 150 Td ({label}) Tj ET "
                    "q 10 0 0 10 20 60 cm /Image Do Q".encode()
                ),
                (
                    "<< /Type /Annot /Subtype /Widget /FT /Tx "
                    f"/T (Field{index + 1}) /V (Value{index + 1}) "
                    f"/Rect [10 10 80 25] /P {page} 0 R >>"
                ).encode(),
                stream_obj(
                    bytes((index * 80, 100, 200)),
                    b"/Type /XObject /Subtype /Image /Width 1 /Height 1 "
                    b"/ColorSpace /DeviceRGB /BitsPerComponent 8",
                ),
            ]
        )
    return assemble_pdf(objects)


@pytest.mark.parametrize(
    ("selection", "expected_pages"),
    [
        (None, [1, 2, 3]),
        ([3, 1, 3, 2], [3, 1, 2]),
        ("3-1,2", [3, 2, 1]),
        (range(3, 0, -1), [3, 2, 1]),
        ((2, 1), [2, 1]),
    ],
)
def test_document_operations_share_page_selection_order(
    selection: PageSelection | None, expected_pages: list[int]
) -> None:
    with PdfDocument(internal_selection_pdf()) as document:
        extracted = document.extract(pages=selection)
        fields = document.extract_form_fields(pages=selection)
        images = document.extract_images(pages=selection)

    assert [page.page_number for page in extracted.pages] == expected_pages
    assert [page.text for page in extracted.pages] == [
        ("PageOne", "PageTwo", "PageThree")[number - 1] for number in expected_pages
    ]
    assert [field.page_number for field in fields] == expected_pages
    assert [field.record.value_text for field in fields] == [
        f"Value{number}" for number in expected_pages
    ]
    assert [image.page_number for image in images] == expected_pages


@pytest.mark.parametrize("method", ["extract", "extract_form_fields", "extract_images"])
@pytest.mark.parametrize(
    ("selection", "error"),
    [
        ([1.9], ValueError),
        ([True], ValueError),
        (["1"], ValueError),
        (b"\x01", TypeError),
        (range(1, 10**100), IndexError),
    ],
)
def test_document_operations_reject_invalid_page_selections(
    method: str, selection: object, error: type[Exception]
) -> None:
    with PdfDocument(internal_selection_pdf()) as document:
        with pytest.raises(error, match="page selection"):
            getattr(document, method)(pages=cast(PageSelection, selection))
