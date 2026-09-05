# SPDX-License-Identifier: AGPL-3.0-only
"""Strict annotation metadata failure does not suppress tolerant appearance capture."""

import pytest

from core_pdf.impl._impl.extract.capture import capture_page
from core_pdf.impl._impl.extract.pipeline import internal_PageExtraction
from tests.helpers.pdf_bytes import one_page_pdf, open_pdf, stream_obj


def appearance_pdf(annotations: bytes) -> bytes:
    return one_page_pdf(
        b"BT /F1 12 Tf 50 700 Td (PageContent) Tj ET",
        page_extra=b"/Annots " + annotations,
        extra_objects=(
            b"<< /Type /Annot /Subtype /FreeText /Rect [100 500 220 520] /AP << /N 7 0 R >> >>",
            stream_obj(
                b"BT /F1 10 Tf 2 4 Td (VisibleAppearance) Tj ET",
                b"/Type /XObject /Subtype /Form /BBox [0 0 120 20] "
                b"/Resources << /Font << /F1 5 0 R >> >>",
            ),
        ),
    )


@pytest.mark.parametrize(
    "malformed",
    [b"null", b"<< /Type /Annot /Subtype /Text /Rect (invalid) >>"],
)
def test_failed_annotation_metadata_keeps_valid_appearance_text(malformed: bytes) -> None:
    pdf = appearance_pdf(b"[" + malformed + b" 6 0 R]")
    with open_pdf(pdf) as document:
        page = document.pages[0]
        with pytest.raises(ValueError, match="invalid (page annotation|box value)"):
            page.get_annotations()

        extraction = internal_PageExtraction(page)

        assert extraction.capture.annotations == ()
        assert "VisibleAppearance" in "".join(extraction.capture.observations.text)
        assert len(extraction.capture.program.appearances) == 1
        # Supplying an empty collection remains an explicit choice. Only a
        # failed metadata read uses the tolerant raw-dictionary fallback.
        suppressed = capture_page(page, fields=(), annotations=())
        assert suppressed.program.appearances == ()
        assert "VisibleAppearance" not in "".join(suppressed.observations.text)

        with pytest.raises(ValueError, match="invalid (page annotation|box value)"):
            page.get_annotations()


def test_empty_recovered_metadata_does_not_suppress_a_non_array_appearance() -> None:
    pdf = appearance_pdf(b"6 0 R").split(b"xref\n", 1)[0]
    pdf += b"trailer << /Root 1 0 R >>\n%%EOF\n"

    with open_pdf(pdf) as document:
        assert document.recovery_enabled
        page = document.pages[0]
        assert page.get_annotations() == []

        extraction = internal_PageExtraction(page)

        assert extraction.capture.annotations == ()
        assert "VisibleAppearance" in "".join(extraction.capture.observations.text)
        assert len(extraction.capture.program.appearances) == 1
