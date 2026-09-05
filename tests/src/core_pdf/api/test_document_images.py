# SPDX-License-Identifier: AGPL-3.0-only

from tests.helpers.pdf_bytes import one_page_pdf, open_pdf


def test_inline_stencil_image_record_retains_captured_paint() -> None:
    pdf = one_page_pdf(
        b"0 0 1 rg /GS1 gs BI /W 1 /H 1 /IM true ID \x00 EI",
        resources=b"<< /ExtGState << /GS1 << /ca 0.5 >> >> >>",
    )
    with open_pdf(pdf) as document:
        images = document.pages[0].extract_images()

    assert len(images) == 1
    assert images[0].kind == "inline-image"
    assert images[0].fill == (0.0, 0.0, 1.0)
    assert images[0].fill_opacity == 0.5
