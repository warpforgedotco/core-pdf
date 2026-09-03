# SPDX-License-Identifier: AGPL-3.0-only
"""Text outside the active clip must not reach extraction.

A form XObject's /BBox and the page box both bound what a reader sees, and
documents lean on that: figures are routinely whole imported pages that are
cropped down to one illustration, with the rest of the source page still in
the stream behind the clip.
"""

from __future__ import annotations

from tests.helpers.pdf_bytes import (
    assemble_pdf,
    first_page_runs,
    one_page_pdf,
    open_pdf,
    stream_obj,
)


def form_xobject_pdf() -> bytes:
    """A form whose /BBox admits only the first of two text lines."""
    form = (
        b"BT /F1 12 Tf 10 60 Td (InsideTheBox) Tj ET\n"
        b"BT /F1 12 Tf 10 200 Td (OutsideTheBox) Tj ET\n"
    )
    content = b"q 1 0 0 1 100 100 cm /Fm1 Do Q\n"
    return assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /XObject << /Fm1 5 0 R >> >> /Contents 4 0 R >>",
            stream_obj(content),
            stream_obj(
                form,
                b"/Type /XObject /Subtype /Form /FormType 1 /BBox [0 0 200 100] "
                b"/Resources << /Font << /F1 6 0 R >> >>",
            ),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


def off_page_pdf() -> bytes:
    """Text placed above the top edge of the page box."""
    content = (
        b"BT /F1 12 Tf 50 400 Td (OnThePage) Tj ET\n"
        b"BT /F1 12 Tf 50 1400 Td (AbovePageTop) Tj ET\n"
        b"BT /F1 12 Tf 50 -300 Td (BelowPageBottom) Tj ET\n"
    )
    return one_page_pdf(content)


def clip_path_pdf() -> bytes:
    """A `W n` clip path admitting only the lower line."""
    content = (
        b"q 0 0 300 100 re W n\n"
        b"BT /F1 12 Tf 20 40 Td (InsideClipPath) Tj ET\n"
        b"BT /F1 12 Tf 20 500 Td (OutsideClipPath) Tj ET\n"
        b"Q\n"
        b"BT /F1 12 Tf 20 600 Td (AfterClipRestored) Tj ET\n"
    )
    return one_page_pdf(content)


def cropbox_pdf() -> bytes:
    """A CropBox narrower than the MediaBox."""
    content = (
        b"BT /F1 12 Tf 50 200 Td (InsideCropBox) Tj ET\n"
        b"BT /F1 12 Tf 50 700 Td (OutsideCropBox) Tj ET\n"
    )
    return one_page_pdf(content, page_extra=b"/CropBox [0 0 612 400]")


def page_text(data: bytes) -> str:
    return "".join(run.text for run in first_page_runs(data))


def test_form_xobject_bbox_clips_the_text_it_excludes() -> None:
    text = page_text(form_xobject_pdf())
    assert "InsideTheBox" in text
    assert "OutsideTheBox" not in text


def test_text_outside_the_page_box_is_discarded() -> None:
    text = page_text(off_page_pdf())
    assert "OnThePage" in text
    assert "AbovePageTop" not in text
    assert "BelowPageBottom" not in text


def test_clip_path_bounds_text_and_is_undone_by_restore() -> None:
    text = page_text(clip_path_pdf())
    assert "InsideClipPath" in text
    assert "OutsideClipPath" not in text
    # Q restores the previous clip, so later text is unaffected.
    assert "AfterClipRestored" in text


def test_cropbox_narrows_the_page_clip() -> None:
    text = page_text(cropbox_pdf())
    assert "InsideCropBox" in text
    assert "OutsideCropBox" not in text


def test_effective_page_clip_intersects_cropbox_with_mediabox() -> None:
    with open_pdf(cropbox_pdf()) as document:
        assert document.pages[0].effective_page_clip() == (0.0, 0.0, 612.0, 400.0)
    with open_pdf(off_page_pdf()) as document:
        assert document.pages[0].effective_page_clip() == (0.0, 0.0, 612.0, 792.0)
