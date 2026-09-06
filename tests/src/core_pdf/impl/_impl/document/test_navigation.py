# SPDX-License-Identifier: AGPL-3.0-only
"""Outline and named-destination resolution.

Both defects here came from real specification PDFs: outline items whose /A
action is an indirect object, and name trees carrying a handful of
destinations that point at pages the document no longer contains.
"""

from __future__ import annotations

from tests.helpers.pdf_bytes import assemble_pdf, open_pdf


def two_page_objects() -> list[bytes]:
    return [
        b"<< /Type /Catalog /Pages 2 0 R /Outlines 6 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Length 0 >>\nstream\n\nendstream",
    ]


def outline_with_indirect_action_pdf() -> bytes:
    """An outline whose second item reaches its page through an indirect /A."""
    return assemble_pdf(
        [
            *two_page_objects(),
            b"<< /Type /Outlines /First 7 0 R /Last 8 0 R /Count 2 >>",
            b"<< /Title (Direct) /Parent 6 0 R /Next 8 0 R /Dest [3 0 R /XYZ null null null] >>",
            b"<< /Title (ViaAction) /Parent 6 0 R /Prev 7 0 R /A 9 0 R >>",
            b"<< /S /GoTo /D [4 0 R /XYZ null null null] >>",
        ]
    )


def dangling_destination_pdf() -> bytes:
    """A name tree where one destination points at a missing page object."""
    return assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R /Names 6 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>",
            b"<< /Length 0 >>\nstream\n\nendstream",
            b"<< /Dests 7 0 R >>",
            b"<< /Names [(broken) [null /XYZ 54 610 null] "
            b"(good) [4 0 R /XYZ null null null] "
            b"(alsogood) [3 0 R /XYZ null null null]] >>",
        ]
    )


def test_outline_action_given_as_an_indirect_reference_resolves() -> None:
    with open_pdf(outline_with_indirect_action_pdf()) as document:
        items = document.iter_outlines()
        assert [item.title for item in items] == ["Direct", "ViaAction"]
        # The /A action is an indirect object; without resolving it first the
        # destination was dropped and the item reported no page at all.
        assert [item.page_index for item in items] == [0, 1]


def test_one_dangling_destination_does_not_discard_the_name_tree() -> None:
    with open_pdf(dangling_destination_pdf()) as document:
        dests = document.named_destinations()
        # The sound entries survive alongside the broken one.
        assert dests["good"].page_index == 1
        assert dests["alsogood"].page_index == 0
        assert dests["broken"].page_index is None
