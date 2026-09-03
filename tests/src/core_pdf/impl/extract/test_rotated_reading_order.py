# SPDX-License-Identifier: AGPL-3.0-only
"""Reading order is decided in the frame the page is displayed in.

/Rotate is not decoration: a page carrying 180 stores its first line at the
bottom of unrotated space. Ordering there walks the page backwards, and a real
document came out reading its outline items l, k, j, i, h.
"""

from __future__ import annotations

import pytest

from tests.helpers.pdf_bytes import first_page_text, one_page_pdf


def rotated_pdf(rotate: int) -> bytes:
    """Three lines whose display order depends on the page rotation.

    In unrotated space ALPHA sits at the top and CHARLIE at the bottom, so an
    unrotated page reads ALPHA, BRAVO, CHARLIE and a page turned 180 reads
    them the other way about.
    """
    content = (
        b"BT /F1 24 Tf 100 700 Td (ALPHA) Tj ET\n"
        b"BT /F1 24 Tf 100 400 Td (BRAVO) Tj ET\n"
        b"BT /F1 24 Tf 100 100 Td (CHARLIE) Tj ET\n"
    )
    return one_page_pdf(content, page_extra=b"/Rotate %d" % rotate)


def order_of(data: bytes) -> list[str]:
    text = first_page_text(data)
    return [word for word in text.split() if word in {"ALPHA", "BRAVO", "CHARLIE"}]


def test_unrotated_page_reads_down_the_page() -> None:
    assert order_of(rotated_pdf(0)) == ["ALPHA", "BRAVO", "CHARLIE"]


def test_page_rotated_180_reads_in_display_order() -> None:
    # Turned upside down, the line stored lowest is the one a reader sees first.
    assert order_of(rotated_pdf(180)) == ["CHARLIE", "BRAVO", "ALPHA"]


@pytest.mark.parametrize("rotate", [90, 270])
def test_every_rotation_still_reports_all_the_text(rotate: int) -> None:
    assert sorted(order_of(rotated_pdf(rotate))) == ["ALPHA", "BRAVO", "CHARLIE"]
