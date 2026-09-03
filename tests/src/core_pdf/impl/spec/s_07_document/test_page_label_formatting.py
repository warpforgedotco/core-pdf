# SPDX-License-Identifier: AGPL-3.0-only
"""Page label formatting (12.4.2) and the number-tree rules around index 0."""

from __future__ import annotations

import re
from io import BytesIO

import pytest

from core_pdf.impl.spec.s_07_document.document import PdfDocument
from core_pdf.impl.spec.s_07_document.document_labels import (
    format_alpha,
    format_page_label,
)
from tests.helpers.pdf_bytes import HELVETICA, assemble_pdf, stream_obj


@pytest.mark.parametrize(
    ("number", "expected"),
    [(1, "a"), (26, "z"), (27, "aa"), (28, "bb"), (52, "zz"), (53, "aaa")],
)
def test_page_label_alphabetic_sequence(number: int, expected: str) -> None:
    assert format_alpha(number) == expected


def test_page_label_without_style_is_prefix_only() -> None:
    assert format_page_label({"P": b"Appendix-"}, 17, lambda value: value) == "Appendix-"


def four_page_pdf_with_labels_from_page_two() -> bytes:
    """Four pages whose /PageLabels tree starts at index 2, which 12.4.2 forbids."""
    kids = " ".join(f"{4 + index * 2} 0 R" for index in range(4))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R /PageLabels << /Nums [2 << /S /D >>] >> >>",
        f"<< /Type /Pages /Kids [{kids}] /Count 4 >>".encode(),
        HELVETICA,
    ]
    for index in range(4):
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
            + f"{5 + index * 2} 0 R >>".encode()
        )
        objects.append(stream_obj(f"BT /F1 10 Tf 36 750 Td (p{index}) Tj ET".encode()))
    return assemble_pdf(objects)


def test_page_labels_require_page_zero_range() -> None:
    with PdfDocument(BytesIO(four_page_pdf_with_labels_from_page_two())) as document:
        with pytest.raises(ValueError, match="page index 0"):
            _ = document.page_labels


def test_recovered_page_labels_fill_missing_initial_range() -> None:
    # A bad startxref forces xref reconstruction, which switches the document
    # into lenient traversal; the missing initial range is then filled in.
    broken = re.sub(
        rb"startxref\n\d+", b"startxref\n999999", four_page_pdf_with_labels_from_page_two()
    )

    with PdfDocument(BytesIO(broken)) as document:
        assert document.xref_was_recovered
        assert document.page_labels == ["", "", "1", "2"]
