# SPDX-License-Identifier: AGPL-3.0-only
"""Page label formatting (12.4.2) and the number-tree rules around index 0."""

from __future__ import annotations

import pytest

from core_pdf.impl.spec.s_07_document.document_labels import format_alpha, format_page_label


@pytest.mark.parametrize(
    ("number", "expected"),
    [(1, "a"), (26, "z"), (27, "aa"), (28, "bb"), (52, "zz"), (53, "aaa")],
)
def test_page_label_alphabetic_sequence(number: int, expected: str) -> None:
    assert format_alpha(number) == expected


def test_page_label_without_style_is_prefix_only() -> None:
    assert format_page_label({"P": b"Appendix-"}, 17, lambda value: value) == "Appendix-"
