# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.engine.extraction.common.page_profile import content_operator_counts


def test_content_operator_counts_skip_inline_image_data() -> None:
    data = b"q BI /F /A85 ID Tj m m m~>\nEI Q"

    counts = content_operator_counts(data, profile_thresholds=True, may_show_text=False)

    assert counts == {"q": 1, "BI": 1, "Q": 1}


def test_content_operator_counts_skip_operator_like_names() -> None:
    data = b"/Tj /Do /m gs"

    counts = content_operator_counts(data, profile_thresholds=True, may_show_text=False)

    assert counts == {"gs": 1}


def test_content_operator_counts_skip_operators_inside_containers() -> None:
    data = b"[Tj m] << /Text TJ /XObject Do >> q"

    counts = content_operator_counts(data, profile_thresholds=True, may_show_text=False)

    assert counts == {"q": 1}


def test_content_operator_counts_do_not_split_on_vertical_tab() -> None:
    counts = content_operator_counts(b"Tj\vDo q", profile_thresholds=True)

    assert counts == {"q": 1}


def test_content_operator_counts_support_reversed_memoryview() -> None:
    content = b"q m Q"

    counts = content_operator_counts(memoryview(content[::-1])[::-1])

    assert counts == {"q": 1, "m": 1, "Q": 1}


@pytest.mark.parametrize("view_kind", ["bytes", "sliced", "reversed"])
def test_content_operator_counts_skip_whitespace_prefixed_comments(view_kind: str) -> None:
    content = b"q \t% fake Do\n\r % fake Tj\r\n Q"
    if view_kind == "sliced":
        data: bytes | memoryview = memoryview(b"prefix" + content + b"suffix")[6:-6]
    elif view_kind == "reversed":
        data = memoryview(content[::-1])[::-1]
    else:
        data = content

    assert content_operator_counts(data) == {"q": 1, "Q": 1}
