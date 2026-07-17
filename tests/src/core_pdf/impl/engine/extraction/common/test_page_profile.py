# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.engine.extraction.common.page_profile import content_operator_counts


def test_content_operator_counts_skip_inline_image_data() -> None:
    data = b"q BI /F /A85 ID Tj m m m~>\nEI Q"

    counts = content_operator_counts(data, profile_thresholds=True, may_show_text=False)

    assert counts == {"q": 1, "BI": 1, "Q": 1}


def test_content_operator_counts_skip_operator_like_names() -> None:
    data = b"/Tj /Do /m gs"

    counts = content_operator_counts(data, profile_thresholds=True, may_show_text=False)

    assert counts == {"gs": 1}
