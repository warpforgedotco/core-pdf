# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.types import PdfByteBuffer


def matches_keyword_with_one_substitution(
    data: PdfByteBuffer | memoryview, pos: int, keyword: bytes
) -> bool:
    """Whether ``keyword`` sits at ``pos`` with exactly one byte substituted."""
    end = pos + len(keyword)
    if end > len(data):
        return False
    mismatches = 0
    for index, expected in enumerate(keyword):
        if data[pos + index] != expected:
            mismatches += 1
            if mismatches > 1:
                return False
    return mismatches == 1
