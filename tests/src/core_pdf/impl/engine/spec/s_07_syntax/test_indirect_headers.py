# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_syntax.indirect_headers import (
    find_indirect_object_header,
)


def test_finds_indirect_object_header_in_full_backing_bytes() -> None:
    data = memoryview(b"prefix\n12 3 obj\nvalue")

    assert find_indirect_object_header(data, 0, len(data)) == len(b"prefix\n")


def test_rejects_object_keyword_cut_off_at_search_end() -> None:
    data = memoryview(b"prefix\n12 3 object\nvalue")
    search_end = data.tobytes().index(b"obj") + len(b"obj")

    assert find_indirect_object_header(data, 0, search_end) is None


def test_rejects_header_whose_token_starts_before_search_window() -> None:
    data = memoryview(b"\n12 3 obj\nvalue")

    assert find_indirect_object_header(data, 2, len(data)) is None


def test_finds_header_from_sliced_memoryview_without_full_source_buffer() -> None:
    wrapped = memoryview(b"outsideprefix\n12 3 obj\nvalueoutside")
    data = wrapped[len(b"outside") : -len(b"outside")]

    assert find_indirect_object_header(data, 0, len(data)) == len(b"prefix\n")
