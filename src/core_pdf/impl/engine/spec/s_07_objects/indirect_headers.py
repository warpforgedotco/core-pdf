# SPDX-License-Identifier: AGPL-3.0-only
"""Native indirect-object header scanning helpers."""

from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_syntax.lexer_helpers import (
    FindableSizedBuffer,
    full_source_buffer,
)
from core_pdf.impl.engine.spec.s_07_syntax.tokens import WS_TABLE


def find_indirect_object_header(
    data: memoryview,
    search_start: int,
    search_end: int,
    source_buffer: FindableSizedBuffer | None = None,
) -> int | None:
    data_len = len(data)
    search_start = max(0, search_start)
    search_end = min(data_len, search_end)
    source = source_buffer if source_buffer is not None else full_source_buffer(data, data_len)
    copied_region = data[search_start:search_end].tobytes() if source is None else None
    pos = search_start
    while pos < search_end:
        if source is not None:
            marker = source.find(b"obj", pos, search_end)
        else:
            assert copied_region is not None
            marker = copied_region.find(b"obj", pos - search_start)
        if marker < 0:
            return None
        if source is None:
            marker += search_start
        parsed = parse_object_header_prefix(data, marker)
        if parsed is not None and parsed >= search_start:
            return parsed
        pos = marker + 3
    return None


def parse_object_header_prefix(data: memoryview, marker: int) -> int | None:
    """Return the offset of the ``N G obj`` header ending at ``marker``, or None.

    Deliberately mirrors ``s_07_syntax.xref.parse_object_marker_prefix``: the scan is
    identical, but this variant only needs the start offset, so it skips that one's two
    ``int()`` conversions and tuple allocation. Keep the two in sync.
    """
    if marker + 3 < len(data) and not WS_TABLE[data[marker + 3]]:
        return None
    pos = marker - 1
    while pos >= 0 and WS_TABLE[data[pos]]:
        pos -= 1
    gen_end = pos + 1
    while pos >= 0 and 48 <= data[pos] <= 57:
        pos -= 1
    gen_start = pos + 1
    if gen_start == gen_end:
        return None
    while pos >= 0 and WS_TABLE[data[pos]]:
        pos -= 1
    obj_end = pos + 1
    while pos >= 0 and 48 <= data[pos] <= 57:
        pos -= 1
    obj_start = pos + 1
    if obj_start == obj_end:
        return None
    if pos >= 0 and not WS_TABLE[data[pos]]:
        return None
    return obj_start
