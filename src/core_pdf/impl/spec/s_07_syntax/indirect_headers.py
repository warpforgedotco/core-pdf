# SPDX-License-Identifier: AGPL-3.0-only
"""Native indirect-object header scanning helpers."""

from __future__ import annotations

from core_pdf.impl.spec.s_07_syntax.xref import parse_object_marker_prefix
from core_pdf.impl.spec.s_07_syntax_primitives.lexer_helpers import (
    FindableSizedBuffer,
    full_source_buffer,
)


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

    Thin wrapper around ``s_07_syntax.xref.parse_object_marker_prefix``, which runs the
    same scan; this variant only needs the start offset. Used on the corrupt-PDF
    recovery path (``ObjectResolver.recover_indirect_object``), not the parse hot path,
    so the extra int-parsing it does is not worth duplicating the scan to avoid.
    """
    result = parse_object_marker_prefix(data, marker)
    return None if result is None else result[0]
