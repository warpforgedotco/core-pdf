# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
from collections.abc import Callable

from core_pdf.impl.types import PdfByteBuffer
from core_pdf_spec.s_07_syntax_primitives.scanning import STRING_ESCAPE, STRING_SPECIAL_TABLE

internal_STRING_SPECIAL_RE = re.compile(b"[()\\\\\r\n]")


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


def read_literal_string(
    data: bytes | memoryview,
    pos: int,
    data_len: int,
    *,
    unknown_escape: Callable[[int], bytes] | None = None,
    eol_pair: Callable[[int, int], bool] | None = None,
) -> tuple[bytes | None, int]:
    """Decode a literal string at ``pos``, returning its value and end position.

    An unterminated string returns ``None`` and the exhausted position so callers
    can preserve their own error type and cursor semantics.
    """
    pos += 1
    if isinstance(data, memoryview) and data.format != "B":
        # CMap callers may supply numeric elements wider than one byte or signed
        # bytes. Decode those elements directly; byte offsets and prefix copies
        # would change their values or consume data past the closing delimiter.
        end_idx = pos
    elif isinstance(data, bytes) or data.c_contiguous:
        match = internal_STRING_SPECIAL_RE.search(data, pos, data_len)
        end_idx = data_len if match is None else match.start()
    else:
        end_idx = pos
        while end_idx < data_len and not STRING_SPECIAL_TABLE[data[end_idx]]:
            end_idx += 1

    if end_idx < data_len and data[end_idx] == 41:
        return bytes(data[pos:end_idx]), end_idx + 1

    out = bytearray()
    if end_idx > pos:
        prefix = data[pos:end_idx]
        out.extend(prefix if isinstance(prefix, bytes) or prefix.c_contiguous else prefix.tobytes())
        pos = end_idx

    depth = 1
    while pos < data_len:
        byte = data[pos]
        pos += 1
        if byte == 40:
            depth += 1
            out.append(byte)
        elif byte == 41:
            depth -= 1
            if depth == 0:
                return bytes(out), pos
            out.append(byte)
        elif byte == 92:
            if pos >= data_len:
                continue
            esc = data[pos]
            pos += 1
            if 48 <= esc <= 55:
                oct_val = esc - 48
                count = 1
                while count < 3 and pos < data_len and 48 <= data[pos] <= 55:
                    oct_val = (oct_val << 3) | (data[pos] - 48)
                    pos += 1
                    count += 1
                out.append(oct_val & 0xFF)
            elif esc in (10, 13):
                if pos < data_len and (
                    eol_pair(esc, data[pos])
                    if eol_pair is not None
                    else esc == 13 and data[pos] == 10
                ):
                    pos += 1
            elif (mapped := STRING_ESCAPE.get(esc)) is not None:
                out.extend(mapped)
            elif unknown_escape is not None:
                out.extend(unknown_escape(esc))
            else:
                out.append(esc)
        elif byte == 13 or byte == 10:
            out.append(10)
            if pos < data_len:
                next_byte = data[pos]
                if (
                    eol_pair(byte, next_byte)
                    if eol_pair is not None
                    else byte == 13 and next_byte == 10
                ):
                    pos += 1
        else:
            out.append(byte)
    return None, pos
