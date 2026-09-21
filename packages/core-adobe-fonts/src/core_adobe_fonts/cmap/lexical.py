# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re

WHITESPACE = b"\x00\t\n\x0c\r "
DELIMITERS = b"()<>[]/%"
SEPARATOR_TABLE = bytes([1 if i in WHITESPACE or i in DELIMITERS else 0 for i in range(256)])
WS_TABLE = bytes([1 if i in WHITESPACE else 0 for i in range(256)])

internal_STRING_SPECIAL_TABLE = bytes([1 if i in b"()\\\r\n" else 0 for i in range(256)])
internal_STRING_SPECIAL_RE = re.compile(b"[" + re.escape(b"()\\\r\n") + b"]")
internal_STRING_ESCAPE = {
    110: b"\n",
    114: b"\r",
    116: b"\t",
    98: b"\b",
    102: b"\x0c",
    40: b"(",
    41: b")",
    92: b"\\",
}


def read_literal_string(
    data: bytes | memoryview, pos: int, data_len: int
) -> tuple[bytes | None, int]:
    pos += 1
    if isinstance(data, memoryview) and data.format != "B":
        end_idx = pos
    elif isinstance(data, bytes) or data.c_contiguous:
        match = internal_STRING_SPECIAL_RE.search(data, pos, data_len)
        end_idx = data_len if match is None else match.start()
    else:
        end_idx = pos
        while end_idx < data_len and not internal_STRING_SPECIAL_TABLE[data[end_idx]]:
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
                if pos < data_len and esc == 13 and data[pos] == 10:
                    pos += 1
            elif (mapped := internal_STRING_ESCAPE.get(esc)) is not None:
                out.extend(mapped)
            else:
                out.append(esc)
        elif byte == 13 or byte == 10:
            out.append(10)
            if pos < data_len and byte == 13 and data[pos] == 10:
                pos += 1
        else:
            out.append(byte)
    return None, pos


__all__ = ("WHITESPACE", "DELIMITERS", "SEPARATOR_TABLE", "WS_TABLE", "read_literal_string")
