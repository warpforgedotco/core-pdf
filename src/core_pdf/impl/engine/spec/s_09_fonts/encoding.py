# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.engine.spec.s_09_fonts.data.core14 import PDFDOC_ENCODING_OVERRIDES

BYTE_CACHE = [bytes([i]) for i in range(256)]
CHR_TABLE: list[str] = [chr(i) for i in range(256)]
PDFDOC_ENCODING_TABLE: list[str] = [
    PDFDOC_ENCODING_OVERRIDES.get(i, CHR_TABLE[i]) for i in range(256)
]


def decode_pdf_text_string(data: bytes | memoryview) -> str:
    if type(data) is memoryview:
        data = data.tobytes()
    if data.startswith(b"\xfe\xff"):
        try:
            return data[2:].decode("utf-16-be")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-16BE data") from exc
    if data.startswith(b"\xff\xfe"):
        try:
            return data[2:].decode("utf-16-le")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-16LE data") from exc
    if data.startswith(b"\xef\xbb\xbf"):
        try:
            return data[3:].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-8 data") from exc
    return "".join(PDFDOC_ENCODING_TABLE[b] for b in data)


def decode_utf16be(data: bytes | memoryview | str) -> str:
    if not data:
        return ""
    if type(data) is memoryview:
        data = data.tobytes()
    if isinstance(data, str):
        data = data.encode("latin-1")
    if len(data) == 1:
        return chr(data[0])
    if data.startswith((b"\xfe\xff", b"\xff\xfe")):
        try:
            return data.decode("utf-16")
        except (UnicodeDecodeError, ValueError):
            raise ValueError("invalid UTF-16BE data")
    buf = data if len(data) % 2 == 0 else b"\x00" + data
    try:
        return buf.decode("utf-16-be", "replace")
    except (UnicodeDecodeError, ValueError):
        return data.decode("latin-1", "replace")
