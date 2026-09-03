"""PDF/CID encoding helpers."""

from __future__ import annotations


def decode_utf16be(data: bytes) -> str:
    if not data:
        return ""
    if data.startswith(b"\xfe\xff"):
        data = data[2:]
    if len(data) == 1:
        return chr(data[0])
    buf = data if len(data) % 2 == 0 else b"\x00" + data
    try:
        return buf.decode("utf-16-be", "replace")
    except (UnicodeDecodeError, ValueError):
        return data.decode("latin-1", "replace")
