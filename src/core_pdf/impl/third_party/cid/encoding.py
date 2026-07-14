from __future__ import annotations

BYTE_CACHE = [bytes([i]) for i in range(256)]


def decode_utf16be(data: bytes | memoryview | str) -> str:
    if not data:
        return ""
    if isinstance(data, memoryview):
        data = data.tobytes()
    if isinstance(data, str):
        data = data.encode("latin-1")
    if len(data) == 1:
        return chr(data[0])
    if data.startswith((b"\xfe\xff", b"\xff\xfe")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError, ValueError:
            raise ValueError("invalid UTF-16BE data")
    buf = data if len(data) % 2 == 0 else b"\x00" + data
    try:
        return buf.decode("utf-16-be", "replace")
    except UnicodeDecodeError, ValueError:
        return data.decode("latin-1", "replace")
