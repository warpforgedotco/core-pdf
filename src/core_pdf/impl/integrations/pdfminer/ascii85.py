# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import base64
import re


def ascii85decode(data: bytes) -> bytes:
    return base64.a85decode(data, adobe=True, ignorechars=b" \t\n\r\v")


def asciihexdecode(data: bytes) -> bytes:
    value = re.sub(rb"\s+", b"", data).rstrip(b">")
    if len(value) % 2:
        value += b"0"
    return bytes.fromhex(value.decode("ascii"))


__all__ = ("ascii85decode", "asciihexdecode")
