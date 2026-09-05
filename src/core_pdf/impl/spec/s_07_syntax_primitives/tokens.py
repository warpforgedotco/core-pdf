# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

"""Shared PDF lexical tokens and compact syntax aliases."""

WHITESPACE = b"\x00\t\n\x0c\r "
DELIMITERS = b"()<>[]{}/%"

SEPARATOR_TABLE = bytes([1 if i in WHITESPACE or i in DELIMITERS else 0 for i in range(256)])
WS_TABLE = bytes([1 if i in WHITESPACE else 0 for i in range(256)])

__all__ = (
    "DELIMITERS",
    "SEPARATOR_TABLE",
    "WHITESPACE",
    "WS_TABLE",
)
