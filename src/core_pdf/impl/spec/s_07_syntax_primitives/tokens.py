# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

"""Shared PDF lexical tokens and compact syntax aliases."""

WHITESPACE = b"\x00\t\n\x0c\r "
DELIMITERS = b"()<>[]{}/%"

SEPARATOR_TABLE = bytes([1 if i in WHITESPACE or i in DELIMITERS else 0 for i in range(256)])
WS_TABLE = bytes([1 if i in WHITESPACE else 0 for i in range(256)])

# Lexical vocabulary used to recognize content streams before they are parsed.
PDF_CONTENT_OPERATOR_BYTES = frozenset(
    b"""BT ET T* Td TD Tj TJ Tm Tf TL Tc Tw Tz Tr Ts ' " Do BI BDC BMC EMC
    q Q cm g rg k G RG K CS cs SC SCN sc scn sh i ri MP DP BX EX d0 d1
    w J j M d gs m l re h c v y W W* S s f F f* B b B* b* n ID EI""".split()
)

__all__ = (
    "DELIMITERS",
    "PDF_CONTENT_OPERATOR_BYTES",
    "SEPARATOR_TABLE",
    "WHITESPACE",
    "WS_TABLE",
)
