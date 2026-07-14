# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

"""Shared PDF lexical tokens and compact syntax aliases."""

WHITESPACE = b"\x00\t\n\x0c\r "
DELIMITERS = b"()<>[]{}/%"

SEPARATOR_TABLE = bytes([1 if i in WHITESPACE or i in DELIMITERS else 0 for i in range(256)])
WS_TABLE = bytes([1 if i in WHITESPACE else 0 for i in range(256)])

INLINE_IMAGE_KEY_MAP = {
    "BPC": "BitsPerComponent",
    "CS": "ColorSpace",
    "D": "Decode",
    "DP": "DecodeParms",
    "F": "Filter",
    "H": "Height",
    "IM": "ImageMask",
    "I": "Interpolate",
    "W": "Width",
}

__all__ = (
    "DELIMITERS",
    "INLINE_IMAGE_KEY_MAP",
    "SEPARATOR_TABLE",
    "WHITESPACE",
    "WS_TABLE",
)
