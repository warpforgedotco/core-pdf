"""High-performance pure-Python CCITT fax decoding."""

from core_ccitt.impl.codec import (
    CcittError,
    CcittParseError,
    CcittUnsupportedError,
    decode_ccitt_fax,
)

__all__ = (
    "CcittError",
    "CcittParseError",
    "CcittUnsupportedError",
    "decode_ccitt_fax",
)
