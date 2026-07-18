"""High-performance pure-Python JBIG2 decoding."""

from core_jbig2.impl.codec import (
    JBIG2_FILE_HEADER,
    Jbig2Error,
    JBIG2Image,
    Jbig2ParseError,
    JBIG2Segment,
    Jbig2UnsupportedError,
    assemble_embedded_jbig2,
    decode_embedded_jbig2,
    parse_jbig2_file,
)

__all__ = (
    "JBIG2Image",
    "JBIG2Segment",
    "JBIG2_FILE_HEADER",
    "Jbig2Error",
    "Jbig2ParseError",
    "Jbig2UnsupportedError",
    "assemble_embedded_jbig2",
    "decode_embedded_jbig2",
    "parse_jbig2_file",
)
