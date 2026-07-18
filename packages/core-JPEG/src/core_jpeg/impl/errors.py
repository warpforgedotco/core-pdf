# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations


class JpegError(Exception):
    """Base error for JPEG-family codec failures."""


class JpegParseError(JpegError):
    """Raised when JPEG-family bytes are malformed."""


class JpegUnsupportedError(JpegError):
    """Raised when a valid JPEG-family feature is not supported."""


__all__ = (
    "JpegError",
    "JpegParseError",
    "JpegUnsupportedError",
)
