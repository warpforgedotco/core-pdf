# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations


class FilterError(Exception):
    """Base error raised by core-filters."""


class FilterParseError(FilterError, ValueError):
    """The encoded stream or its parameters are malformed."""


class FilterUnsupportedError(FilterError):
    """The stream uses a valid but unsupported feature."""
