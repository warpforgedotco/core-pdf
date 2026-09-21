# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations


class FilterError(Exception):
    pass


class FilterParseError(FilterError, ValueError):
    pass


class FilterUnsupportedError(FilterError):
    pass


__all__ = (
    "FilterError",
    "FilterParseError",
    "FilterUnsupportedError",
)
