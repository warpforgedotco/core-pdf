# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 7.4: the errors a filter or predictor raises."""

from __future__ import annotations


class FilterError(Exception):
    pass


class FilterParseError(FilterError, ValueError):
    pass


class FilterUnsupportedError(FilterError):
    pass


class PredictorError(ValueError):
    pass


class UnsupportedPngFilterError(PredictorError):
    pass


__all__ = (
    "FilterError",
    "FilterParseError",
    "FilterUnsupportedError",
    "PredictorError",
    "UnsupportedPngFilterError",
)
