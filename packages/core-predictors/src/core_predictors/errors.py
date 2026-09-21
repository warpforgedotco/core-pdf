# SPDX-License-Identifier: AGPL-3.0-only
"""Predictor error family shared by the TIFF and PNG kernels."""

from __future__ import annotations


class PredictorError(ValueError):
    """Invalid predictor parameters or data."""


class UnsupportedPngFilterError(PredictorError):
    """Unsupported PNG predictor row filter."""


__all__ = ("PredictorError", "UnsupportedPngFilterError")
