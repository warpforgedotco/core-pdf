# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations


class PredictorError(ValueError):
    pass


class UnsupportedPngFilterError(PredictorError):
    pass


__all__ = ("PredictorError", "UnsupportedPngFilterError")
