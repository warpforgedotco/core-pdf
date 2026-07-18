# SPDX-License-Identifier: AGPL-3.0-only
"""Compatibility imports for the shared core-filters package."""

from core_filters.impl.errors import FilterParseError
from core_filters.impl.flate import apply_flate as _apply_flate

from core_pdf.impl.exceptions import PdfParseError


def apply_flate(data: bytes, parms: object) -> bytes:
    try:
        return _apply_flate(data, parms)
    except FilterParseError as exc:
        raise PdfParseError(str(exc)) from exc
