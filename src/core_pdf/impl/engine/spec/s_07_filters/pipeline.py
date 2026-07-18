# SPDX-License-Identifier: AGPL-3.0-only
"""Compatibility imports for the shared core-filters package."""

from core_filters.impl.errors import FilterParseError, FilterUnsupportedError
from core_filters.impl.pipeline import *  # noqa: F401,F403
from core_filters.impl.pipeline import decode_stream_data as _decode_stream_data

from core_pdf.impl.exceptions import PdfParseError, PdfUnsupportedError


def decode_stream_data(
    data: bytes | memoryview,
    dictionary: object,
    *,
    parent_dictionary: object | None = None,
) -> bytes:
    try:
        return _decode_stream_data(data, dictionary, parent_dictionary=parent_dictionary)
    except FilterParseError as exc:
        raise PdfParseError(str(exc)) from exc
    except FilterUnsupportedError as exc:
        raise PdfUnsupportedError(str(exc)) from exc
