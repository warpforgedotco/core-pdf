# SPDX-License-Identifier: AGPL-3.0-only
"""Compatibility imports for the shared core-filters package."""

from core_filters.impl.decode_spec import StreamDecodeSpec
from core_filters.impl.decode_spec import normalize_stream_decode_spec as _normalize
from core_filters.impl.errors import FilterParseError

from core_pdf.impl.exceptions import PdfParseError


def normalize_stream_decode_spec(dictionary: object) -> StreamDecodeSpec:
    try:
        return _normalize(dictionary)
    except FilterParseError as exc:
        raise PdfParseError(str(exc)) from exc
