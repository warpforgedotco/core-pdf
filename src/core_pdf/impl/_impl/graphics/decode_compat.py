# SPDX-License-Identifier: AGPL-3.0-only
"""Normalize stream dictionaries into a decode specification."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TypeAlias

from core_pdf.impl._impl.graphics.filter_registry import (
    CCITT_FILTERS,
    FILTER_NAME_ALIASES,
)
from core_pdf.impl._impl.model.pdf_values import is_pdf_null
from core_pdf.impl._impl.runtime.scalars import parse_int
from core_pdf.impl.spec.s_07_filters.decode_spec import FilterParams as PdfFilterParams
from core_pdf.impl.spec.s_07_filters.decode_spec import StreamDecodeSpec
from core_pdf.impl.spec.s_07_filters.errors import FilterParseError
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf.impl.types import PdfReference

DecodeParam: TypeAlias = object


@dataclass(frozen=True, slots=True)
class FilterParams(PdfFilterParams):
    """Decode parameters associated with a PDF stream filter."""

    @classmethod
    def from_parms(cls, parms: object) -> "FilterParams":
        if not isinstance(parms, dict):
            if is_pdf_null(parms) or isinstance(parms, PdfReference):
                return cls()
            raise ValueError("invalid DecodeParms dictionary")
        normalized = dict(parms)
        integer_names = (
            "EarlyChange",
            "Predictor",
            "Columns",
            "Colors",
            "BitsPerComponent",
            "K",
            "Rows",
            "DamagedRowsBeforeError",
        )
        for name in integer_names:
            value = parms.get(name)
            if is_pdf_null(value):
                normalized.pop(name, None)
                continue
            if name == "DamagedRowsBeforeError" and type(value) is bool:
                normalized[name] = int(value)
                continue
            if type(value) is bool or not isinstance(value, (int, bytes, str)):
                raise ValueError(f"invalid DecodeParms {name}")
            parsed = parse_int(value, None)
            if parsed is None:
                raise ValueError(f"invalid DecodeParms {name}")
            normalized[name] = parsed
        for name in ("BlackIs1", "EncodedByteAlign"):
            value = parms.get(name)
            if is_pdf_null(value):
                normalized.pop(name, None)
            elif type(value) is bool:
                normalized[name] = value
            elif value in (0, 1):
                normalized[name] = bool(value)
            else:
                raise ValueError(f"invalid DecodeParms {name}")
        if is_pdf_null(parms.get("JBIG2Globals")):
            normalized["JBIG2Globals"] = None
        parsed_params = PdfFilterParams.from_parms(normalized)
        return cls(
            **{item.name: getattr(parsed_params, item.name) for item in fields(parsed_params)}
        )


def with_ccitt_image_rows(parms: object, dictionary: object) -> object:
    if type(parms) is FilterParams:
        return parms
    height = dictionary.get("Height") if isinstance(dictionary, dict) else None
    if is_pdf_null(height):
        return parms
    if is_pdf_null(parms):
        return {"Rows": height}
    if not isinstance(parms, dict):
        return parms
    if not is_pdf_null(parms.get("Rows")):
        return parms
    updated = dict(parms)
    updated["Rows"] = height
    return updated


def normalize_stream_decode_spec(dictionary: object) -> StreamDecodeSpec:
    if not isinstance(dictionary, dict):
        raise FilterParseError("invalid stream dictionary")
    raw_filters = dictionary.get("Filter")
    # ISO 32000-1 Table 5: on a regular stream /F is a *file specification* and
    # the filters for that external data are named by /FFilter, so /FFilter is
    # consulted first. /F means "Filter" only for inline images (Table 93), and
    # those are already normalized to Filter/DecodeParms before reaching here --
    # the abbreviation fallback below is kept only as leniency for writers that
    # use it on a regular stream.
    if is_pdf_null(raw_filters):
        raw_filters = dictionary.get("FFilter")
    if is_pdf_null(raw_filters):
        raw_filters = dictionary.get("F")
    if is_pdf_null(raw_filters):
        filters: list[object] = []
    else:
        filters = list(raw_filters) if isinstance(raw_filters, (list, tuple)) else [raw_filters]
    parms_raw = dictionary.get("DecodeParms")
    if is_pdf_null(parms_raw):
        parms_raw = dictionary.get("FDecodeParms")
    if is_pdf_null(parms_raw):
        parms_raw = dictionary.get("DP")
    raw_param_items = list(parms_raw) if isinstance(parms_raw, (list, tuple)) else None

    names: list[str] = []
    kept_filter_indexes: list[int] = []
    for filter_index, item in enumerate(filters):
        if is_pdf_null(item) or normalize_pdf_name(item) == "null":
            continue
        name = normalize_pdf_name(item)
        if name is None:
            raise FilterParseError("invalid stream decode filter")
        name = FILTER_NAME_ALIASES.get(name.lower(), name)
        names.append(name)
        kept_filter_indexes.append(filter_index)

    if is_pdf_null(parms_raw) or normalize_pdf_name(parms_raw) == "null":
        decode_parms: list[object] = []
    elif raw_param_items is not None:
        decode_parms = [
            None if is_pdf_null(item) or normalize_pdf_name(item) == "null" else item
            for item in raw_param_items
        ]
        if len(decode_parms) >= len(filters):
            decode_parms = [
                decode_parms[index] for index in kept_filter_indexes if index < len(decode_parms)
            ]
    else:
        if len(names) > 1:
            raise FilterParseError("invalid stream decode parameters")
        decode_parms = [parms_raw]

    if not names:
        decode_parms = []

    if isinstance(parms_raw, (list, tuple)) and len(decode_parms) < len(names):
        decode_parms.extend([None] * (len(names) - len(decode_parms)))

    if isinstance(parms_raw, (list, tuple)) and len(decode_parms) > len(names):
        decode_parms = decode_parms[: len(names)]

    if len(decode_parms) not in {0, 1, len(names)}:
        raise FilterParseError("invalid stream decode parameters")

    if len(decode_parms) == 1 and len(names) > 1:
        raise FilterParseError("invalid stream decode parameters")

    params: list[DecodeParam] = []
    for index, filter_name in enumerate(names):
        if len(decode_parms) == 1:
            parms = decode_parms[0]
        elif len(decode_parms) == len(names):
            parms = decode_parms[index]
        else:
            parms = None
        if filter_name in CCITT_FILTERS:
            parms = with_ccitt_image_rows(parms, dictionary)
        params.append(parms)
    return StreamDecodeSpec(filters=tuple(names), params=tuple(params))
