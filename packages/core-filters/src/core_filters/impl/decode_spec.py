# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TypeAlias, cast

from core_filters.impl.errors import FilterParseError
from core_filters.impl.helpers import (
    is_pdf_null,
    lookup_dict_key,
    normalize_pdf_name,
    parse_int,
)

DecodeParam: TypeAlias = object


class FilterParams:
    """Decode parameters associated with a PDF stream filter."""

    __slots__ = (
        "early_change",
        "predictor",
        "columns",
        "colors",
        "bits_per_component",
        "k",
        "damaged_rows_before_error",
        "rows",
        "encoded_byte_align",
        "has_columns",
        "jbig2_globals",
    )

    early_change: int
    predictor: int
    columns: int
    colors: int
    bits_per_component: int
    k: int
    damaged_rows_before_error: bool
    rows: int
    encoded_byte_align: bool
    has_columns: bool

    def __init__(
        self,
        early_change: int = 1,
        predictor: int = 1,
        columns: int = 1,
        colors: int = 1,
        bits_per_component: int = 8,
        k: int = 0,
        damaged_rows_before_error: bool = False,
        rows: int = 0,
        encoded_byte_align: bool = False,
        has_columns: bool = False,
        jbig2_globals: object | None = None,
    ) -> None:
        self.early_change = early_change
        self.predictor = predictor
        self.columns = columns
        self.colors = colors
        self.bits_per_component = bits_per_component
        self.k = k
        self.damaged_rows_before_error = damaged_rows_before_error
        self.rows = rows
        self.encoded_byte_align = encoded_byte_align
        self.has_columns = has_columns
        self.jbig2_globals = jbig2_globals

    def replace(self, **kwargs: object) -> "FilterParams":
        return FilterParams(
            early_change=cast(int, kwargs.get("early_change", self.early_change)),
            predictor=cast(int, kwargs.get("predictor", self.predictor)),
            columns=cast(int, kwargs.get("columns", self.columns)),
            colors=cast(int, kwargs.get("colors", self.colors)),
            bits_per_component=cast(int, kwargs.get("bits_per_component", self.bits_per_component)),
            k=cast(int, kwargs.get("k", self.k)),
            damaged_rows_before_error=cast(
                bool,
                kwargs.get("damaged_rows_before_error", self.damaged_rows_before_error),
            ),
            rows=cast(int, kwargs.get("rows", self.rows)),
            encoded_byte_align=cast(
                bool, kwargs.get("encoded_byte_align", self.encoded_byte_align)
            ),
            has_columns=cast(bool, kwargs.get("has_columns", self.has_columns)),
            jbig2_globals=kwargs.get("jbig2_globals", self.jbig2_globals),
        )

    @classmethod
    def from_parms(cls, parms: object) -> "FilterParams":
        if not isinstance(parms, dict):
            if is_pdf_null(parms) or (
                hasattr(parms, "object_number") and hasattr(parms, "generation_number")
            ):
                return cls()
            raise ValueError("invalid DecodeParms dictionary")

        def require_int(name: str, default: int | None = None) -> int:
            value = lookup_dict_key(parms, name)
            if is_pdf_null(value):
                value = default
            parsed = parse_int(value, default)
            if parsed is None:
                raise ValueError(f"invalid DecodeParms {name}")
            if value is not None and type(value) is bool:
                raise ValueError(f"invalid DecodeParms {name}")
            if value is not None and not isinstance(value, (int, bytes, str)):
                raise ValueError(f"invalid DecodeParms {name}")
            return parsed

        def require_pos_int(name: str, default: int | None = None) -> int:
            parsed = require_int(name, default)
            if parsed <= 0:
                raise ValueError(f"invalid DecodeParms {name}")
            return parsed

        def require_predictor(name: str) -> int:
            value = require_pos_int(name, 1)
            if value > 15:
                raise ValueError(f"invalid DecodeParms {name}")
            return value

        def require_bits_per_component(name: str) -> int:
            value = require_pos_int(name, 8)
            if value > 16:
                raise ValueError(f"invalid DecodeParms {name}")
            return value

        def require_bool(name: str, default: bool = False) -> bool:
            value = lookup_dict_key(parms, name)
            if is_pdf_null(value):
                value = default
            if type(value) is bool:
                return value
            if value in (0, 1):
                return bool(value)
            if value is default and isinstance(default, bool):
                return default
            raise ValueError(f"invalid DecodeParms {name}")

        def require_nonneg_int(name: str, default: int | None = None) -> int:
            parsed = require_int(name, default)
            if parsed < 0:
                raise ValueError(f"invalid DecodeParms {name}")
            return parsed

        def require_early_change() -> int:
            value = require_int("EarlyChange", 1)
            if value not in (0, 1):
                raise ValueError("invalid DecodeParms EarlyChange")
            return 0 if value == 0 else 1

        columns_value = lookup_dict_key(parms, "Columns")
        jbig2_globals = lookup_dict_key(parms, "JBIG2Globals")
        return cls(
            early_change=require_early_change(),
            predictor=require_predictor("Predictor"),
            columns=require_pos_int("Columns", 1),
            colors=require_pos_int("Colors", 1),
            bits_per_component=require_bits_per_component("BitsPerComponent"),
            k=require_int("K", 0) or 0,
            damaged_rows_before_error=require_bool("DamagedRowsBeforeError", False),
            rows=require_nonneg_int("Rows", 0),
            encoded_byte_align=require_bool("EncodedByteAlign", False),
            has_columns=not is_pdf_null(columns_value),
            jbig2_globals=None if is_pdf_null(jbig2_globals) else jbig2_globals,
        )


class StreamDecodeSpec:
    """Normalized stream filter pipeline and per-filter parameters."""

    __slots__ = ("filters", "params")

    filters: tuple[str, ...]
    params: tuple[DecodeParam, ...]

    def __init__(self, filters: tuple[str, ...], params: tuple[DecodeParam, ...]) -> None:
        self.filters = filters
        self.params = params

    def replace(self, **kwargs: object) -> "StreamDecodeSpec":
        return StreamDecodeSpec(
            filters=cast(tuple[str, ...], kwargs.get("filters", self.filters)),
            params=cast(tuple[DecodeParam, ...], kwargs.get("params", self.params)),
        )


FILTER_NAME_ALIASES = {
    "a85": "A85",
    "ahx": "AHx",
    "ascii85decode": "ASCII85Decode",
    "asciihexdecode": "ASCIIHexDecode",
    "ccf": "CCF",
    "ccitt": "CCITTFaxDecode",
    "ccittfaxdecode": "CCITTFaxDecode",
    "crypt": "Crypt",
    "dct": "DCT",
    "dctdecode": "DCTDecode",
    "flatedecode": "FlateDecode",
    "fl": "Fl",
    "identity": "Identity",
    "jbig2decode": "JBIG2Decode",
    "jpxdecode": "JPXDecode",
    "platedecode": "FlateDecode",
    "rl": "RL",
    "runlengthdecode": "RunLengthDecode",
    "runlength": "RunLengthDecode",
    "none": "None",
    "lzwdecode": "LZWDecode",
    "lzw": "LZW",
}
CCITT_FILTERS = {"CCITTFaxDecode", "CCF"}
normalize_filter_name = normalize_pdf_name


def with_ccitt_image_rows(parms: object, dictionary: object) -> object:
    if type(parms) is FilterParams:
        return parms
    height = lookup_dict_key(dictionary, "Height")
    if is_pdf_null(height):
        return parms
    if is_pdf_null(parms):
        return {"Rows": height}
    if not isinstance(parms, dict):
        return parms
    if not is_pdf_null(lookup_dict_key(parms, "Rows")):
        return parms
    updated = dict(parms)
    updated["Rows"] = height
    return updated


def normalize_stream_decode_spec(dictionary: object) -> StreamDecodeSpec:
    if not isinstance(dictionary, dict):
        raise FilterParseError("invalid stream dictionary")
    raw_filters = lookup_dict_key(dictionary, "Filter")
    if is_pdf_null(raw_filters):
        raw_filters = lookup_dict_key(dictionary, "F")
    if is_pdf_null(raw_filters):
        raw_filters = lookup_dict_key(dictionary, "FFilter")
    if is_pdf_null(raw_filters):
        filters: list[object] = []
    else:
        filters = list(raw_filters) if isinstance(raw_filters, (list, tuple)) else [raw_filters]
    parms_raw = lookup_dict_key(dictionary, "DecodeParms")
    if is_pdf_null(parms_raw):
        parms_raw = lookup_dict_key(dictionary, "DP")
    if is_pdf_null(parms_raw):
        parms_raw = lookup_dict_key(dictionary, "FDecodeParms")
    raw_param_items = list(parms_raw) if isinstance(parms_raw, (list, tuple)) else None

    names: list[str] = []
    kept_filter_indexes: list[int] = []
    for filter_index, item in enumerate(filters):
        if is_pdf_null(item) or normalize_filter_name(item) == "null":
            continue
        name = normalize_filter_name(item)
        if name is None:
            raise FilterParseError("invalid stream decode filter")
        name = FILTER_NAME_ALIASES.get(name.lower(), name)
        names.append(name)
        kept_filter_indexes.append(filter_index)

    if is_pdf_null(parms_raw) or normalize_filter_name(parms_raw) == "null":
        decode_parms: list[object] = []
    elif raw_param_items is not None:
        decode_parms = [
            None if is_pdf_null(item) or normalize_filter_name(item) == "null" else item
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
