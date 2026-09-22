# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, ClassVar, Self, cast

from core_pdf.impl._impl.graphics.filter_registry import (
    CCITT_FILTERS,
    FILTER_NAME_ALIASES,
)
from core_pdf.impl._impl.model.pdf_values import is_pdf_null
from core_pdf.impl._impl.pdf_names import recover_pdf_name
from core_pdf.impl._impl.runtime.scalars import parse_int
from core_pdf.impl.types import PdfReference
from core_pdf_spec.s_07_filters.decode_spec import FilterParams as PdfFilterParams
from core_pdf_spec.s_07_filters.decode_spec import FilterStep, StreamDecodeSpec
from core_pdf_spec.s_07_filters.errors import FilterParseError

internal_frozen_setattr = object.__setattr__


class FilterParams(PdfFilterParams):
    __slots__ = ()

    __fields__: ClassVar[tuple[str, ...]] = (
        "early_change",
        "predictor",
        "columns",
        "colors",
        "bits_per_component",
        "k",
        "damaged_rows_before_error",
        "black_is_1",
        "rows",
        "encoded_byte_align",
        "has_columns",
        "jbig2_globals",
    )
    __match_args__ = (
        "early_change",
        "predictor",
        "columns",
        "colors",
        "bits_per_component",
        "k",
        "damaged_rows_before_error",
        "black_is_1",
        "rows",
        "encoded_byte_align",
        "has_columns",
        "jbig2_globals",
    )

    def __init__(
        self,
        early_change: int = 1,
        predictor: int = 1,
        columns: int = 1,
        colors: int = 1,
        bits_per_component: int = 8,
        k: int = 0,
        damaged_rows_before_error: int = 0,
        black_is_1: bool = False,
        rows: int = 0,
        encoded_byte_align: bool = False,
        has_columns: bool = False,
        jbig2_globals: object | None = None,
    ) -> None:
        internal_frozen_setattr(self, "early_change", early_change)
        internal_frozen_setattr(self, "predictor", predictor)
        internal_frozen_setattr(self, "columns", columns)
        internal_frozen_setattr(self, "colors", colors)
        internal_frozen_setattr(self, "bits_per_component", bits_per_component)
        internal_frozen_setattr(self, "k", k)
        internal_frozen_setattr(self, "damaged_rows_before_error", damaged_rows_before_error)
        internal_frozen_setattr(self, "black_is_1", black_is_1)
        internal_frozen_setattr(self, "rows", rows)
        internal_frozen_setattr(self, "encoded_byte_align", encoded_byte_align)
        internal_frozen_setattr(self, "has_columns", has_columns)
        internal_frozen_setattr(self, "jbig2_globals", jbig2_globals)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"early_change={self.early_change!r}, "
            f"predictor={self.predictor!r}, "
            f"columns={self.columns!r}, "
            f"colors={self.colors!r}, "
            f"bits_per_component={self.bits_per_component!r}, "
            f"k={self.k!r}, "
            f"damaged_rows_before_error={self.damaged_rows_before_error!r}, "
            f"black_is_1={self.black_is_1!r}, "
            f"rows={self.rows!r}, "
            f"encoded_byte_align={self.encoded_byte_align!r}, "
            f"has_columns={self.has_columns!r}, "
            f"jbig2_globals={self.jbig2_globals!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.early_change == other.early_change
            and self.predictor == other.predictor
            and self.columns == other.columns
            and self.colors == other.colors
            and self.bits_per_component == other.bits_per_component
            and self.k == other.k
            and self.damaged_rows_before_error == other.damaged_rows_before_error
            and self.black_is_1 == other.black_is_1
            and self.rows == other.rows
            and self.encoded_byte_align == other.encoded_byte_align
            and self.has_columns == other.has_columns
            and self.jbig2_globals == other.jbig2_globals
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.early_change,
                self.predictor,
                self.columns,
                self.colors,
                self.bits_per_component,
                self.k,
                self.damaged_rows_before_error,
                self.black_is_1,
                self.rows,
                self.encoded_byte_align,
                self.has_columns,
                self.jbig2_globals,
            )
        )

    def __replace__(self, /, **changes: Any) -> Self:
        early_change = changes.pop("early_change", self.early_change)
        predictor = changes.pop("predictor", self.predictor)
        columns = changes.pop("columns", self.columns)
        colors = changes.pop("colors", self.colors)
        bits_per_component = changes.pop("bits_per_component", self.bits_per_component)
        k = changes.pop("k", self.k)
        damaged_rows_before_error = changes.pop(
            "damaged_rows_before_error", self.damaged_rows_before_error
        )
        black_is_1 = changes.pop("black_is_1", self.black_is_1)
        rows = changes.pop("rows", self.rows)
        encoded_byte_align = changes.pop("encoded_byte_align", self.encoded_byte_align)
        has_columns = changes.pop("has_columns", self.has_columns)
        jbig2_globals = changes.pop("jbig2_globals", self.jbig2_globals)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            early_change,
            predictor,
            columns,
            colors,
            bits_per_component,
            k,
            damaged_rows_before_error,
            black_is_1,
            rows,
            encoded_byte_align,
            has_columns,
            jbig2_globals,
        )

    @classmethod
    def from_parms(cls, parms: object) -> FilterParams:
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
        return cast("FilterParams", super().from_parms(normalized))


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
        if is_pdf_null(item) or recover_pdf_name(item) == "null":
            continue
        name = recover_pdf_name(item)
        if name is None:
            raise FilterParseError("invalid stream decode filter")
        name = FILTER_NAME_ALIASES.get(name.lower(), name)
        names.append(name)
        kept_filter_indexes.append(filter_index)

    if is_pdf_null(parms_raw) or recover_pdf_name(parms_raw) == "null":
        decode_parms: list[object] = []
    elif raw_param_items is not None:
        decode_parms = [
            None if is_pdf_null(item) or recover_pdf_name(item) == "null" else item
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

    steps: list[FilterStep] = []
    for index, filter_name in enumerate(names):
        if len(decode_parms) == 1:
            parms = decode_parms[0]
        elif len(decode_parms) == len(names):
            parms = decode_parms[index]
        else:
            parms = None
        if filter_name in CCITT_FILTERS:
            parms = with_ccitt_image_rows(parms, dictionary)
        steps.append(FilterStep(filter_name, parms))
    return StreamDecodeSpec(tuple(steps))
