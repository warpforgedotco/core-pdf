# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, ClassVar, NoReturn, Protocol, Self, TypeAlias

from core_pdf_spec.s_07_filters.errors import FilterParseError
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    decoded_name,
    is_pdf_null,
    require_pdf_integer,
)

frozen_setattr = object.__setattr__


DecodeParam: TypeAlias = object


class FilterParams:
    __slots__ = (
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

    early_change: int
    predictor: int
    columns: int
    colors: int
    bits_per_component: int
    k: int
    damaged_rows_before_error: int
    black_is_1: bool
    rows: int
    encoded_byte_align: bool
    has_columns: bool
    jbig2_globals: object | None

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
        frozen_setattr(self, "early_change", early_change)
        frozen_setattr(self, "predictor", predictor)
        frozen_setattr(self, "columns", columns)
        frozen_setattr(self, "colors", colors)
        frozen_setattr(self, "bits_per_component", bits_per_component)
        frozen_setattr(self, "k", k)
        frozen_setattr(self, "damaged_rows_before_error", damaged_rows_before_error)
        frozen_setattr(self, "black_is_1", black_is_1)
        frozen_setattr(self, "rows", rows)
        frozen_setattr(self, "encoded_byte_align", encoded_byte_align)
        frozen_setattr(self, "has_columns", has_columns)
        frozen_setattr(self, "jbig2_globals", jbig2_globals)

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

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

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
            if is_pdf_null(parms):
                return cls()
            raise ValueError("invalid DecodeParms dictionary")

        def require_int(name: str, default: int | None = None) -> int:
            value = parms.get(name)
            if is_pdf_null(value):
                value = default
            return require_pdf_integer(value, f"invalid DecodeParms {name}")

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
            value = parms.get(name)
            if is_pdf_null(value):
                value = default
            if type(value) is bool:
                return value
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
            return value

        columns_value = parms.get("Columns")
        jbig2_globals = parms.get("JBIG2Globals")
        return cls(
            early_change=require_early_change(),
            predictor=require_predictor("Predictor"),
            columns=require_pos_int("Columns", 1),
            colors=require_pos_int("Colors", 1),
            bits_per_component=require_bits_per_component("BitsPerComponent"),
            k=require_int("K", 0),
            damaged_rows_before_error=require_nonneg_int("DamagedRowsBeforeError", 0),
            black_is_1=require_bool("BlackIs1", False),
            rows=require_nonneg_int("Rows", 0),
            encoded_byte_align=require_bool("EncodedByteAlign", False),
            has_columns=not is_pdf_null(columns_value),
            jbig2_globals=None if is_pdf_null(jbig2_globals) else jbig2_globals,
        )


class FilterStep:
    __slots__ = ("name", "params")

    name: str
    params: DecodeParam

    __fields__: ClassVar[tuple[str, ...]] = ("name", "params")
    __match_args__ = ("name", "params")

    def __init__(self, name: str, params: DecodeParam = None) -> None:
        frozen_setattr(self, "name", name)
        frozen_setattr(self, "params", params)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(name={self.name!r}, params={self.params!r})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.name == other.name and self.params == other.params

    def __hash__(self) -> int:
        return hash((self.name, self.params))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        name = changes.pop("name", self.name)
        params = changes.pop("params", self.params)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(name, params)


class StreamDecodeSpec:
    __slots__ = ("steps",)

    steps: tuple[FilterStep, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("steps",)
    __match_args__ = ("steps",)

    def __init__(self, steps: tuple[FilterStep, ...]) -> None:
        frozen_setattr(self, "steps", steps)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(steps={self.steps!r})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.steps == other.steps

    def __hash__(self) -> int:
        return hash((self.steps,))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        steps = changes.pop("steps", self.steps)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(steps)


def normalize_stream_decode_spec(dictionary: object) -> StreamDecodeSpec:
    if not isinstance(dictionary, dict):
        raise FilterParseError("invalid stream dictionary")
    raw_filters = dictionary.get("Filter")
    if is_pdf_null(raw_filters):
        raw_filters = dictionary.get("FFilter")
    if is_pdf_null(raw_filters):
        filters: list[object] = []
    else:
        filters = list(raw_filters) if isinstance(raw_filters, (list, tuple)) else [raw_filters]
    parms_raw = dictionary.get("DecodeParms")
    if is_pdf_null(parms_raw):
        parms_raw = dictionary.get("FDecodeParms")

    names: list[str] = []
    for item in filters:
        name = decoded_name(item)
        if name is None:
            raise FilterParseError("invalid stream decode filter")
        names.append(name)

    if not names:
        params: tuple[DecodeParam, ...] = ()
    elif is_pdf_null(parms_raw):
        params = (None,) * len(names)
    elif isinstance(parms_raw, (list, tuple)):
        if len(parms_raw) != len(names):
            raise FilterParseError("invalid stream decode parameters")
        params = tuple(parms_raw)
    elif len(names) == 1:
        params = (parms_raw,)
    else:
        raise FilterParseError("invalid stream decode parameters")
    return StreamDecodeSpec(tuple(FilterStep(name, param) for name, param in zip(names, params)))


class StreamDecoder(Protocol):
    def __call__(
        self,
        data: bytes | memoryview,
        dictionary: object | StreamDecodeSpec | None,
        *,
        parent_dictionary: object | None = None,
    ) -> bytes: ...


__all__ = (
    "DecodeParam",
    "FilterParams",
    "FilterStep",
    "StreamDecodeSpec",
    "normalize_stream_decode_spec",
    "StreamDecoder",
)
