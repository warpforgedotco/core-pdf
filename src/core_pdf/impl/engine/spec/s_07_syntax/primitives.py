# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import builtins
from collections.abc import Callable
from functools import lru_cache
from typing import Protocol, TypeAlias, TypeGuard, TypeVar

"""Core PDF primitive object types."""


MISSING = object()
PDF_NAME_CACHE: dict[bytes, "PdfName"] = {}
PdfScalar: TypeAlias = int | float | str | bytes
PdfDictLike: TypeAlias = dict[object, "PdfObject"]
_T = TypeVar("_T")


class _UnsetType:
    pass


_UNSET = _UnsetType()


def _value_or(value: _T | _UnsetType, default: _T) -> _T:
    if isinstance(value, _UnsetType):
        return default
    return value


@lru_cache(maxsize=4096)
def parse_name_str(value: str) -> str:
    return value


@lru_cache(maxsize=4096)
def parse_name_bytes(value: bytes) -> str:
    return value.decode("latin-1")


def parse_name(value: PdfObject, default: str | None = None) -> str | None:
    """Coerce value to a PDF name string."""
    if type(value) is PdfName:
        return str(value)
    if type(value) is str:
        return parse_name_str(value)
    if type(value) is bytes:
        return parse_name_bytes(value)
    return default


def parse_int(value: PdfObject, default: int | None = None) -> int | None:
    """Coerce value to an integer."""
    if type(value) is int:
        return value
    if type(value) is bytes:
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return default
    if not isinstance(value, (int, str, bytes)):
        return default
    try:
        return int(value)
    except TypeError, ValueError:
        return default


def parse_int_strict(value: PdfObject) -> int:
    """Coerce value to an integer or raise ValueError."""
    parsed = parse_int(value)
    if parsed is None:
        raise ValueError(f"invalid integer {value!r}")
    return parsed


def parse_float(value: PdfObject, default: float = 0.0) -> float:
    """Coerce value to a float."""
    if type(value) is float:
        return value
    if not isinstance(value, (int, float, str, bytes)):
        return default
    try:
        return float(value)
    except TypeError, ValueError:
        return default


class PdfName:
    """PDF name object. Internally stores raw bytes for zero-copy."""

    __slots__ = ("_value", "str", "hash")

    _value: bytes
    str: builtins.str | None
    hash: int | None

    def __init__(self, value: bytes) -> None:
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "str", value.decode("latin-1"))
        object.__setattr__(self, "hash", hash(self.str))

    @property
    def value(self) -> builtins.str:
        return self.str or ""

    @property
    def text(self) -> builtins.str:
        return self.str or ""

    @classmethod
    def of(cls, value: builtins.str | bytes | memoryview | "PdfName") -> "PdfName":
        # Fast-path for already resolved names
        if type(value) is PdfName:
            return value

        cache = PDF_NAME_CACHE
        if type(value) is str:
            string_value: builtins.str = value
            b_value = string_value.encode("latin-1")
            n = cache.get(b_value)
            if n is not None:
                return n
            cache[b_value] = n = cls(b_value)
            return n

        # memoryview is hashable and can match bytes keys in dict.get()
        key_bytes: bytes
        if type(value) is memoryview:
            key_bytes = bytes(value)
        elif type(value) is bytes:
            key_bytes = value
        else:
            key_value: builtins.str
            if isinstance(value, PdfName):
                key_bytes = value._value
            else:
                key_value = value if type(value) is str else str(value)
                key_bytes = key_value.encode("latin-1")
        n = cache.get(key_bytes)
        if n is None:
            cache[key_bytes] = n = cls(key_bytes)
        return n

    def __str__(self) -> builtins.str:
        return self.str or ""

    def __repr__(self) -> builtins.str:
        return f"PdfName({self._value!r})"

    def __hash__(self) -> int:
        h = self.hash
        if h is None:
            h = hash(self.str)
            object.__setattr__(self, "hash", h)
        return h

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PdfName):
            return self._value == other._value
        if isinstance(other, bytes):
            return self._value == other
        if isinstance(other, str):
            s = self.str
            if s is not None:
                return s == other
            return self._value == other.encode("latin-1")
        return False

    def __setattr__(self, name: builtins.str, value: object) -> None:
        raise AttributeError(f"cannot assign to field {name!r}")


def lookup_dict_key(value: PdfDictLike | None, key: str) -> "PdfObject":
    """Look up ``key`` in a PDF dict, tolerating PdfName and ``/``-prefixed keys."""
    if not isinstance(value, dict):
        return None

    # Fast path: direct lookup
    res = value.get(key)
    if res is not None:
        return res

    # Check for PdfName key
    pdf_key = PdfName.of(key)
    res = value.get(pdf_key)
    if res is not None:
        return res

    # Slowest path: iterate and match by string (needed for some malformed PDFs)
    for k in value:
        if isinstance(k, PdfName):
            if k.value == key:
                return value[k]
        elif str(k).lstrip("/") == key:
            return value[k]

    return None


def collect_inherited_values(
    node: PdfDictLike,
    keys: tuple[str, ...],
    deep_resolve: Callable[["PdfObject", set[int]], "PdfObject"],
    cache: dict[int, dict[str, "PdfObject"]] | None = None,
) -> dict[str, "PdfObject"]:
    """Collect inheritable values from a PDF node and its Parent chain."""
    values: dict[str, PdfObject] = {}
    current: PdfDictLike | PdfObject = node
    ancestors: list[tuple[int, PdfDictLike]] = []
    cached_values: dict[str, PdfObject] | None = None
    seen: set[int] = set()
    while isinstance(current, dict):
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)
        if cache is not None:
            cached_values = cache.get(marker)
            if cached_values is not None:
                for key, value in cached_values.items():
                    if key not in values:
                        values[key] = value
                break

        ancestors.append((marker, current))
        for key in keys:
            if key not in values:
                value = lookup_dict_key(current, key)
                if value is not None:
                    values[key] = value

        parent = lookup_dict_key(current, "Parent")
        current = deep_resolve(parent, set()) if parent is not None else None

    if cache is not None:
        running_values = cached_values if cached_values is not None else {}
        for marker, node_dict in reversed(ancestors):
            merged = running_values.copy()
            for key in keys:
                value = lookup_dict_key(node_dict, key)
                if value is not None:
                    merged[key] = value
            cache[marker] = merged
            running_values = merged

    return values


class PdfReference:
    """Indirect object reference."""

    __slots__ = ("object_number", "generation_number")

    object_number: int
    generation_number: int

    def __init__(self, object_number: int, generation_number: int = 0) -> None:
        if object_number < 0 or generation_number < 0:
            raise ValueError("invalid PDF reference")
        self.object_number = object_number
        self.generation_number = generation_number

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PdfReference):
            return (
                self.object_number == other.object_number
                and self.generation_number == other.generation_number
            )
        return False

    def __hash__(self) -> int:
        return hash((self.object_number, self.generation_number))

    @property
    def obj_num(self) -> int:
        return self.object_number

    @property
    def gen_num(self) -> int:
        return self.generation_number


class PdfString:
    """PDF string object."""

    __slots__ = ("data",)

    data: bytes

    def __init__(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise ValueError("invalid PDF string")
        self.data = data

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PdfString):
            return self.data == other.data
        return False

    def __hash__(self) -> int:
        return hash(self.data)

    def __repr__(self) -> str:
        return f"PdfString({self.data!r})"


class FilterParams:
    """Parameters for stream filters."""

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

    def replace(
        self,
        *,
        early_change: int | _UnsetType = _UNSET,
        predictor: int | _UnsetType = _UNSET,
        columns: int | _UnsetType = _UNSET,
        colors: int | _UnsetType = _UNSET,
        bits_per_component: int | _UnsetType = _UNSET,
        k: int | _UnsetType = _UNSET,
        damaged_rows_before_error: bool | _UnsetType = _UNSET,
        rows: int | _UnsetType = _UNSET,
        encoded_byte_align: bool | _UnsetType = _UNSET,
        has_columns: bool | _UnsetType = _UNSET,
    ) -> "FilterParams":
        """Create a new FilterParams with modified fields."""
        return FilterParams(
            early_change=_value_or(early_change, self.early_change),
            predictor=_value_or(predictor, self.predictor),
            columns=_value_or(columns, self.columns),
            colors=_value_or(colors, self.colors),
            bits_per_component=_value_or(bits_per_component, self.bits_per_component),
            k=_value_or(k, self.k),
            damaged_rows_before_error=_value_or(
                damaged_rows_before_error, self.damaged_rows_before_error
            ),
            rows=_value_or(rows, self.rows),
            encoded_byte_align=_value_or(encoded_byte_align, self.encoded_byte_align),
            has_columns=_value_or(has_columns, self.has_columns),
        )

    @classmethod
    def from_parms(cls, parms: "PdfObject") -> "FilterParams":
        if not isinstance(parms, dict):
            if parms is None:
                return cls()
            raise ValueError("invalid DecodeParms dictionary")

        def require_int(name: str, default: int | None = None) -> int:
            value = parms.get(name, default)
            parsed = parse_int(value, default)
            if parsed is None:
                raise ValueError(f"invalid DecodeParms {name}")
            if value is not None and not isinstance(value, (int, bytes, str)):
                raise ValueError(f"invalid DecodeParms {name}")
            return parsed

        def require_bool(name: str, default: bool = False) -> bool:
            value = parms.get(name, default)
            if isinstance(value, bool):
                return value
            if value in (0, 1):
                return bool(value)
            if value is default and isinstance(default, bool):
                return default
            raise ValueError(f"invalid DecodeParms {name}")

        return cls(
            early_change=0 if require_int("EarlyChange", 1) == 0 else 1,
            predictor=require_int("Predictor", 1) or 1,
            columns=require_int("Columns", 1) or 1,
            colors=require_int("Colors", 1) or 1,
            bits_per_component=require_int("BitsPerComponent", 8) or 8,
            k=require_int("K", 0) or 0,
            damaged_rows_before_error=require_bool("DamagedRowsBeforeError", False),
            rows=require_int("Rows", 0) or 0,
            encoded_byte_align=require_bool("EncodedByteAlign", False),
            has_columns="Columns" in parms,
        )


class StreamDecodeSpec:
    """Specification for decoding a stream."""

    __slots__ = ("filters", "params")

    filters: list[str]
    params: list[FilterParams]

    def __init__(self, filters: list[str], params: list[FilterParams]) -> None:
        self.filters = filters
        self.params = params

    def replace(
        self,
        *,
        filters: list[str] | _UnsetType = _UNSET,
        params: list[FilterParams] | _UnsetType = _UNSET,
    ) -> "StreamDecodeSpec":
        """Create a new StreamDecodeSpec with modified fields."""
        return StreamDecodeSpec(
            filters=_value_or(filters, self.filters),
            params=_value_or(params, self.params),
        )


class PdfStream:
    """Stream object with its associated dictionary and data. Decodes lazily."""

    __slots__ = ("dictionary", "raw_data", "spec", "decoded_data")

    dictionary: PdfDictLike

    def __init__(
        self,
        dictionary: PdfDictLike | None = None,
        raw_data: bytes = b"",
        spec: StreamDecodeSpec | None = None,
        decoded_data: bytes | None = None,
    ) -> None:
        if dictionary is not None and not isinstance(dictionary, dict):
            raise ValueError("invalid stream dictionary")
        if not isinstance(raw_data, bytes):
            raise ValueError("invalid stream data")
        if spec is not None and not isinstance(spec, StreamDecodeSpec):
            raise ValueError("invalid stream decode spec")
        if decoded_data is not None and not isinstance(decoded_data, bytes):
            raise ValueError("invalid stream data")
        self.dictionary = dictionary if dictionary is not None else {}
        self.raw_data = raw_data
        self.spec = spec
        self.decoded_data = decoded_data

    def replace(
        self,
        *,
        dictionary: PdfDictLike | _UnsetType = _UNSET,
        raw_data: bytes | _UnsetType = _UNSET,
        spec: StreamDecodeSpec | None | _UnsetType = _UNSET,
        decoded_data: bytes | None | _UnsetType = _UNSET,
    ) -> "PdfStream":
        """Create a new PdfStream with modified fields."""
        resolved_spec = _value_or(spec, self.spec)
        if resolved_spec is not None and not isinstance(resolved_spec, StreamDecodeSpec):
            raise ValueError("invalid stream decode spec")
        return PdfStream(
            dictionary=_value_or(dictionary, self.dictionary),
            raw_data=_value_or(raw_data, self.raw_data),
            spec=resolved_spec,
            decoded_data=_value_or(decoded_data, self.decoded_data),
        )

    @property
    def data(self) -> bytes:
        if self.decoded_data is None:
            from core_pdf.impl.engine.spec.s_07_filters.filters import decode_stream_data

            self.decoded_data = decode_stream_data(self.raw_data, self.spec)
        return self.decoded_data

    @property
    def data_view(self) -> memoryview:
        return memoryview(self.data)

    @property
    def _raw_data(self) -> bytes:
        return self.raw_data


class PdfReadableSource(Protocol):
    def read(self) -> bytes | bytearray | memoryview: ...


PdfSource: TypeAlias = str | bytes | bytearray | memoryview | PdfReadableSource
"""PDF matrix helpers."""

PdfPrimitiveObject: TypeAlias = (
    None | bool | int | float | str | bytes | PdfName | PdfReference | PdfString
)
PdfObject: TypeAlias = (
    PdfPrimitiveObject | PdfStream | list["PdfObject"] | tuple["PdfObject", ...] | PdfDictLike
)


Matrix = tuple[float, float, float, float, float, float]
IDENTITY_MATRIX: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def is_float_matrix_values(
    value: object,
) -> TypeGuard[tuple[float, float, float, float, float, float]]:
    return (
        isinstance(value, tuple)
        and len(value) == 6
        and all(isinstance(item, float) for item in value)
    )


def is_pdf_scalar_matrix_values(
    value: object,
) -> TypeGuard[tuple[PdfScalar, PdfScalar, PdfScalar, PdfScalar, PdfScalar, PdfScalar]]:
    return (
        isinstance(value, tuple)
        and len(value) == 6
        and all(isinstance(item, (int, float, str, bytes)) for item in value)
    )


def parse_matrix_operand(operands: "PdfObject") -> Matrix:
    if not isinstance(operands, (list, tuple)) or len(operands) < 6:
        raise ValueError("invalid matrix operand")
    a0, a1, a2, a3, a4, a5 = (
        operands[0],
        operands[1],
        operands[2],
        operands[3],
        operands[4],
        operands[5],
    )
    raw_values = (a0, a1, a2, a3, a4, a5)
    if is_float_matrix_values(raw_values):
        return (
            raw_values[0],
            raw_values[1],
            raw_values[2],
            raw_values[3],
            raw_values[4],
            raw_values[5],
        )
    if not is_pdf_scalar_matrix_values(raw_values):
        raise ValueError("invalid matrix operand")
    try:
        return (
            float(raw_values[0]),
            float(raw_values[1]),
            float(raw_values[2]),
            float(raw_values[3]),
            float(raw_values[4]),
            float(raw_values[5]),
        )
    except TypeError, ValueError:
        raise ValueError("invalid matrix operand")


def matrix_multiply(
    left: Matrix,
    right: Matrix,
) -> Matrix:
    # Fast path: if right is identity, return left unchanged
    if right == IDENTITY_MATRIX:
        return left
    a1, b1, c1, d1, e1, f1 = left
    a2, b2, c2, d2, e2, f2 = right
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def normalize_pdf_name(value: "PdfObject", default: str | None = None) -> str | None:
    name = parse_name(value, default)
    if name is not None and name.startswith("/"):
        return name[1:]
    return name


def coerce_to_bytes(value: "PdfObject") -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, PdfString):
        return value.data
    if isinstance(value, str):
        return value.encode("latin-1")
    data = getattr(value, "data", None)
    if isinstance(data, bytes):
        return data
    raise TypeError(f"cannot coerce {type(value).__name__} to bytes")


def coerce_value(
    value: "PdfObject",
    string_decoder: Callable[[bytes], object] | None = None,
) -> object:
    if string_decoder is not None:
        if isinstance(value, PdfString):
            return string_decoder(value.data)
        if isinstance(value, bytes):
            return string_decoder(value)
    if isinstance(value, dict):
        return {str(key): coerce_value(val, string_decoder) for key, val in value.items()}
    if isinstance(value, list):
        return [coerce_value(item, string_decoder) for item in value]
    if isinstance(value, tuple):
        return [coerce_value(item, string_decoder) for item in value]
    return value


STROKE_OPS = frozenset({"S", "s", "B", "b", "B*", "b*"})
CLEAR_OPS = frozenset({"f", "f*", "F", "n"})
CLOSE_OPS = frozenset({"s", "b", "b*"})
