# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.spec.s_07_filters.decode_spec import FilterParams
from core_pdf.impl.spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf.impl.spec.s_07_filters.predictor_impl import (
    PredictorError,
    UnsupportedPngFilterError,
    png_predict,
    tiff_predict,
)

SUPPORTED_PREDICTOR_BITS = frozenset({1, 2, 4, 8, 16})


def apply_tiff_predictor(data: bytes | memoryview, params: FilterParams) -> bytes:
    if params.bits_per_component in SUPPORTED_PREDICTOR_BITS:
        if not data:
            return b""
        row_length = (params.columns * params.colors * params.bits_per_component + 7) // 8
        if row_length and len(data) % row_length:
            raise FilterParseError("truncated TIFF predictor row")
    try:
        return tiff_predict(
            data,
            columns=params.columns,
            colors=params.colors,
            bits_per_component=params.bits_per_component,
        )
    except PredictorError as exc:
        raise FilterParseError(str(exc)) from exc


def apply_png_predictor(data: bytes | memoryview, params: FilterParams) -> bytes:
    if params.bits_per_component in SUPPORTED_PREDICTOR_BITS:
        if not data:
            return b""
        row_length = (params.columns * params.colors * params.bits_per_component + 7) // 8
        stride = row_length + 1
        if len(data) % stride and not params.damaged_rows_before_error:
            raise FilterParseError("truncated PNG predictor row")
    try:
        return png_predict(
            data,
            columns=params.columns,
            colors=params.colors,
            bits_per_component=params.bits_per_component,
            damaged_rows_before_error=params.damaged_rows_before_error,
        )
    except UnsupportedPngFilterError as exc:
        raise FilterUnsupportedError(str(exc)) from exc
    except PredictorError as exc:
        raise FilterParseError(str(exc)) from exc


def apply_predictor(data: bytes | memoryview, parms: object) -> bytes:
    if parms is None or parms == {}:
        return bytes(data)
    params = parms if type(parms) is FilterParams else FilterParams.from_parms(parms)
    predictor = params.predictor
    if predictor == 1:
        return bytes(data)
    if predictor == 2:
        return apply_tiff_predictor(data, params)
    if predictor >= 10:
        return apply_png_predictor(data, params)
    raise FilterParseError(f"invalid stream predictor {predictor}")


__all__ = ("apply_png_predictor", "apply_predictor", "apply_tiff_predictor")
