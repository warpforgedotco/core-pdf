# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import core_predictors.png as strict
import core_predictors.tiff as strict_tiff
from core_pdf.impl.graphics.decode_compat import FilterParams
from core_pdf.impl.runtime.codec_backends import (
    png_predict_codec,
    tiff_predict_8,
    tiff_predict_16,
    tiff_predict_bits,
)
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_filters.predictors import SUPPORTED_PREDICTOR_BITS
from core_predictors.errors import PredictorError, UnsupportedPngFilterError


def png_predict(
    data: bytes | memoryview,
    *,
    columns: int,
    colors: int,
    bits_per_component: int,
    damaged_rows_before_error: int = 0,
) -> bytes:
    if bits_per_component not in SUPPORTED_PREDICTOR_BITS:
        raise PredictorError(f"invalid PNG predictor bits {bits_per_component}")
    try:
        decoded = png_predict_codec(
            data, columns=columns, colors=colors, bits_per_component=bits_per_component
        )
    except Exception:
        decoded = None
    if decoded is not None:
        return decoded
    stride = max(1, (colors * columns * bits_per_component + 7) // 8) + 1
    stop = len(data) // stride * stride
    if damaged_rows_before_error:
        for start in range(0, stop, stride):
            if data[start] > 4:
                stop = start
                break
    return strict.png_predict(
        data[:stop], columns=columns, colors=colors, bits_per_component=bits_per_component
    )


def predictor_row_length(params: FilterParams) -> int:
    return (params.columns * params.colors * params.bits_per_component + 7) // 8


def apply_tiff_predictor(data: bytes | memoryview, params: FilterParams) -> bytes:
    if params.bits_per_component in SUPPORTED_PREDICTOR_BITS:
        if not data:
            return b""
        row_length = predictor_row_length(params)
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
        stride = predictor_row_length(params) + 1
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


def tiff_predict(
    data: bytes | memoryview, *, columns: int, colors: int, bits_per_component: int
) -> bytes:
    try:
        if bits_per_component == 8:
            return tiff_predict_8(data, columns, colors)
        if bits_per_component == 16:
            return tiff_predict_16(data, columns, colors)
        if bits_per_component in {1, 2, 4}:
            return tiff_predict_bits(data, columns, colors, bits_per_component)
    except Exception:
        pass
    return strict_tiff.tiff_predict(
        data, columns=columns, colors=colors, bits_per_component=bits_per_component
    )
