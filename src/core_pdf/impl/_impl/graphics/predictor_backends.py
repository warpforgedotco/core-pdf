# SPDX-License-Identifier: AGPL-3.0-only
"""Predictor acceleration and damaged-row compatibility policy."""

from __future__ import annotations

from core_pdf.impl._impl.graphics.decode_compat import FilterParams
from core_pdf.impl._impl.runtime.codec_backends import internal_png_predict_codec
from core_pdf.impl.spec.s_07_filters import predictors as strict
from core_pdf.impl.spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf.impl.spec.s_07_filters.predictors import (
    SUPPORTED_PREDICTOR_BITS,
    PredictorError,
    UnsupportedPngFilterError,
)

PNG_CODEC_THRESHOLD = 0


def png_predict(
    data: bytes | memoryview,
    *,
    columns: int,
    colors: int,
    bits_per_component: int,
    damaged_rows_before_error: int = 0,
) -> bytes:
    if bits_per_component not in {1, 2, 4, 8, 16}:
        raise PredictorError(f"invalid PNG predictor bits {bits_per_component}")
    if len(data) >= PNG_CODEC_THRESHOLD:
        try:
            decoded = internal_png_predict_codec(
                data, columns=columns, colors=colors, bits_per_component=bits_per_component
            )
        except Exception:
            decoded = None
        if decoded is not None:
            return decoded
    stride = max(1, (colors * columns * bits_per_component + 7) // 8) + 1
    # The historical direct helper retains complete rows from a truncated stream.
    stop = len(data) // stride * stride
    if damaged_rows_before_error:
        for start in range(0, stop, stride):
            if data[start] > 4:
                stop = start
                break
    return strict.png_predict(
        data[:stop], columns=columns, colors=colors, bits_per_component=bits_per_component
    )


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


def tiff_predict(
    data: bytes | memoryview, *, columns: int, colors: int, bits_per_component: int
) -> bytes:
    from core_pdf.impl._impl.runtime.codec_backends import (
        tiff_predict_8,
        tiff_predict_16,
        tiff_predict_bits,
    )

    if bits_per_component == 8:
        return tiff_predict_8(data, columns, colors)
    if bits_per_component == 16:
        return tiff_predict_16(data, columns, colors)
    if bits_per_component not in {1, 2, 4}:
        raise PredictorError(f"invalid TIFF predictor bits {bits_per_component}")
    return tiff_predict_bits(data, columns, colors, bits_per_component)
