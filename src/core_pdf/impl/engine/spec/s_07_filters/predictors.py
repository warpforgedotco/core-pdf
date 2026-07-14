# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_filters.decode_spec import FilterParams
from core_pdf.impl.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf.impl.third_party.filters.predictors import (
    PredictorError,
    UnsupportedPngFilterError,
    png_predict,
    tiff_predict,
)


def apply_tiff_predictor(data: bytes | memoryview, params: FilterParams) -> bytes:
    try:
        return tiff_predict(
            data,
            columns=params.columns,
            colors=params.colors,
            bits_per_component=params.bits_per_component,
        )
    except PredictorError as exc:
        raise PdfParseError(str(exc)) from exc


def apply_png_predictor(data: bytes | memoryview, params: FilterParams) -> bytes:
    try:
        return png_predict(
            data,
            columns=params.columns,
            colors=params.colors,
            bits_per_component=params.bits_per_component,
            damaged_rows_before_error=params.damaged_rows_before_error,
        )
    except UnsupportedPngFilterError as exc:
        raise PdfUnsupportedError(str(exc)) from exc
    except PredictorError as exc:
        raise PdfParseError(str(exc)) from exc


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
    raise PdfParseError(f"invalid stream predictor {predictor}")


__all__ = ("apply_png_predictor", "apply_predictor", "apply_tiff_predictor")
