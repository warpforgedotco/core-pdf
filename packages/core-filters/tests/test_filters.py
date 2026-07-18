from __future__ import annotations

import zlib

import pytest
from core_filters.impl.codecs import apply_ascii85, apply_ascii_hex
from core_filters.impl.decode_spec import FilterParams
from core_filters.impl.errors import FilterParseError
from core_filters.impl.flate import apply_flate
from core_filters.impl.predictors import apply_png_predictor, apply_tiff_predictor


def test_ascii_decoders() -> None:
    assert apply_ascii85(b"87cURD]j7BEbo80", {}) == b"Hello world!"
    assert apply_ascii_hex(b"61 62 63>", {}) == b"abc"


def test_flate_decodes_raw_stream() -> None:
    compressor = zlib.compressobj(wbits=-15)
    encoded = compressor.compress(b"hello") + compressor.flush()
    assert apply_flate(encoded, {}) == b"hello"


def test_flate_rejects_garbage() -> None:
    with pytest.raises(FilterParseError):
        apply_flate(b"not compressed data", {})


def test_predictors_reject_truncated_rows() -> None:
    params = FilterParams(columns=4, colors=1, bits_per_component=8)
    with pytest.raises(FilterParseError, match="truncated TIFF predictor row"):
        apply_tiff_predictor(b"abc", params)
    with pytest.raises(FilterParseError, match="truncated PNG predictor row"):
        apply_png_predictor(b"\x00abc", params)
