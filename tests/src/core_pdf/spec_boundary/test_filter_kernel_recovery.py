"""Reader filter policies remain separate from strict sample and predictor kernels."""

import numpy
import pytest

from core_pdf.impl._impl.graphics import predictor_backends
from core_pdf.impl._impl.graphics.decode_compat import FilterParams, normalize_stream_decode_spec
from core_pdf.impl._impl.graphics.image_kernels import unpack_subbyte_image_samples
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_filters.predictors import png_predict as strict_png_predict


def test_reader_normalization_keeps_parameter_padding_and_skipped_filter_alignment() -> None:
    params = {"Columns": 3}
    spec = normalize_stream_decode_spec(
        {"Filter": [None, "ASCII85Decode", "FlateDecode"], "DecodeParms": [None, None, params]}
    )
    assert tuple(step.name for step in spec.steps) == ("ASCII85Decode", "FlateDecode")
    assert spec.steps[0].params is None
    assert spec.steps[1].params is params
    padded = normalize_stream_decode_spec(
        {"Filter": ["ASCII85Decode", "FlateDecode"], "DecodeParms": [None]}
    )
    assert tuple(step.params for step in padded.steps) == (None, None)


@pytest.mark.parametrize(
    ("bits", "width", "colors", "data", "expected"),
    [
        (1, 3, 3, b"\xaa\xff", [1, 0, 1, 0, 1, 0, 1, 0, 1]),
        (2, 3, 1, b"\x1b\xe7", [0, 1, 2, 3, 2, 1]),
        (4, 1, 3, b"\x12\x3f", [1, 2, 3]),
    ],
)
def test_reader_acceleration_retains_row_aligned_sample_values(
    bits: int, width: int, colors: int, data: bytes, expected: list[int]
) -> None:
    height = len(expected) // (width * colors)
    actual = unpack_subbyte_image_samples(data, bits, width, height, colors)
    assert numpy.asarray(actual).tolist() == expected


def test_reader_png_fallback_retains_complete_rows_when_permitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        predictor_backends, "internal_png_predict_codec", lambda *args, **kwargs: None
    )
    calls: list[bytes] = []

    def record_strict_fallback(data: bytes | memoryview, **kwargs: int) -> bytes:
        calls.append(bytes(data))
        return strict_png_predict(data, **kwargs)

    monkeypatch.setattr(predictor_backends.strict, "png_predict", record_strict_fallback)
    data = b"\x00\x11\x00"
    assert (
        predictor_backends.png_predict(data, columns=1, colors=1, bits_per_component=8) == b"\x11"
    )
    assert (
        predictor_backends.apply_png_predictor(data, FilterParams(damaged_rows_before_error=1))
        == b"\x11"
    )
    assert calls == [b"\x00\x11", b"\x00\x11"]
    with pytest.raises(FilterParseError, match="^truncated PNG predictor row$"):
        predictor_backends.apply_png_predictor(data, FilterParams())


def test_reader_png_fallback_stops_before_a_damaged_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        predictor_backends, "internal_png_predict_codec", lambda *args, **kwargs: None
    )
    data = b"\x00\x11\x05\x22\x00\x33"
    assert (
        predictor_backends.apply_png_predictor(data, FilterParams(damaged_rows_before_error=1))
        == b"\x11"
    )
    with pytest.raises(FilterUnsupportedError, match="Unsupported PNG predictor filter 5"):
        predictor_backends.apply_png_predictor(data, FilterParams())
