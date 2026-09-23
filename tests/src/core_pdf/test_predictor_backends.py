from typing import Any

import pytest

from core_pdf.impl.graphics import stream_decoding as reader
from core_pdf_spec.s_07_filters.errors import FilterParseError


@pytest.mark.parametrize("outcome", [b"codec", None])
@pytest.mark.parametrize("buffer_type", [bytes, memoryview])
def test_png_codec_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
    outcome: bytes | None,
    buffer_type: Any,
) -> None:
    def codec(*args: Any, **kwargs: Any) -> bytes | None:
        return outcome

    monkeypatch.setattr(reader, "png_predict_codec", codec)
    assert reader.png_predict_tolerant(
        buffer_type(b"\x00\x05\x07"), columns=2, colors=1, bits_per_component=8
    ) == (outcome if outcome is not None else b"\x05\x07")


def test_png_fallback_preserves_empty_truncated_and_damaged_row_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reader, "png_predict_codec", lambda *args, **kwargs: None)
    options = {"columns": 2, "colors": 1, "bits_per_component": 8}
    assert reader.png_predict_tolerant(b"", **options) == b""
    assert reader.png_predict_tolerant(b"\x00\x05\x07\x00", **options) == b"\x05\x07"
    assert (
        reader.png_predict_tolerant(
            b"\x00\x05\x07\x09\x00\x00", damaged_rows_before_error=1, **options
        )
        == b"\x05\x07"
    )
    with pytest.raises(FilterParseError, match="truncated PNG"):
        reader.apply_predictor(b"\x00\x05\x07\x00", {"Predictor": 12, "Columns": 2})
    with pytest.raises(reader.PredictorError, match="invalid PNG predictor bits"):
        reader.png_predict_tolerant(b"", columns=2, colors=1, bits_per_component=3)
