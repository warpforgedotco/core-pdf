"""A JPX image the native path declines is decoded once, to the filter chain's bytes."""

from typing import Any

import imagecodecs
import numpy
import pytest

from core_pdf.impl import graphics_codec_backends as codec_backends
from core_pdf.impl.graphics_images import decode_image_samples
from core_pdf.impl.graphics_stream_decoding import decode_stream_data
from core_pdf.impl.types import PdfName


def jpx_image(color_space: object) -> tuple[bytes, dict[Any, Any]]:
    pixels = numpy.arange(6 * 5 * 3, dtype=numpy.uint8).reshape(6, 5, 3)
    data = bytes(imagecodecs.jpeg2k_encode(pixels, level=0))
    dictionary = {
        PdfName.of(b"Width"): 5,
        PdfName.of(b"Height"): 6,
        PdfName.of(b"BitsPerComponent"): 8,
        PdfName.of(b"ColorSpace"): color_space,
        PdfName.of(b"Filter"): PdfName.of(b"JPXDecode"),
    }
    return data, dictionary


def test_a_declined_image_is_decoded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    data, dictionary = jpx_image([PdfName.of(b"CalRGB"), {}])
    expected = decode_stream_data(data, dictionary)
    calls: list[int] = []
    original = codec_backends.decode_jpx_image

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(codec_backends, "decode_jpx_image", counted)
    samples = decode_image_samples(data, dictionary)
    assert samples == expected
    assert type(samples) is bytes
    assert len(calls) == 1


def test_an_accepted_image_is_still_the_native_array() -> None:
    data, dictionary = jpx_image(PdfName.of(b"DeviceRGB"))
    samples = decode_image_samples(data, dictionary)
    assert not isinstance(samples, (bytes, bytearray, memoryview))
    assert samples is not None
