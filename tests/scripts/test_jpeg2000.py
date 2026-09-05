from __future__ import annotations

import imagecodecs
import numpy
import pytest

from scripts.jpeg2000 import internal_jpx_uses_irreversible_wavelet


@pytest.mark.parametrize(
    "codec_format", [imagecodecs.JPEG2K.CODEC.J2K, imagecodecs.JPEG2K.CODEC.JP2]
)
@pytest.mark.parametrize(("reversible", "expected"), [(True, False), (False, True)])
def test_jpx_wavelet_classifier_reads_raw_and_boxed_codestreams(
    codec_format: imagecodecs.JPEG2K.CODEC,
    *,
    reversible: bool,
    expected: bool,
) -> None:
    source = numpy.arange(256, dtype=numpy.uint8).reshape(16, 16)
    encoded = bytes(
        imagecodecs.jpeg2k_encode(
            source,
            codecformat=codec_format,
            reversible=reversible,
        )
    )

    assert internal_jpx_uses_irreversible_wavelet(encoded) is expected


def test_jpx_wavelet_classifier_does_not_guess_for_invalid_streams() -> None:
    assert internal_jpx_uses_irreversible_wavelet(b"") is None
    assert internal_jpx_uses_irreversible_wavelet(b"not a JPEG 2000 stream") is None
    assert internal_jpx_uses_irreversible_wavelet(b"\xff\x4f\xff\x52\x00") is None


def test_jpx_wavelet_classifier_rejects_reserved_transform() -> None:
    source = numpy.arange(256, dtype=numpy.uint8).reshape(16, 16)
    encoded = bytearray(
        imagecodecs.jpeg2k_encode(
            source,
            codecformat=imagecodecs.JPEG2K.CODEC.J2K,
            reversible=True,
        )
    )
    cod = encoded.find(b"\xff\x52")
    assert cod >= 0
    encoded[cod + 13] = 2

    assert internal_jpx_uses_irreversible_wavelet(bytes(encoded)) is None
