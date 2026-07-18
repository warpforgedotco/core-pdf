import concurrent.futures
import sys
from dataclasses import FrozenInstanceError

import pytest

from core_jpeg import (
    DecodedJpxComponent,
    DecodedJpxImage,
    JpegError,
    JpegParseError,
    JpegUnsupportedError,
    decode_dct,
    decode_jpx,
    decode_jpx_image,
)


def test_error_types_share_public_base() -> None:
    assert issubclass(JpegParseError, JpegError)
    assert issubclass(JpegUnsupportedError, JpegError)


def test_decode_dct_rejects_empty_input() -> None:
    with pytest.raises(JpegUnsupportedError, match="JPEGDecode failed"):
        decode_dct(b"")


@pytest.mark.parametrize("decoder", [decode_jpx, decode_jpx_image])
def test_jpx_decoders_reject_empty_input(decoder: object) -> None:
    with pytest.raises(JpegParseError, match="unexpected end"):
        decoder(b"")  # type: ignore[operator]


def test_jpx_tile_executor_falls_back_on_python_313(monkeypatch: pytest.MonkeyPatch) -> None:
    from core_jpeg.impl.codecs.jpx import tiles

    monkeypatch.setattr(sys, "version_info", (3, 13, 0))

    assert tiles._jpx_tile_executor_class() is concurrent.futures.ProcessPoolExecutor


def test_decoded_image_models_are_immutable() -> None:
    component = DecodedJpxComponent(
        index=0,
        width=1,
        height=1,
        precision=8,
        is_signed=False,
        data=b"\x7f",
    )
    image = DecodedJpxImage(
        width=1,
        height=1,
        color_space="gray",
        components=(component,),
        interleaved=b"\x7f",
    )

    assert image.components == (component,)
    with pytest.raises(FrozenInstanceError):
        image.width = 2  # type: ignore[misc]
