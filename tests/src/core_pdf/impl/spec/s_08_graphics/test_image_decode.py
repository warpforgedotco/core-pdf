# SPDX-License-Identifier: AGPL-3.0-only
"""decode_pdf_image and ImageSource: decoding, soft masks, and stencil masks."""

from __future__ import annotations

import imagecodecs
import numpy

from core_pdf.impl.spec.s_08_graphics.color import internal_convert_cmyk
from core_pdf.impl.spec.s_08_graphics.image_decode import (
    ImageSource,
    SoftMask,
    decode_pdf_image,
)


def test_image_decode_recovers_unambiguous_samples_from_malformed_icc() -> None:
    samples = bytes((10, 20, 30, 40, 50, 60))

    decoded = decode_pdf_image(
        samples,
        {
            "Width": 2,
            "Height": 1,
            "BitsPerComponent": 8,
            "ColorSpace": ["ICCBased", object()],
        },
    )

    assert decoded is not None
    assert decoded.channels == 3
    assert decoded.data == samples


def test_image_decode_converts_cmyk_jpeg_samples_to_rgb() -> None:
    samples = numpy.array([[[10, 20, 30, 40]]], dtype=numpy.uint8)
    encoded = bytes(imagecodecs.jpeg_encode(samples, level=100))

    decoded = decode_pdf_image(
        encoded,
        {
            "Filter": "DCTDecode",
            "Width": 1,
            "Height": 1,
            "BitsPerComponent": 8,
            "ColorSpace": "DeviceCMYK",
        },
    )

    assert decoded is not None
    assert decoded.channels == 3
    jpeg_samples = numpy.asarray(imagecodecs.jpeg_decode(encoded), dtype=numpy.uint8)
    expected = internal_convert_cmyk(jpeg_samples.reshape(-1))
    numpy.testing.assert_array_equal(decoded.data, expected)


def test_image_source_decodes_into_read_only_ndarray() -> None:
    source = ImageSource(
        bytes((10, 20, 30, 40, 50, 60)),
        {
            "Width": 2,
            "Height": 1,
            "ColorSpace": "DeviceRGB",
            "BitsPerComponent": 8,
        },
    )

    first = source.decode()
    second = source.decode()

    assert first is not None
    assert second is not None
    numpy.testing.assert_array_equal(second.array, first.array)
    assert first.array.shape == (1, 2, 3)
    assert first.array.strides[0] == first.stride
    assert not first.array.flags.writeable


def test_image_source_applies_soft_mask() -> None:
    source = ImageSource(
        bytes((10, 20, 30, 40, 50, 60)),
        {
            "Width": 2,
            "Height": 1,
            "ColorSpace": "DeviceRGB",
            "BitsPerComponent": 8,
        },
        soft_mask=SoftMask(
            bytes((0, 255)),
            {
                "Width": 2,
                "Height": 1,
                "ColorSpace": "DeviceGray",
                "BitsPerComponent": 8,
            },
        ),
    )

    raster = source.decode()

    assert raster is not None
    assert raster.has_alpha
    numpy.testing.assert_array_equal(raster.array[0, :, 3], (0, 255))


def test_image_source_prepares_native_soft_mask() -> None:
    source = ImageSource(
        bytes((10, 20, 30)),
        {
            "Width": 1,
            "Height": 1,
            "ColorSpace": "DeviceRGB",
            "BitsPerComponent": 8,
        },
        soft_mask=SoftMask(
            bytes((0, 64, 128, 255)),
            {
                "Width": 2,
                "Height": 2,
                "ColorSpace": "DeviceGray",
                "BitsPerComponent": 8,
            },
        ),
    )

    prepared = source.prepare()

    assert prepared is not None
    assert prepared.soft_mask is not None
    assert prepared.soft_mask.array.shape == (2, 2, 1)
    numpy.testing.assert_array_equal(prepared.soft_mask.array[:, :, 0], ((0, 64), (128, 255)))


def test_image_mask_source_exposes_alpha_channel() -> None:
    source = ImageSource(
        bytes((0b10000000,)),
        {
            "Width": 2,
            "Height": 1,
            "ImageMask": True,
            "BitsPerComponent": 1,
        },
    )

    raster = source.decode()

    assert raster is not None
    assert raster.color_model == "gray"
    # ISO 32000-1 8.9.6.2: with the default Decode [0 1] "a sample value of 0
    # shall mark the page ... and a 1 shall leave the previous contents
    # unchanged", so the 0 bit is the opaque one.
    numpy.testing.assert_array_equal(raster.array[0, :, 1], (0, 255))
