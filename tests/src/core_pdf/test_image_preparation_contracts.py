import zlib

import numpy as np
import pytest

from core_pdf.impl import graphics_images as images
from core_pdf_spec.s_08_graphics.image_spec import ImageSource, SoftMask


@pytest.mark.parametrize(("model", "channels"), [("gray", 1), ("gray", 2), ("rgb", 3), ("rgb", 4)])
@pytest.mark.parametrize("strided", [False, True])
def test_canonical_raster_retains_samples_and_exposes_readonly_layout(model, channels, strided):
    source = np.arange(4 * 3 * channels, dtype=np.uint8).reshape(4, 3, channels)
    if strided:
        source = source[::-1]
    raster = images.ImageRaster(source, model)
    assert (raster.width, raster.height, raster.channels) == (3, 4, channels)
    assert raster.stride == 3 * channels
    assert raster.has_alpha is (channels in {2, 4})
    np.testing.assert_array_equal(raster.array, source)
    assert raster.array.flags.c_contiguous
    assert not raster.array.flags.writeable
    with pytest.raises(ValueError):
        raster.array[0, 0, 0] = 1


def test_two_dimensional_gray_samples_gain_a_channel_axis():
    raster = images.ImageRaster(np.array([[1, 2], [3, 4]], dtype=np.uint8), "gray")
    assert raster.array.tolist() == [[[1], [2]], [[3], [4]]]


@pytest.mark.parametrize(
    ("shape", "model"),
    [
        ((2,), "gray"),
        ((1, 1, 5), "rgb"),
        ((1, 1, 3), "gray"),
        ((1, 1, 1), "rgb"),
        ((1, 1, 3), "cmyk"),
    ],
)
def test_invalid_raster_layouts_are_rejected(shape, model):
    with pytest.raises(ValueError, match="image raster"):
        images.ImageRaster(np.zeros(shape, dtype=np.uint8), model)


@pytest.mark.parametrize(("model", "channels"), [("gray", 2), ("rgb", 3), ("rgb", 4)])
def test_prepared_soft_mask_requires_gray_without_intrinsic_alpha(model, channels):
    raster = images.ImageRaster(np.zeros((1, 1, 3), dtype=np.uint8), "rgb")
    mask = images.ImageRaster(np.zeros((1, 1, channels), dtype=np.uint8), model)
    with pytest.raises(ValueError, match="soft mask must be grayscale"):
        images.PreparedImage(raster, mask)


@pytest.mark.parametrize("decode", [None, [0, 1], [1, 0], [], ["invalid", 0]])
@pytest.mark.parametrize("compressed", [False, True])
def test_packed_stencil_rows_ignore_padding_and_honor_decode_direction(decode, compressed):
    raw = bytes([0b10101010, 0b11000000, 0b01010101, 0])
    dictionary = {"ImageMask": True, "Width": 10, "Height": 2, "Decode": decode}
    if compressed:
        dictionary["Filter"] = "FlateDecode"
        raw = zlib.compress(raw)
    prepared = images.prepare_image(ImageSource(raw, dictionary))
    assert prepared is not None
    assert prepared.is_stencil
    alpha = np.array([[0, 255] * 4 + [0, 0], [255, 0] * 4 + [255, 255]], dtype=np.uint8)
    if decode == [1, 0]:
        alpha = 255 - alpha
    np.testing.assert_array_equal(prepared.raster.array[:, :, 1], alpha)
    assert not prepared.raster.array[:, :, 0].any()


@pytest.mark.parametrize(
    ("width", "height", "raw", "filter_name"),
    [(0, 1, b"x", None), (1, -1, b"x", None), (10, 2, b"x", None), (1, 1, b"bad", "FlateDecode")],
)
def test_invalid_stencil_geometry_or_encoding_is_dropped(width, height, raw, filter_name):
    assert (
        images.prepare_image(
            ImageSource(
                raw, {"ImageMask": True, "Width": width, "Height": height, "Filter": filter_name}
            )
        )
        is None
    )


@pytest.mark.parametrize(
    ("bits", "raw"),
    [(1, b"\x40"), (2, b"\x30"), (4, b"\x0f"), (8, b"\0\xff"), (16, b"\0\0\xff\xff")],
)
@pytest.mark.parametrize("decode", [(0, 1), (1, 0), (-1, 2)])
def test_matte_alpha_decodes_original_sample_precision(bits, raw, decode):
    source = ImageSource(b"", {"Width": 2, "Height": 1})
    mask = SoftMask(
        raw, {"Width": 2, "Height": 1, "BitsPerComponent": bits, "Matte": [0.5], "Decode": decode}
    )
    matte, alpha = images.decode_matte(source, mask)
    assert matte == (0.5,)
    np.testing.assert_array_equal(alpha, [1, 0] if decode == (1, 0) else [0, 1])


@pytest.mark.parametrize(
    "changes", [{"Width": 3}, {"Height": 2}, {"Decode": [0]}, {"Decode": [0, 1, 2]}]
)
def test_invalid_matte_dimensions_and_decode_are_rejected(changes):
    source = ImageSource(b"", {"Width": 2, "Height": 1})
    dictionary = {"Width": 2, "Height": 1, "BitsPerComponent": 8, "Matte": [0.5], **changes}
    with pytest.raises(ValueError, match="matte requires matching|soft mask Decode"):
        images.decode_matte(source, SoftMask(b"\0\xff", dictionary))


@pytest.mark.parametrize(("model", "channels"), [("DeviceGray", 1), ("DeviceRGB", 3)])
@pytest.mark.parametrize("color_key", [False, True])
def test_soft_mask_replaces_color_key_without_mutating_source_dictionary(
    model, channels, color_key
):
    dictionary = {"Width": 2, "Height": 2, "BitsPerComponent": 8, "ColorSpace": model}
    if color_key:
        dictionary["Mask"] = [0, 255] * channels
    original = dictionary.copy()
    source = ImageSource(
        bytes([50] * 4 * channels),
        dictionary,
        soft_mask=SoftMask(b"\0\xff", {"Width": 1, "Height": 2}),
    )
    prepared = images.prepare_image(source)
    assert prepared is not None
    assert prepared.soft_mask is not None
    assert prepared.raster.has_alpha
    np.testing.assert_array_equal(prepared.raster.array[:, :, -1], [[0, 0], [255, 255]])
    np.testing.assert_array_equal(prepared.raster.array[:, :, :-1], np.full((2, 2, channels), 50))
    assert (prepared.soft_mask.width, prepared.soft_mask.height) == (1, 2)
    assert dictionary == original


@pytest.mark.parametrize(
    "mask_dictionary",
    [{"Width": 0, "Height": 1}, {"Width": 1, "Height": 1, "Filter": "FlateDecode"}],
)
def test_unusable_soft_mask_does_not_discard_valid_color_samples(mask_dictionary):
    source = ImageSource(
        b"\x32",
        {"Width": 1, "Height": 1, "ColorSpace": "DeviceGray"},
        soft_mask=SoftMask(b"bad", mask_dictionary),
    )
    prepared = images.prepare_image(source)
    assert prepared is not None
    assert prepared.soft_mask is None
    assert prepared.raster.array.tolist() == [[[50]]]


@pytest.mark.parametrize("value", [0, 50, 255])
def test_declared_filter_runs_when_encoded_length_matches_sample_count(value):
    samples = bytes([value] * 11)
    raw = zlib.compress(samples)
    assert len(raw) == len(samples)
    source = ImageSource(
        raw,
        {
            "Width": 11,
            "Height": 1,
            "BitsPerComponent": 8,
            "ColorSpace": "DeviceGray",
            "Filter": "FlateDecode",
        },
    )
    raster = images.decode_image(source)
    assert raster is not None
    assert raster.array.reshape(-1).tolist() == list(samples)
