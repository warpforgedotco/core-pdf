import numpy as np
import pytest

from core_pdf.impl._impl.graphics import images
from core_pdf.impl._impl.graphics.image_models import DecodedImage


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize(("space", "channels"), [("DeviceGray", 1), ("DeviceRGB", 3)])
@pytest.mark.parametrize("selector", [0, 1, 2])
@pytest.mark.parametrize("level", [0, 1, 2])
def test_jpx_opacity_selector_distinguishes_ignored_straight_and_premultiplied(
    dtype, space, channels, selector, level
):
    maximum = np.iinfo(dtype).max
    value = [0, (maximum + 1) // 2, maximum][level]
    samples = DecodedImage(np.array([[[value] * (channels + 1)]], dtype=dtype), "jpx")
    dictionary = {
        "Width": 1,
        "Height": 1,
        "Filter": "JPXDecode",
        "ColorSpace": space,
        "SMaskInData": selector,
    }
    result = images.internal_canonical_image_array(samples, dictionary)
    assert result is not None
    array, count = result
    ordinary = round(value * 255 / maximum)
    color = 255 if selector == 2 and value else ordinary
    expected = [color] * channels + ([ordinary] if selector else [])
    assert count == len(expected)
    assert array.tolist() == expected
    assert samples.array.reshape(-1).tolist() == [value] * (channels + 1)


@pytest.mark.parametrize(
    ("shape", "space", "selector"),
    [
        ((1, 1, 2), "DeviceRGB", 1),
        ((1, 1, 3), "DeviceRGB", 1),
        ((1, 1, 5), "DeviceRGB", 0),
        ((1, 1, 2), "invalid", 1),
    ],
)
def test_inconsistent_jpx_component_layout_is_rejected(shape, space, selector):
    result = images.internal_canonical_image_array(
        DecodedImage(np.zeros(shape, dtype=np.uint8), "jpx"),
        {"Filter": "JPXDecode", "ColorSpace": space, "SMaskInData": selector},
    )
    assert result is None


@pytest.mark.parametrize("selector", [-1, 3, None, "invalid"])
def test_invalid_jpx_opacity_selector_uses_reader_default(selector):
    sample = DecodedImage(np.array([[[20, 40]]], dtype=np.uint8), "jpx")
    result = images.internal_canonical_image_array(
        sample, {"Filter": "JPXDecode", "ColorSpace": "DeviceGray", "SMaskInData": selector}
    )
    assert result is not None
    array, count = result
    assert count == 1
    assert array.tolist() == [20]


@pytest.mark.parametrize(
    ("shape", "space"),
    [((1, 2), "DeviceGray"), ((1, 2, 1), "DeviceGray"), ((1, 2, 3), "DeviceRGB")],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_native_image_precision_normalizes_endpoints(shape, space, dtype):
    maximum = np.iinfo(dtype).max
    sample = np.resize(np.array([0, maximum], dtype=dtype), shape)
    result = images.internal_canonical_image_array(
        DecodedImage(sample, "jpeg"), {"Width": 2, "Height": 1, "ColorSpace": space}
    )
    assert result is not None
    array, channels = result
    assert channels == (1 if space == "DeviceGray" else 3)
    np.testing.assert_array_equal(array, np.where(sample.reshape(-1) == 0, 0, 255))


@pytest.mark.parametrize("version", [None, "1.7", "2.0", "3.0"])
@pytest.mark.parametrize("explicit_space", [False, True])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_jpx_decode_obeys_document_version_and_explicit_color_space(version, explicit_space, dtype):
    from core_pdf_spec.standards import PdfVersion, SemanticContext

    maximum = np.iinfo(dtype).max
    sample = DecodedImage(np.array([[0, maximum]], dtype=dtype), "jpx")
    dictionary = {"Width": 2, "Height": 1, "Filter": "JPXDecode", "Decode": [1, 0]}
    if explicit_space:
        dictionary["ColorSpace"] = "DeviceGray"
    context = SemanticContext(PdfVersion.parse(version)) if version else None
    result = images.internal_canonical_image_array(sample, dictionary, semantic_context=context)
    assert result is not None
    array, channels = result
    assert channels == 1
    assert array.tolist() == ([255, 0] if version == "2.0" and explicit_space else [0, 255])


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.uint32])
def test_native_decoded_samples_reject_unsupported_precision(dtype):
    with pytest.raises(ValueError, match="must be uint8 or uint16"):
        DecodedImage(np.zeros((1, 1), dtype=dtype), "jpx")


@pytest.mark.parametrize("shape", [(1,), (1, 1, 1, 1)])
def test_native_decoded_samples_reject_unsupported_dimensions(shape):
    with pytest.raises(ValueError, match="two or three dimensions"):
        DecodedImage(np.zeros(shape, dtype=np.uint8), "jpx")


def test_native_decoded_samples_require_contiguous_storage():
    with pytest.raises(ValueError, match="must be C-contiguous"):
        DecodedImage(np.zeros((2, 2), dtype=np.uint8)[:, ::-1], "jpx")
