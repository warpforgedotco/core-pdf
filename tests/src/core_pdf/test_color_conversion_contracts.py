import numpy as np
import pytest

from core_pdf.impl.graphics.color import color_operands_to_srgb, convert_image_data
from core_pdf.impl.graphics.color_spec import parse_color_space
from core_pdf.impl.graphics.image_samples import (
    convert_integer_samples,
    distinct_component_rows,
)


@pytest.mark.parametrize(
    "space",
    [
        "DeviceGray",
        "DeviceRGB",
        "DeviceCMYK",
        ["CalGray", {"WhitePoint": [0.9505, 1, 1.089]}],
        ["CalRGB", {"WhitePoint": [0.9505, 1, 1.089]}],
        ["Lab", {"WhitePoint": [0.9642, 1, 0.8249]}],
        ["ICCBased", {"N": 3, "Alternate": "DeviceRGB"}],
        ["Separation", "Spot", "DeviceGray", lambda tint: 1 - tint],
        ["DeviceN", ["A", "B"], "DeviceRGB", lambda a, b: (1 - a, 1 - b, 1)],
    ],
)
@pytest.mark.parametrize("reverse_decode", [False, True])
def test_equivalent_eight_and_sixteen_bit_components(space, reverse_decode):
    spec = parse_color_space(space)
    count = len(spec.component_ranges)
    samples = np.tile(np.array([0, 64, 128, 255], dtype=np.uint8)[:, None], (1, count))
    dictionary = {"ColorSpace": space, "Width": 4, "Height": 1, "BitsPerComponent": 8}
    if reverse_decode:
        dictionary["Decode"] = [n for low, high in spec.component_ranges for n in (high, low)]
    actual = convert_image_data(samples.reshape(-1), dictionary)
    high = convert_integer_samples(samples.astype(np.uint16) * 257, dictionary)
    assert actual is not None
    np.testing.assert_array_equal(np.asarray(actual).reshape(4, -1), high)


@pytest.mark.parametrize("function", [None, {}, lambda t: (t, t), lambda t: float("nan")])
@pytest.mark.parametrize("tint", [0, 0.5, 1])
def test_unusable_tint_has_same_subtractive_recovery_everywhere(function, tint):
    space = ["Separation", "Spot", "DeviceGray", function]
    vector = color_operands_to_srgb(parse_color_space(space), [tint])
    expected = round((1 - tint) * 255)
    assert vector == (expected / 255,) * 3
    for bits in (8, 16):
        raw = bytes(bits // 8)
        image = convert_image_data(
            raw,
            {
                "ColorSpace": space,
                "Width": 1,
                "Height": 1,
                "BitsPerComponent": bits,
                "Decode": [tint, tint],
            },
        )
        np.testing.assert_array_equal(image, [expected] * 3)


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16])
def test_indexed_decode_selects_palette_before_alternate_conversion(bits):
    space = ["Indexed", "DeviceRGB", 1, b"\xff\x00\x00\x00\xff\x00"]
    raw = bytes(2 if bits == 16 else 1)
    actual = convert_image_data(
        raw,
        {
            "ColorSpace": space,
            "BitsPerComponent": bits,
            "Width": 1,
            "Height": 1,
            "Decode": [1, 0],
        },
    )
    np.testing.assert_array_equal(actual, [0, 255, 0])
    assert color_operands_to_srgb(parse_color_space(space), [1]) == (0, 1, 0)


@pytest.mark.parametrize(
    ("space", "raw"), [("DeviceGray", b"\x80"), ("DeviceRGB", b"\x80\x00\xff")]
)
def test_device_passthrough_preserves_buffer_identity(space, raw):
    assert (
        convert_image_data(
            raw,
            {
                "ColorSpace": space,
                "Width": 1,
                "Height": 1,
                "BitsPerComponent": 8,
            },
        )
        is raw
    )


def test_colour_key_mask_uses_low_sixteen_bits_before_decode():
    result = convert_integer_samples(
        np.array([[256], [257]], dtype=np.uint16),
        {
            "ColorSpace": "DeviceGray",
            "Decode": [1, 0],
            "Mask": [256, 256],
        },
    )
    assert result[:, 0].tolist() == [254, 254]
    assert result[:, 1].tolist() == [0, 255]


def test_soft_mask_precedes_colour_key_mask():
    result = convert_integer_samples(
        np.array([[0]], dtype=np.uint16),
        {
            "ColorSpace": "DeviceGray",
            "Mask": [0, 0],
            "SMask": object(),
        },
    )
    assert result.shape == (1, 1)


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("inks", [(0, 0, 0, 0), (19, 64, 201, 128), (255, 255, 255, 255)])
def test_device_cmyk_vector_and_image_share_profile_and_fallback(monkeypatch, fallback, inks):
    from core_pdf.impl.graphics import device_profiles

    device_profiles.cmyk_byte_tuple_to_srgb.cache_clear()
    try:
        if fallback:
            monkeypatch.setattr(device_profiles, "default_cmyk_transform", lambda: None)
        vector = device_profiles.cmyk_floats_to_srgb(*(ink / 255 for ink in inks))
        image = convert_integer_samples(
            np.array([inks], dtype=np.uint16) * 257, {"ColorSpace": "DeviceCMYK"}
        )
        np.testing.assert_array_equal(image[0], vector)
        if fallback:
            expected = np.rint(255 * (1 - np.array(inks[:3]) / 255) * (1 - inks[3] / 255))
            np.testing.assert_array_equal(vector, expected)
    finally:
        device_profiles.cmyk_byte_tuple_to_srgb.cache_clear()


@pytest.mark.parametrize("components", [1, 2, 3, 4])
@pytest.mark.parametrize("distinct_values", [1, 2, 3, 256])
@pytest.mark.parametrize("rows", [0, 1, 2, 17, 1000])
def test_distinct_component_rows_matches_the_row_sort(rows, distinct_values, components):
    rng = np.random.default_rng(17)
    values = rng.integers(0, distinct_values, (rows, components)).astype(np.float64)
    expected_distinct, expected_inverse = np.unique(values, axis=0, return_inverse=True)
    distinct, inverse = distinct_component_rows(values)
    assert distinct.shape == expected_distinct.shape
    assert np.array_equal(distinct, expected_distinct)
    assert inverse.shape == expected_inverse.shape
    assert np.array_equal(inverse, expected_inverse)
    # The scatter is what the tint path actually depends on.
    assert np.array_equal(distinct[inverse], values)


def test_distinct_component_rows_keeps_the_row_sort_for_wide_spaces():
    values = np.array([[1.0, 2.0], [1.0, 1.0], [1.0, 2.0]])
    distinct, inverse = distinct_component_rows(values)
    assert np.array_equal(distinct, np.array([[1.0, 1.0], [1.0, 2.0]]))
    assert np.array_equal(inverse, np.array([1, 0, 1]))


def test_distinct_component_rows_handles_signed_zero_and_extremes():
    values = np.array([[-0.0], [0.0], [1e308], [1e-308], [-0.0]])
    expected_distinct, expected_inverse = np.unique(values, axis=0, return_inverse=True)
    distinct, inverse = distinct_component_rows(values)
    assert np.array_equal(distinct, expected_distinct)
    assert np.array_equal(inverse, expected_inverse)


def test_distinct_component_rows_collapses_nan_rows():
    # The row sort compares raw bytes and keeps each NaN row separate; the
    # one-dimensional sort collapses them. Both scatter the same tint output
    # over the same pixels, because the tint function is deterministic, so the
    # difference is a saved evaluation rather than a different image. This test
    # pins the collapse so the divergence is a decision, not a surprise.
    values = np.array([[np.nan], [1.0], [np.nan], [2.0]])
    distinct, inverse = distinct_component_rows(values)
    assert distinct.shape == (3, 1)
    assert np.array_equal(inverse, np.array([2, 0, 2, 1]))
    assert np.isnan(distinct[2, 0])
    # Every NaN pixel still selects a NaN entry, so a deterministic tint gives
    # each of them the same colour either way.
    assert np.isnan(distinct[inverse][0, 0])
    assert np.isnan(distinct[inverse][2, 0])
