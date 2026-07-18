import pytest

from core_jpeg.impl.codecs.dct.color import (
    clamp_u8,
    cmyk_to_rgb_channels,
    inverted_cmyk_to_rgb_channels,
    ycbcr_to_rgb_channels,
)
from core_jpeg.impl.graphics.color_math import (
    adapt_d50_to_d65,
    lab_to_xyz,
    linear_to_srgb,
    xyz_to_srgb,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1, 0), (0, 0), (127, 127), (255, 255), (256, 255)],
)
def test_clamp_u8(value: int, expected: int) -> None:
    assert clamp_u8(value) == expected


def test_neutral_ycbcr_is_gray() -> None:
    assert ycbcr_to_rgb_channels(128, 128, 128) == (128, 128, 128)


def test_cmyk_conversions_handle_white_and_black() -> None:
    assert cmyk_to_rgb_channels(0, 0, 0, 0) == (255, 255, 255)
    assert cmyk_to_rgb_channels(0, 0, 0, 255) == (0, 0, 0)
    assert inverted_cmyk_to_rgb_channels(255, 255, 255, 255) == (255, 255, 255)


def test_linear_to_srgb_uses_both_transfer_segments() -> None:
    assert linear_to_srgb(0.0) == 0.0
    assert linear_to_srgb(0.0031308) == pytest.approx(0.040449936)
    assert linear_to_srgb(1.0) == pytest.approx(1.0)


def test_lab_white_maps_to_reference_white() -> None:
    white = (0.9642, 1.0, 0.8249)
    assert lab_to_xyz(100.0, 0.0, 0.0, white) == pytest.approx(white)


def test_d50_white_adapts_to_d65_and_srgb_white() -> None:
    d65 = adapt_d50_to_d65(0.9642, 1.0, 0.8249)

    assert d65 == pytest.approx((0.95047, 1.0, 1.08883), abs=3e-4)
    assert xyz_to_srgb(*d65) == pytest.approx((1.0, 1.0, 1.0), abs=5e-4)
