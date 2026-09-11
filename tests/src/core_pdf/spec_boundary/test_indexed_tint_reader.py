# SPDX-License-Identifier: AGPL-3.0-only
"""Indexed lookup feeds tint inputs before alternate-space output conversion."""

from typing import Any

import imagecodecs
import numpy
import pytest

from core_pdf.impl._impl.graphics.color import color_operands_to_srgb, internal_convert_image_data
from core_pdf.impl._impl.graphics.color_spec import parse_color_space
from core_pdf.impl._impl.graphics.icc_profiles import parse_icc_transform
from core_pdf.impl._impl.graphics.images import prepare_image
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.s_08_graphics.image_spec import ImageSource, SoftMask


def internal_separation() -> list[Any]:
    function = {"FunctionType": 2, "Domain": [0, 1], "C0": [1, 0, 0], "C1": [0, 0, 1], "N": 1}
    return ["Separation", "Spot", "DeviceRGB", function]


def internal_devicen() -> list[Any]:
    function = PdfStream(
        {
            "FunctionType": 0,
            "Domain": [0, 1] * 2,
            "Range": [0, 1] * 3,
            "Size": [2, 2],
            "BitsPerSample": 8,
        },
        bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]),
    )
    return ["DeviceN", ["First", "Second"], "DeviceRGB", function]


def internal_packed(rows: list[list[int]], depth: int) -> bytes:
    if depth in {8, 16}:
        return numpy.asarray(rows, dtype=numpy.uint8 if depth == 8 else ">u2").tobytes()
    result = bytearray()
    for row in rows:
        bits = "".join(f"{value:0{depth}b}" for value in row)
        padding = (8 - len(bits) % 8) % 8
        result.extend(int(bits + "0" * padding, 2).to_bytes((len(bits) + padding) // 8, "big"))
    return bytes(result)


def internal_scalar(
    space: object, index: float, rendering: ColorRendering = DEFAULT_COLOR_RENDERING
) -> tuple[int, ...]:
    result = color_operands_to_srgb(parse_color_space(space), [index], rendering=rendering)
    assert result is not None
    return tuple(round(value * 255) for value in result)


@pytest.mark.parametrize(
    ("index", "entry"), [(-10, 0), (0.49, 0), (0.5, 1), (1.49, 1), (1.5, 2), (2.5, 3), (99, 3)]
)
def test_scalar_palette_indices_round_half_up_and_clamp_before_tint(
    index: float, entry: int
) -> None:
    space = ["Indexed", internal_separation(), 3, bytes([0, 85, 170, 255])]
    expected = [(255, 0, 0), (170, 0, 85), (85, 0, 170), (0, 0, 255)]
    assert internal_scalar(space, index) == expected[entry]


@pytest.mark.parametrize("kind", ["Separation", "DeviceN"])
@pytest.mark.parametrize("depth", [1, 2, 4, 8, 16])
def test_packed_image_rows_decode_indices_then_palette_then_tint(kind: str, depth: int) -> None:
    base = internal_separation() if kind == "Separation" else internal_devicen()
    palette = bytes([0, 255]) if kind == "Separation" else bytes([0, 0, 0, 255])
    space = ["Indexed", base, 1, palette]
    maximum = (1 << depth) - 1
    rows = [[0, maximum, 0], [maximum, 0, maximum]]
    image = prepare_image(
        ImageSource(
            internal_packed(rows, depth),
            {
                "Width": 3,
                "Height": 2,
                "BitsPerComponent": depth,
                "ColorSpace": space,
                "Decode": [1, 0],
            },
        )
    )
    assert image is not None
    red, blue = [255, 0, 0], [0, 0, 255]
    assert image.raster.array.tolist() == [[blue, red, blue], [red, blue, red]]


@pytest.mark.parametrize("depth", [1, 2, 4, 8, 16])
def test_color_key_mask_uses_source_index_before_reversed_decode(depth: int) -> None:
    space = ["Indexed", internal_separation(), 1, bytes([0, 255])]
    maximum = (1 << depth) - 1
    image = prepare_image(
        ImageSource(
            internal_packed([[0, maximum]], depth),
            {
                "Width": 2,
                "Height": 1,
                "BitsPerComponent": depth,
                "ColorSpace": space,
                "Decode": [1, 0],
                "Mask": [maximum, maximum],
            },
        )
    )
    assert image is not None
    assert image.raster.array.tolist() == [[[0, 0, 255, 255], [255, 0, 0, 0]]]


@pytest.mark.parametrize("depth", [1, 2, 4, 8, 16])
def test_soft_mask_overrides_palette_color_key_mask(depth: int) -> None:
    space = ["Indexed", internal_separation(), 1, bytes([0, 255])]
    image = prepare_image(
        ImageSource(
            internal_packed([[0, 1]], depth),
            {
                "Width": 2,
                "Height": 1,
                "BitsPerComponent": depth,
                "ColorSpace": space,
                "Mask": [0, 1],
            },
            soft_mask=SoftMask(
                bytes([255, 0]),
                {"Width": 2, "Height": 1, "BitsPerComponent": 8, "ColorSpace": "DeviceGray"},
            ),
        )
    )
    assert image is not None
    assert image.raster.array.tolist() == [[[255, 0, 0, 255], [0, 0, 255, 0]]]


def test_ordinary_devicen_palette_uses_each_component_in_order() -> None:
    space = ["Indexed", internal_devicen(), 3, bytes([0, 0, 255, 0, 0, 255, 255, 255])]
    expected = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
    assert [internal_scalar(space, index) for index in range(4)] == expected
    for depth in (2, 4, 8, 16):
        image = prepare_image(
            ImageSource(
                internal_packed([[0, 1, 2, 3]], depth),
                {"Width": 4, "Height": 1, "BitsPerComponent": depth, "ColorSpace": space},
            )
        )
        assert image is not None
        assert image.raster.array.tolist() == [[list(color) for color in expected]]


@pytest.mark.parametrize("index", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_palette_index_keeps_scalar_recovery(index: float) -> None:
    space = ["Indexed", internal_separation(), 1, bytes([0, 255])]
    assert color_operands_to_srgb(parse_color_space(space), [index]) is None


@pytest.mark.parametrize("malformation", ["palette", "function"])
def test_malformed_palette_or_tint_keeps_existing_reader_recovery(malformation: str) -> None:
    base = internal_separation()
    if malformation == "function":
        base[3] = {"FunctionType": 99}
    space = ["Indexed", base, 1, b"" if malformation == "palette" else bytes([0, 255])]
    assert color_operands_to_srgb(parse_color_space(space), [0]) is None
    # The existing malformed 8-bit image recovery preserves unambiguous raw gray.
    image = prepare_image(
        ImageSource(b"\x00", {"Width": 1, "Height": 1, "BitsPerComponent": 8, "ColorSpace": space})
    )
    assert image is not None
    assert image.raster.array.tolist() == [[[0]]]


@pytest.mark.parametrize(
    "rendering",
    [
        DEFAULT_COLOR_RENDERING,
        ColorRendering(black_point_compensation="OFF"),
        ColorRendering("AbsoluteColorimetric", "ON"),
    ],
)
def test_indexed_tint_retains_embedded_icc_and_rendering_settings(
    rendering: ColorRendering,
) -> None:
    profile = bytes(imagecodecs.cms_profile("linearrgb"))
    icc = ["ICCBased", PdfStream({"N": 3}, profile)]
    function = {
        "FunctionType": 2,
        "Domain": [0, 1],
        "C0": [0.25, 0.5, 0.75],
        "C1": [0.25, 0.5, 0.75],
        "N": 1,
    }
    space = ["Indexed", ["Separation", "Spot", icc, function], 0, b"\x00"]
    expected = parse_icc_transform(profile).apply_uint16(
        numpy.array([[16384, 32768, 49151]], dtype=numpy.uint16), rendering=rendering
    )[0]
    assert internal_scalar(space, 0, rendering) == tuple(expected)
    image = prepare_image(
        ImageSource(
            b"\x00",
            {"Width": 1, "Height": 1, "BitsPerComponent": 8, "ColorSpace": space},
            color_rendering=rendering,
        )
    )
    assert image is not None
    numpy.testing.assert_array_equal(image.raster.array[0, 0], expected)


def test_device_palette_defaults_remain_unchanged() -> None:
    for base, palette, expected in (
        ("DeviceGray", bytes([0, 255]), [0, 0, 0, 255, 255, 255]),
        ("DeviceRGB", bytes([255, 0, 0, 0, 0, 255]), [255, 0, 0, 0, 0, 255]),
    ):
        space = ["Indexed", base, 1, palette]
        result = internal_convert_image_data(
            bytes([0, 1]), {"Width": 2, "Height": 1, "BitsPerComponent": 8, "ColorSpace": space}
        )
        assert result is not None
        assert numpy.frombuffer(result, dtype=numpy.uint8).tolist() == expected
