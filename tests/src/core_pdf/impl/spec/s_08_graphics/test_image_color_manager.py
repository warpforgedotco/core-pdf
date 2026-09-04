# SPDX-License-Identifier: AGPL-3.0-only
"""ImageColorManager conversions: CMYK, sampled Separation and Indexed samples."""

from __future__ import annotations

import numpy

from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_08_graphics.color import (
    ImageColorManager,
    internal_sampled_separation_rgb_lut,
    internal_tint_operands_to_srgb,
)
from core_pdf.impl.spec.s_08_graphics.color_spec import ImageColorSpec


def test_cmyk_conversion_handles_process_inks_and_black() -> None:
    # Solid inks come out as the colours a SWOP press actually prints, not as
    # the saturated (0, 255, 255) / (255, 0, 255) / (255, 255, 0) the naive
    # 255*(1-ink)*(1-black) formula produces. 100% K is the profile's black
    # after black point compensation, which is a near-neutral very dark grey.
    samples = bytes.fromhex("00000000 ff000000 00ff0000 0000ff00 0a141e28 000000ff")
    expected = bytes.fromhex("ffffff 00aef0 ec0a8d fff300 d6cdc6 292728")

    numpy.testing.assert_array_equal(
        ImageColorManager.convert_cmyk(samples),
        numpy.frombuffer(expected, dtype=numpy.uint8),
    )


def test_cmyk_conversion_is_monotonic_in_ink_and_black() -> None:
    """Properties any usable CMYK profile has, independent of which one ships."""
    no_ink = ImageColorManager.convert_cmyk(bytes(4))
    assert tuple(no_ink) == (255, 255, 255)

    for channel in range(4):
        ramp = bytearray()
        for level in (0, 64, 128, 192, 255):
            ink = [0, 0, 0, 0]
            ink[channel] = level
            ramp.extend(ink)
        luminance = ImageColorManager.convert_cmyk(bytes(ramp)).reshape(-1, 3).sum(axis=1)
        assert list(luminance) == sorted(luminance, reverse=True), (
            f"channel {channel} does not darken monotonically: {luminance}"
        )

    black = ImageColorManager.convert_cmyk(bytes.fromhex("000000ff")).reshape(3)
    assert black.max() < 64
    assert int(black.max()) - int(black.min()) <= 8


def test_sampled_separation_conversion_builds_exact_readonly_rgb_lut() -> None:
    tint_function = PdfStream(
        {
            "FunctionType": 0,
            "BitsPerSample": 8,
            "Size": [256],
            "Domain": [0, 1],
            "Range": [0, 1],
        },
        bytes(range(256)),
    )
    spec = ImageColorSpec(
        kind="Separation",
        params={},
        bits_per_component=8,
        alt="DeviceGray",
        tint_fn=tint_function,
    )
    samples = bytes((0, 64, 128, 192, 255, 64, 0))

    first = ImageColorManager.convert_separation(samples, spec)
    second = ImageColorManager.convert_separation(samples, spec)
    expected = numpy.repeat(numpy.frombuffer(samples, dtype=numpy.uint8), 3)

    numpy.testing.assert_array_equal(first, expected)
    numpy.testing.assert_array_equal(second, expected)
    lut = internal_sampled_separation_rgb_lut(tint_function, "DeviceGray")
    expected_lut = numpy.repeat(numpy.arange(256, dtype=numpy.uint8)[:, None], 3, axis=1)
    numpy.testing.assert_array_equal(lut, expected_lut)
    assert not lut.flags.writeable

    # An equivalent function built from fresh bytes must produce the same table.
    twin = PdfStream(dict(tint_function.dictionary), bytes(range(256)))
    numpy.testing.assert_array_equal(
        internal_sampled_separation_rgb_lut(twin, "DeviceGray"),
        lut,
    )


def test_indexed_conversion_uses_rgb_lookup_entries() -> None:
    lookup = bytes(value for index in range(4) for value in (index, index + 10, index + 20))
    spec = ImageColorSpec("Indexed", {}, base="DeviceRGB", hival=3, lookup=lookup)

    numpy.testing.assert_array_equal(
        ImageColorManager.convert_indexed(bytes((3, 0, 2)), spec),
        numpy.asarray((3, 13, 23, 0, 10, 20, 2, 12, 22), dtype=numpy.uint8),
    )


def test_separation_image_matches_the_same_tint_used_as_a_fill_colour() -> None:
    """A FunctionType 2 or 3 tint must convert the same in both directions.

    The image path only understood a sampled PdfStream, so a tint transform
    given as a plain dictionary -- the simplest form ISO 32000-1 allows -- fell
    through to the identity, failed the component-count check, and raised. That
    ValueError was then absorbed by image_decode as if it were a malformed
    ICCBased reference, so the image rendered with the tint transform silently
    dropped while the identical colour space painted correctly as a fill.
    """
    for tint, alt in (
        (
            {
                "FunctionType": 2,
                "N": 1,
                "C0": [0.0, 0.0, 0.0, 0.0],
                "C1": [0.0, 0.9, 0.9, 0.1],
            },
            "DeviceCMYK",
        ),
        ({"FunctionType": 2, "N": 1, "C0": [1.0, 0.0, 0.0], "C1": [0.0, 0.0, 1.0]}, "DeviceRGB"),
        (
            {
                "FunctionType": 3,
                "Bounds": [0.5],
                "Encode": [0, 1, 0, 1],
                "Functions": [
                    {"FunctionType": 2, "N": 1, "C0": [1, 0, 0], "C1": [0, 1, 0]},
                    {"FunctionType": 2, "N": 1, "C0": [0, 1, 0], "C1": [0, 0, 1]},
                ],
            },
            "DeviceRGB",
        ),
    ):
        spec = ImageColorSpec(kind="Separation", params={}, alt=alt, channels=1, tint_fn=tint)
        for byte in (0, 64, 128, 255):
            operand = internal_tint_operands_to_srgb(spec, [byte / 255.0])
            assert operand is not None
            image = numpy.asarray(ImageColorManager.convert_separation(bytes([byte]), spec))[:3]
            assert tuple(image) == tuple(round(value * 255.0) for value in operand)
