# SPDX-License-Identifier: AGPL-3.0-only
"""Colour-operator conformance rules from ISO 32000-1 clause 8.6.

`sc`/`scn` operands are not device components in every space: 8.6.6.3 makes an
Indexed operand a palette index, and 8.6.6.4 makes a Separation/DeviceN operand
a subtractive tint. Clamping either to 0..1 and painting it directly rendered a
spot colour as an inverted grey and ignored the palette entirely.
"""

from __future__ import annotations

import pytest

from core_pdf.impl.spec.s_08_graphics.color import color_operands_to_srgb
from core_pdf.impl.spec.s_08_graphics.color_spec import color_spec_from_value
from tests.helpers.pdf_bytes import assemble_pdf, open_pdf, stream_obj

# Tint 0 is the lightest colour, tint 1 the darkest -- here white to green.
SPOT_GREEN = [
    "Separation",
    "SpotGreen",
    "DeviceRGB",
    {"FunctionType": 2, "Domain": [0, 1], "C0": [1, 1, 1], "C1": [0, 0.5, 0], "N": 1},
]
PALETTE = ["Indexed", "DeviceRGB", 2, bytes([255, 0, 0, 0, 255, 0, 0, 0, 255])]


@pytest.mark.parametrize(
    ("tint", "expected"),
    [(0.0, (1.0, 1.0, 1.0)), (1.0, (0.0, 0.502, 0.0))],
)
def test_separation_tint_runs_through_the_tint_transform(
    tint: float, expected: tuple[float, float, float]
) -> None:
    """8.6.6.4: "a tint value of 0.0 denotes the lightest colour that can be
    achieved with the given colorant, and 1.0 is the darkest"."""
    spec = color_spec_from_value(SPOT_GREEN)

    result = color_operands_to_srgb(spec, [tint])

    assert result == pytest.approx(expected, abs=0.004)


def test_separation_without_a_tint_transform_stays_subtractive() -> None:
    """Even with no usable transform, 8.6.6.4 fixes the direction of the ink."""
    spec = color_spec_from_value(["Separation", "Spot", "DeviceRGB", None])

    assert color_operands_to_srgb(spec, [0.0]) == pytest.approx((1.0, 1.0, 1.0))
    assert color_operands_to_srgb(spec, [1.0]) == pytest.approx((0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, (1.0, 0.0, 0.0)), (1, (0.0, 1.0, 0.0)), (2, (0.0, 0.0, 1.0))],
)
def test_indexed_operand_selects_the_palette_entry(
    index: int, expected: tuple[float, float, float]
) -> None:
    """8.6.6.3, EXAMPLE 2: "123 sc" selects the colour an image sample of 123 does."""
    spec = color_spec_from_value(PALETTE)

    assert color_operands_to_srgb(spec, [index]) == pytest.approx(expected)


@pytest.mark.parametrize("index", [-4, 3, 99])
def test_indexed_operand_outside_the_range_is_clamped(index: int) -> None:
    """8.6.6.3: "if it is outside the range 0 to hival, it shall be adjusted to
    the nearest value within that range"."""
    spec = color_spec_from_value(PALETTE)
    expected = (1.0, 0.0, 0.0) if index < 0 else (0.0, 0.0, 1.0)

    assert color_operands_to_srgb(spec, [index]) == pytest.approx(expected)


def test_indexed_real_operand_is_rounded() -> None:
    """8.6.6.3: "If the value is a real number, it shall be rounded to the nearest integer"."""
    spec = color_spec_from_value(PALETTE)

    assert color_operands_to_srgb(spec, [0.6]) == pytest.approx((0.0, 1.0, 0.0))


@pytest.mark.parametrize("name", ["DeviceGray", "DeviceRGB", "DeviceCMYK"])
def test_device_spaces_are_left_to_their_own_components(name: str) -> None:
    """Device operands already are their components; nothing to resolve."""
    assert color_operands_to_srgb(color_spec_from_value(name), [0.2, 0.4, 0.6]) is None


def internal_spot_pdf(content: bytes) -> bytes:
    """A one-page PDF whose only resource is the SpotGreen Separation space."""
    tint = b"<< /FunctionType 2 /Domain [0 1] /C0 [1 1 1] /C1 [0 0.5 0] /N 1 >>"
    return assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 16 16] /Contents 4 0 R "
            b"/Resources << /ColorSpace << /CS0 [/Separation /SpotGreen /DeviceRGB 5 0 R] >> >> >>",
            stream_obj(content),
            tint,
        ],
        version="1.7",
    )


def internal_first_fill(content: bytes) -> tuple[float, ...] | None:
    with open_pdf(internal_spot_pdf(content)) as document:
        drawings = document.pages[0].get_drawings()
    return drawings[0].fill if drawings else None


def test_separation_fill_reaches_the_page_as_the_transformed_colour() -> None:
    fill = internal_first_fill(b"/CS0 cs 1 scn 0 0 16 16 re f")

    assert fill == pytest.approx((0.0, 0.502, 0.0), abs=0.004)


def test_colour_space_is_restored_by_q_and_Q() -> None:
    """The resolved space must travel with its name through the graphics state.

    Inside q/Q the space is switched to DeviceRGB; after Q the Separation space
    is current again, so the tint must be transformed rather than painted raw.
    """
    content = b"/CS0 cs q /DeviceRGB cs 1 0 0 rg Q 1 scn 0 0 16 16 re f"

    assert internal_first_fill(content) == pytest.approx((0.0, 0.502, 0.0), abs=0.004)
