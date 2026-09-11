# SPDX-License-Identifier: AGPL-3.0-only
"""Calibrated colour controls reach physical pixels, including their black endpoint."""

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.graphics.calibrated_colors import calibrated_xyz_to_srgb
from core_pdf.impl._impl.graphics.images import prepare_image
from core_pdf_spec.s_08_graphics.color_rendering import ColorRendering, RenderingIntent
from core_pdf_spec.s_08_graphics.image_spec import ImageSource


@pytest.mark.parametrize("intent", ["RelativeColorimetric", "Perceptual", "Saturation"])
def test_calibrated_black_point_control_changes_shadows_and_preserves_white(
    intent: RenderingIntent,
) -> None:
    white = (0.9642, 1.0, 0.8249)
    black = (0.09642, 0.1, 0.08249)
    values = numpy.array([black, [0.24105, 0.25, 0.206225], white])
    enabled = calibrated_xyz_to_srgb(values, white, black, ColorRendering(intent, "ON"))
    disabled = calibrated_xyz_to_srgb(values, white, black, ColorRendering(intent, "OFF"))
    # The selected sRGB profile has a zero black endpoint. Its independent sRGB
    # transfer curve yields 89 at 10% luminance and 137 at 25%, without BPC.
    numpy.testing.assert_allclose(enabled, [[0] * 3, [113] * 3, [255] * 3], atol=1)
    numpy.testing.assert_allclose(disabled, [[89] * 3, [137] * 3, [255] * 3], atol=1)


def test_absolute_colorimetric_ignores_black_point_compensation() -> None:
    # ISO 32000-2, 8.6.5.9: AbsoluteColorimetric ignores UseBlackPtComp.
    white = (0.9642, 1.0, 0.8249)
    black = (0.09642, 0.1, 0.08249)
    values = numpy.array([black, white])
    enabled = calibrated_xyz_to_srgb(
        values, white, black, ColorRendering("AbsoluteColorimetric", "ON")
    )
    disabled = calibrated_xyz_to_srgb(
        values, white, black, ColorRendering("AbsoluteColorimetric", "OFF")
    )
    numpy.testing.assert_array_equal(enabled, disabled)
    numpy.testing.assert_allclose(enabled, [[89] * 3, [255] * 3], atol=1)


@pytest.mark.parametrize("bits", [8, 16])
def test_calibrated_image_samples_honor_state_and_local_intent(bits: int) -> None:
    maximum = 2**bits - 1
    samples = numpy.array([0, maximum // 3, maximum], dtype="u1" if bits == 8 else ">u2")
    dictionary = {
        "Width": 3,
        "Height": 1,
        "BitsPerComponent": bits,
        "ColorSpace": [
            "CalGray",
            {"WhitePoint": [0.9642, 1, 0.8249], "BlackPoint": [0.09642, 0.1, 0.08249]},
        ],
    }
    enabled = prepare_image(
        ImageSource(
            samples.tobytes(),
            dictionary,
            color_rendering=ColorRendering("RelativeColorimetric", "ON"),
        )
    )
    absolute = prepare_image(
        ImageSource(
            samples.tobytes(),
            {**dictionary, "Intent": "AbsoluteColorimetric"},
            color_rendering=ColorRendering("RelativeColorimetric", "ON"),
        )
    )
    assert enabled is not None
    assert absolute is not None
    # Encoded thirds are exact at both depths. Independent sRGB transfer values
    # at luminance 1/3 and its endpoint-compensated value 7/27 are 156 and 139.
    numpy.testing.assert_allclose(enabled.raster.array[0], [[0] * 3, [139] * 3, [255] * 3], atol=1)
    numpy.testing.assert_allclose(absolute.raster.array[0], [[0] * 3, [156] * 3, [255] * 3], atol=1)


def internal_document(space: bytes, dark: bytes) -> bytes:
    content = (
        b"/C cs " + dark + b" sc /On gs 0 0 10 10 re f "
        b"/Off gs 10 0 10 10 re f /On gs /AbsoluteColorimetric ri 20 0 10 10 re f "
        b"/RelativeColorimetric ri 30 0 10 10 re f"
    )
    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 40 10] /Resources << "
        b"/ColorSpace << /C " + space + b" >> /ExtGState << "
        b"/On << /UseBlackPtComp /ON >> /Off << /UseBlackPtComp /OFF >> >> >> "
        b"/Contents 4 0 R >>",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    output = b"%PDF-2.0\n"
    offsets = [0]
    for number, body in enumerate(bodies, 1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(output)
    output += b"xref\n0 5\n0000000000 65535 f \n"
    output += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    return output + f"trailer << /Size 5 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()


@pytest.mark.parametrize(
    ("space", "dark"),
    [
        (b"CalGray", b"0.1"),
        (b"CalRGB", b"0.09642 0.1 0.08249"),
        (b"Lab", b"37.8424304699 0 0"),
    ],
)
def test_document_calibrated_paint_uses_current_controls(space: bytes, dark: bytes) -> None:
    # Paint-time conversion (ISO 32000-2, 11.7.5.3) applies even though the colour
    # components were set before all four paints. Absolute does not erase stored ON.
    color_space = (
        b"[" + space + b" << /WhitePoint [.9642 1 .8249] /BlackPoint [.09642 .1 .08249] >>]"
    )
    with PdfDocument(internal_document(color_space, dark)) as document:
        pixels = document.pages[0].render().rasterize().array()
    numpy.testing.assert_allclose(
        pixels[5, [5, 15, 25, 35], :3], [[0] * 3, [89] * 3, [89] * 3, [0] * 3], atol=1
    )
