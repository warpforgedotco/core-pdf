# SPDX-License-Identifier: AGPL-3.0-only
"""Paint-time intent and PDF 2.0 BPC reach real selected ICC output."""

from typing import Any

import imagecodecs
import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.graphics.color import color_operands_to_srgb
from core_pdf.impl._impl.graphics.color_spec import parse_color_space
from core_pdf.impl._impl.graphics.device_profiles import cmyk_floats_to_srgb, default_cmyk_transform
from core_pdf.impl._impl.graphics.icc_profiles import IccTransform
from core_pdf.impl._impl.graphics.images import prepare_image
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.color_rendering import (
    DEFAULT_COLOR_RENDERING,
    ColorRendering,
    parse_rendering_intent,
)
from core_pdf_spec.s_08_graphics.image_spec import ImageSource

OFF = ColorRendering(black_point_compensation="OFF")
ON = ColorRendering(black_point_compensation="ON")
ABSOLUTE = ColorRendering("AbsoluteColorimetric", "ON")
SETTINGS = (OFF, ON, OFF, ABSOLUTE)


def internal_transform() -> IccTransform:
    transform = default_cmyk_transform()
    assert transform is not None
    return transform


def internal_expected(rendering: ColorRendering) -> tuple[int, int, int]:
    values = internal_transform().apply_uint16(
        numpy.asarray([[0, 0, 0, 65535]], dtype=numpy.uint16), rendering=rendering
    )[0]
    return int(values[0]), int(values[1]), int(values[2])


def internal_stream(raw: bytes, entries: bytes = b"") -> bytes:
    return b"<< " + entries + f" /Length {len(raw)} >>\nstream\n".encode() + raw + b"\nendstream"


def internal_document(
    content: bytes, resources: bytes = b"", extra: tuple[bytes, ...] = ()
) -> bytes:
    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 40 10] /Resources << "
        b"/ColorSpace << /C [/ICCBased 5 0 R] >> "
        b"/ExtGState << /Off << /UseBlackPtComp /OFF >> /On << /UseBlackPtComp /ON >> "
        b"/Abs << /RI /AbsoluteColorimetric /UseBlackPtComp /ON >> >> "
        + resources
        + b" >> /Contents 4 0 R >>",
        internal_stream(content),
        internal_stream(internal_transform().profile, b"/N 4 /Alternate /DeviceCMYK"),
        *extra,
    ]
    result = b"%PDF-2.0\n"
    offsets = [0]
    for number, body in enumerate(bodies, 1):
        offsets.append(len(result))
        result += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(result)
    result += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    result += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    return (
        result
        + f"trailer << /Root 1 0 R /Size {len(offsets)} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )


def internal_pixels(content: bytes, resources: bytes = b"", extra: tuple[bytes, ...] = ()) -> list:
    with PdfDocument(internal_document(content, resources, extra)) as document:
        pixels = document.pages[0].render().rasterize().array()
        return [tuple(int(value) for value in pixels[5, x, :3]) for x in (5, 15, 25, 35)]


def internal_repeated_paint(operation: bytes) -> bytes:
    return b" ".join(
        f"q /{name} gs 1 0 0 1 {index * 10} 0 cm ".encode() + operation + b" Q"
        for index, name in enumerate(("Off", "On", "Off", "Abs"))
    )


def test_actual_icc_dark_pixels_and_scalar_cache_are_isolated() -> None:
    assert internal_expected(OFF) != internal_expected(ON)
    assert internal_expected(DEFAULT_COLOR_RENDERING) == internal_expected(ON)
    assert internal_expected(ABSOLUTE) == internal_expected(
        ColorRendering("AbsoluteColorimetric", "OFF")
    )
    for rendering in (*SETTINGS, ON, DEFAULT_COLOR_RENDERING):
        assert cmyk_floats_to_srgb(0, 0, 0, 1, rendering=rendering) == internal_expected(rendering)


@pytest.mark.parametrize("depth", [8, 16])
@pytest.mark.parametrize("rows", [1, 5000], ids=["single", "deduplicated"])
@pytest.mark.parametrize(
    ("intent", "code"),
    [
        ("Perceptual", 0),
        ("RelativeColorimetric", 1),
        ("Saturation", 2),
        ("AbsoluteColorimetric", 3),
    ],
)
def test_icc_scalar_array_and_deduplication_use_selected_options(
    monkeypatch: pytest.MonkeyPatch, depth: int, rows: int, intent: str, code: int
) -> None:
    calls: list[tuple[int, int]] = []
    original = imagecodecs.cms_transform

    def record(*args: Any, **kwargs: Any) -> Any:
        calls.append((kwargs["intent"], kwargs["flags"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(imagecodecs, "cms_transform", record)
    params = ColorRendering(parse_rendering_intent(intent), "ON")
    maximum = (1 << depth) - 1
    values = numpy.tile([0, 0, 0, maximum], (rows, 1))
    result = (
        internal_transform().apply_uint8(values.astype(numpy.uint8), rendering=params)
        if depth == 8
        else internal_transform().apply_uint16(values.astype(numpy.uint16), rendering=params)
    )
    numpy.testing.assert_array_equal(result, numpy.tile(result[0], (rows, 1)))
    assert calls == [(code, 0x0100 if code == 3 else 0x2100)]


def test_path_color_uses_paint_time_parameters_and_q_Q_restores_state() -> None:
    pixels = internal_pixels(
        b"/C cs 0 0 0 1 scn /Off gs 0 0 10 10 re f "
        b"q /On gs 10 0 10 10 re f Q 20 0 10 10 re f /Abs gs 30 0 10 10 re f"
    )
    assert pixels == [internal_expected(value) for value in SETTINGS]


def test_stroke_colors_are_converted_at_paint_time() -> None:
    content = b"/C CS 0 0 0 1 SCN 10 w " + internal_repeated_paint(b"0 5 m 10 5 l S")
    assert internal_pixels(content) == [internal_expected(value) for value in SETTINGS]


@pytest.mark.parametrize("mode", [0, 1], ids=["fill", "stroke"])
def test_text_glyphs_use_paint_time_parameters(mode: int) -> None:
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    content = b"/C cs /C CS 0 0 0 1 scn 0 0 0 1 SCN 0.8 w " + internal_repeated_paint(
        f"BT /F 10 Tf {mode} Tr 1 0 0 1 0 1 Tm (H) Tj ET".encode()
    )
    with PdfDocument(internal_document(content, b"/Font << /F 6 0 R >>", (font,))) as document:
        pixels = document.pages[0].render().rasterize().array()
    for index, setting in enumerate(SETTINGS):
        cell = pixels[:, index * 10 : (index + 1) * 10, :3]
        assert numpy.any(numpy.all(cell == internal_expected(setting), axis=2))


def test_scalar_capture_cache_reuses_components_but_keeps_state_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core_pdf.impl._impl.capture import recording

    original = recording.color_operands_to_srgb
    calls: list[ColorRendering] = []

    def record(*args: Any, **kwargs: Any) -> Any:
        if args[0].kind == "ICCBased":
            calls.append(kwargs["rendering"])
        return original(*args, **kwargs)

    monkeypatch.setattr(recording, "color_operands_to_srgb", record)
    content = b"/C cs 0 0 0 1 scn " + internal_repeated_paint(b"0 0 10 10 re f " * 20)
    assert internal_pixels(content) == [internal_expected(value) for value in SETTINGS]
    assert calls == [OFF, ON, ABSOLUTE]


def test_inline_images_and_stencils_inherit_parameters() -> None:
    inline = b"10 0 0 10 0 0 cm BI /W 1 /H 1 /BPC 8 /CS /C ID " + bytes([0, 0, 0, 255]) + b" EI"
    assert internal_pixels(internal_repeated_paint(inline)) == [
        internal_expected(value) for value in SETTINGS
    ]
    stencil = internal_stream(
        b"\x00",
        b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 1 "
        b"/ImageMask true /Intent /AbsoluteColorimetric",
    )
    content = b"/C cs 0 0 0 1 scn " + internal_repeated_paint(b"10 0 0 10 0 0 cm /Im Do")
    assert internal_pixels(content, b"/XObject << /Im 6 0 R >>", (stencil,)) == [
        internal_expected(value) for value in SETTINGS
    ]


@pytest.mark.parametrize("group", [False, True])
def test_reused_form_inherits_paint_parameters_and_restores_local_changes(group: bool) -> None:
    form = internal_stream(
        b"/C cs 0 0 0 1 scn 0 0 10 10 re f /Perceptual ri",
        b"/Type /XObject /Subtype /Form /BBox [0 0 10 10] "
        b"/Resources << /ColorSpace << /C [/ICCBased 5 0 R] >> >> "
        + (b"/Group << /S /Transparency /I true >>" if group else b""),
    )
    pixels = internal_pixels(internal_repeated_paint(b"/F Do"), b"/XObject << /F 6 0 R >>", (form,))
    assert pixels == [internal_expected(value) for value in SETTINGS]


@pytest.mark.parametrize("depth", [8, 16])
@pytest.mark.parametrize("override", [False, True])
def test_reused_image_uses_paint_state_and_its_intent_override(depth: int, override: bool) -> None:
    raw = bytes([0, 0, 0, 255]) if depth == 8 else bytes.fromhex("0000 0000 0000 ffff")
    image = internal_stream(
        raw,
        f"/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent {depth} ".encode()
        + b"/ColorSpace [/ICCBased 5 0 R] "
        + (b"/Intent /AbsoluteColorimetric" if override else b""),
    )
    pixels = internal_pixels(
        internal_repeated_paint(b"10 0 0 10 0 0 cm /Im Do"),
        b"/XObject << /Im 6 0 R >>",
        (image,),
    )
    expected = [ABSOLUTE] * 4 if override else SETTINGS
    assert pixels == [internal_expected(value) for value in expected]


def internal_shading() -> bytes:
    return (
        b"<< /ShadingType 2 /ColorSpace [/ICCBased 5 0 R] /Coords [0 0 10 0] /Extend [true true] "
        b"/Function << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 1] /C1 [0 0 0 1] /N 1 >> >>"
    )


def test_direct_shading_converts_full_icc_profile_at_paint_time() -> None:
    pixels = internal_pixels(
        internal_repeated_paint(b"0 0 10 10 re W n /S sh"),
        b"/Shading << /S 6 0 R >>",
        (internal_shading(),),
    )
    assert pixels == [internal_expected(value) for value in SETTINGS]


@pytest.mark.parametrize("override", [False, True])
def test_shading_pattern_uses_current_parameters_and_pattern_extgstate(override: bool) -> None:
    pattern = (
        b"<< /PatternType 2 /Shading 6 0 R "
        + (b"/ExtGState << /UseBlackPtComp /ON >>" if override else b"")
        + b" >>"
    )
    content = b"/Pattern cs /P scn " + internal_repeated_paint(b"0 0 10 10 re f")
    pixels = internal_pixels(content, b"/Pattern << /P 7 0 R >>", (internal_shading(), pattern))
    settings = (ON, ON, ON, ABSOLUTE) if override else SETTINGS
    assert pixels == [internal_expected(value) for value in settings]


def test_reused_tiling_pattern_cache_includes_rendering_parameters() -> None:
    pattern = internal_stream(
        b"/C cs 0 0 0 1 scn 0 0 10 10 re f",
        b"/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 10 10] "
        b"/XStep 10 /YStep 10 /Resources << /ColorSpace << /C [/ICCBased 5 0 R] >> >>",
    )
    content = b"/Pattern cs /P scn " + internal_repeated_paint(b"0 0 10 10 re f")
    pixels = internal_pixels(content, b"/Pattern << /P 6 0 R >>", (pattern,))
    assert pixels == [internal_expected(value) for value in SETTINGS]


@pytest.mark.parametrize("rendering", [DEFAULT_COLOR_RENDERING, OFF, ON, ABSOLUTE])
def test_nested_icc_spaces_keep_profiles_for_scalar_and_image_conversion(
    rendering: ColorRendering,
) -> None:
    icc = ["ICCBased", PdfStream({"N": 4}, internal_transform().profile)]
    function = {"FunctionType": 2, "Domain": [0, 1], "C0": [0, 0, 0, 1], "C1": [0, 0, 0, 1], "N": 1}
    for space in (
        ["Indexed", icc, 0, bytes([0, 0, 0, 255])],
        ["Separation", "Ink", icc, function],
        ["DeviceN", ["Ink"], icc, function],
    ):
        color = color_operands_to_srgb(parse_color_space(space), [0], rendering=rendering)
        assert color is not None
        assert tuple(round(value * 255) for value in color) == internal_expected(rendering)
        for depth in (8, 16):
            image = prepare_image(
                ImageSource(
                    bytes(depth // 8),
                    {"Width": 1, "Height": 1, "BitsPerComponent": depth, "ColorSpace": space},
                    color_rendering=rendering,
                )
            )
            assert image is not None
            assert tuple(image.raster.array[0, 0]) == internal_expected(rendering)
