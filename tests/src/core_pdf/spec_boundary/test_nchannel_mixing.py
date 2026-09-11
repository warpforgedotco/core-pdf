# SPDX-License-Identifier: AGPL-3.0-only
"""Core's selected screen approximation, not a mandated spectral mixing model."""

from typing import Any

import imagecodecs
import numpy
import pytest

from core_pdf.impl._impl.graphics.color import color_operands_to_srgb
from core_pdf.impl._impl.graphics.color_spec import parse_color_space
from core_pdf.impl._impl.graphics.image_samples import internal_convert_components
from core_pdf.impl._impl.graphics.images import prepare_image
from core_pdf_spec.exceptions import PdfDecryptionError, PdfParseError, PdfUnsupportedError
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.s_08_graphics.image_spec import ImageSource


def internal_spot(
    name: str, solid: tuple[float, ...], alternate: object = "DeviceRGB"
) -> list[Any]:
    return [
        "Separation",
        name,
        alternate,
        {"FunctionType": 2, "Domain": [0, 1], "N": 1, "C0": [1] * len(solid), "C1": list(solid)},
    ]


def internal_space(
    process: object = "DeviceRGB",
    components: tuple[str, ...] = ("R", "G", "B"),
    spots: tuple[tuple[str, tuple[float, ...]], ...] = (("YellowSpot", (1, 1, 0)),),
) -> list[Any]:
    names = [*components, *(name for name, _ in spots)]
    attributes: dict[str, Any] = {
        "Subtype": "NChannel",
        "Colorants": {name: internal_spot(name, solid) for name, solid in spots},
    }
    if process is not None:
        attributes["Process"] = {"ColorSpace": process, "Components": list(components)}
    return [
        "DeviceN",
        names,
        "DeviceRGB",
        PdfStream(
            {
                "FunctionType": 0,
                "Domain": [0, 1] * len(names),
                "Range": [0, 1] * 3,
                "Size": [1] * len(names),
                "BitsPerSample": 8,
            },
            b"\xff\x00\xff",  # Global fallback is deliberately magenta.
        ),
        attributes,
    ]


def internal_scalar(
    raw: object, values: list[float], rendering: ColorRendering = DEFAULT_COLOR_RENDERING
) -> tuple[int, ...]:
    result = color_operands_to_srgb(parse_color_space(raw), values, rendering=rendering)
    assert result is not None
    return tuple(round(value * 255) for value in result)


def internal_image(
    raw: object,
    values: list[int],
    depth: int,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
    **entries: object,
) -> numpy.ndarray:
    raster = prepare_image(
        ImageSource(
            numpy.asarray(values, dtype=numpy.uint8 if depth == 8 else ">u2").tobytes(),
            {"Width": 1, "Height": 1, "BitsPerComponent": depth, "ColorSpace": raw, **entries},
            color_rendering=rendering,
        )
    )
    assert raster is not None
    return raster.raster.array[0, 0]


@pytest.mark.parametrize("depth", [8, 16])
def test_process_rgb_and_spot_appearances_multiply_with_original_precision(depth: int) -> None:
    raw = internal_space()
    maximum = (1 << depth) - 1
    words = [maximum // 5, maximum * 2 // 5, maximum * 4 // 5, maximum // 2]
    values = [word / maximum for word in words]
    # First convert each component appearance to the selected output, then mix.
    process = numpy.rint(numpy.asarray(values[:3]) * 255)
    spot = numpy.rint(numpy.array([1, 1, 1 - values[3]]) * 255)
    expected = numpy.rint(process * spot / 255).astype(numpy.uint8)
    assert internal_scalar(raw, values) == tuple(expected)
    numpy.testing.assert_array_equal(internal_image(raw, words, depth), expected)


@pytest.mark.parametrize("depth", [8, 16])
def test_all_spot_nchannel_starts_on_white_and_zero_tints_leave_it_white(depth: int) -> None:
    raw = internal_space(None, (), (("CyanSpot", (0, 1, 1)), ("MagentaSpot", (1, 0, 1))))
    assert internal_scalar(raw, [0, 0]) == (255, 255, 255)
    assert internal_scalar(raw, [1, 1]) == (0, 0, 255)
    maximum = (1 << depth) - 1
    assert tuple(internal_image(raw, [maximum, maximum], depth)) == (0, 0, 255)
    # Reversing the input order does not create an implicit ink laydown order.
    raw[1].reverse()
    assert internal_scalar(raw, [1, 0]) == (255, 0, 255)


def test_gray_process_and_gray_spot_expand_to_rgb_appearances() -> None:
    raw = internal_space("DeviceGray", ("Gray",), (("Ink", (0,)),))
    raw[4]["Colorants"]["Ink"][2] = "DeviceGray"
    assert internal_scalar(raw, [0.8, 0.5]) == (102, 102, 102)


def test_missing_cmyk_components_have_zero_ink_before_spot_mixing() -> None:
    raw = internal_space("DeviceCMYK", ("Cyan", "Magenta", "Yellow", "Black"))
    raw[1] = ["Yellow", "YellowSpot"]
    baseline = internal_convert_components(
        numpy.array([[0, 0, 1, 0.0]]), parse_color_space("DeviceCMYK")
    )
    expected = tuple(int(value) for value in baseline[0] * numpy.array([1, 1, 0]))
    assert internal_scalar(raw, [1, 1]) == expected


@pytest.mark.parametrize(
    "hints",
    [
        {"Solidities": {"YellowSpot": 1}, "PrintingOrder": ["R", "G", "B", "YellowSpot"]},
        {"PrintingOrder": ["YellowSpot"]},
        {"DotGain": {"Default": {"FunctionType": 2, "Domain": [0, 1], "N": 2}}},
        {"PrivateHint": 42},
    ],
)
def test_nonempty_mixing_hints_use_the_whole_global_tint(hints: dict[str, Any]) -> None:
    raw = internal_space()
    raw[4]["MixingHints"] = hints
    assert internal_scalar(raw, [0.2, 0.4, 0.8, 0.5]) == (255, 0, 255)
    assert tuple(internal_image(raw, [0, 0, 0, 0], 8)) == (255, 0, 255)


@pytest.mark.parametrize("hints", [None, {}])
def test_absent_or_empty_hints_permit_screen_mixing(hints: object) -> None:
    raw = internal_space()
    raw[4]["MixingHints"] = hints
    assert internal_scalar(raw, [1, 1, 1, 1]) == (255, 255, 0)


def test_nonwhite_zero_endpoint_uses_global_tint_without_paper_normalization() -> None:
    raw = internal_space()
    raw[4]["Colorants"]["YellowSpot"][3]["C0"] = [0.8, 0.8, 0.8]
    assert internal_scalar(raw, [1, 1, 1, 0]) == (255, 0, 255)


def test_bad_individual_spot_does_not_return_a_partial_mix() -> None:
    raw = internal_space(None, (), (("First", (0, 1, 1)), ("Second", (1, 0, 1))))
    raw[4]["Colorants"]["Second"][3] = {"FunctionType": 99}
    assert internal_scalar(raw, [1, 1]) == (255, 0, 255)


@pytest.mark.parametrize(
    "error", [FilterParseError, FilterUnsupportedError, PdfParseError, PdfUnsupportedError]
)
def test_unreadable_spot_sampled_function_uses_global_tint(error: type[Exception]) -> None:
    def decode(*args: Any, **kwargs: Any) -> bytes:
        raise error("unreadable spot samples")

    raw = internal_space()
    raw[4]["Colorants"]["YellowSpot"][3] = PdfStream(
        {"FunctionType": 0, "Domain": [0, 1], "Range": [0, 1] * 3, "Size": [2], "BitsPerSample": 8},
        b"samples",
        decoder=decode,
    )
    assert internal_scalar(raw, [1, 1, 1, 0.5]) == (255, 0, 255)


@pytest.mark.parametrize("error", [RuntimeError, PdfDecryptionError])
def test_spot_stream_does_not_hide_decoder_bugs_or_authentication_errors(
    error: type[Exception],
) -> None:
    def decode(*args: Any, **kwargs: Any) -> bytes:
        raise error("must propagate")

    raw = internal_space()
    raw[4]["Colorants"]["YellowSpot"][3] = PdfStream(
        {"FunctionType": 0, "Domain": [0, 1], "Range": [0, 1] * 3, "Size": [2], "BitsPerSample": 8},
        b"samples",
        decoder=decode,
    )
    with pytest.raises(error, match="must propagate"):
        internal_scalar(raw, [1, 1, 1, 0.5])


@pytest.mark.parametrize("depth", [8, 16])
@pytest.mark.parametrize(
    "rendering",
    [
        DEFAULT_COLOR_RENDERING,
        ColorRendering(black_point_compensation="OFF"),
        ColorRendering("AbsoluteColorimetric", "ON"),
    ],
)
def test_process_and_spot_icc_profiles_keep_rendering_controls(
    depth: int, rendering: ColorRendering, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = bytes(imagecodecs.cms_profile("linearrgb"))
    icc = ["ICCBased", PdfStream({"N": 3, "Alternate": "DeviceRGB"}, profile)]
    raw = internal_space(icc)
    raw[4]["Colorants"]["YellowSpot"][2] = icc
    maximum = (1 << depth) - 1
    words = [maximum // 7, maximum // 5, maximum // 3, maximum // 2]
    values = numpy.asarray([words], dtype=numpy.float64) / maximum
    process = internal_convert_components(
        values[:, :3], parse_color_space(icc), rendering=rendering
    )
    spot = internal_convert_components(
        numpy.array([[1, 1, 1 - values[0, 3]]]), parse_color_space(icc), rendering=rendering
    )
    expected = numpy.rint(process.astype(numpy.float64) * spot / 255).astype(numpy.uint8)[0]
    calls: list[tuple[int, int, str]] = []
    original = imagecodecs.cms_transform

    def record(samples: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((kwargs["intent"], kwargs["flags"], str(samples.dtype)))
        return original(samples, *args, **kwargs)

    monkeypatch.setattr(imagecodecs, "cms_transform", record)
    assert internal_scalar(raw, values[0].tolist(), rendering) == tuple(expected)
    numpy.testing.assert_array_equal(internal_image(raw, words, depth, rendering), expected)
    intent = 3 if rendering.intent == "AbsoluteColorimetric" else 1
    flags = 0x0100 if rendering.black_point_compensation == "OFF" or intent == 3 else 0x2100
    assert len(calls) >= 6  # Zero endpoint, process and spot, for scalar and image.
    assert set(calls) == {(intent, flags, "uint16")}
