# SPDX-License-Identifier: AGPL-3.0-only
"""NChannel process selection is reader policy around strict component mappings."""

from collections.abc import Mapping
from typing import Any, cast

import imagecodecs
import numpy
import pytest

from core_pdf.impl._impl.graphics.color import color_operands_to_srgb
from core_pdf.impl._impl.graphics.color_spec import parse_color_space
from core_pdf.impl._impl.graphics.device_profiles import default_cmyk_transform
from core_pdf.impl._impl.graphics.icc_profiles import parse_icc_transform
from core_pdf.impl._impl.graphics.images import prepare_image
from core_pdf_spec.exceptions import PdfDecryptionError, PdfParseError, PdfUnsupportedError
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.s_08_graphics.image_spec import ImageSource


def internal_tint(count: int) -> PdfStream:
    return PdfStream(
        {
            "FunctionType": 0,
            "Domain": [0, 1] * count,
            "Range": [0, 1] * 3,
            "Size": [1] * count,
            "BitsPerSample": 8,
            "Encode": [0, 0] * count,
        },
        bytes([255, 0, 255]),
    )


def internal_nchannel(
    process_space: object,
    components: list[str],
    *,
    names: list[str] | None = None,
    attributes: dict[str, Any] | None = None,
) -> list[Any]:
    source_names = components if names is None else names
    return [
        "DeviceN",
        source_names,
        "DeviceRGB",
        internal_tint(len(source_names)),
        attributes
        if attributes is not None
        else {
            "Subtype": "NChannel",
            "Process": {"ColorSpace": process_space, "Components": components},
        },
    ]


def internal_scalar(
    space: object, values: list[float], rendering: ColorRendering = DEFAULT_COLOR_RENDERING
) -> tuple[int, ...]:
    converted = color_operands_to_srgb(parse_color_space(space), values, rendering=rendering)
    assert converted is not None
    return tuple(round(value * 255) for value in converted)


def internal_image(
    space: object,
    values: list[int],
    depth: int,
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
    **entries: object,
) -> numpy.ndarray:
    raw = numpy.asarray(values, dtype=numpy.uint8 if depth == 8 else ">u2").tobytes()
    prepared = prepare_image(
        ImageSource(
            raw,
            {"Width": 1, "Height": 1, "BitsPerComponent": depth, "ColorSpace": space, **entries},
            color_rendering=rendering,
        )
    )
    assert prepared is not None
    return prepared.raster.array[0, 0]


def test_attributes_are_preserved_and_invalid_attributes_keep_global_tint_fallback() -> None:
    for raw_attributes in ({"Subtype": "Unknown"}, {"Subtype": "NChannel"}, ["malformed"]):
        raw = internal_nchannel("DeviceRGB", ["R", "G", "B"])
        raw[4] = raw_attributes
        space = parse_color_space(raw)
        assert space.params["Attributes"] == raw_attributes
        assert space.devicen_attributes is None
        assert internal_scalar(raw, [0, 1, 0]) == (255, 0, 255)
        for depth in (8, 16):
            assert tuple(internal_image(raw, [0, (1 << depth) - 1, 0], depth)) == (255, 0, 255)
    raw = internal_nchannel("DeviceRGB", ["R", "G", "B"])
    space = parse_color_space(raw)
    retained = space.params["Attributes"]
    assert isinstance(retained, Mapping)
    raw[4]["Subtype"] = "ChangedAfterParsing"
    assert cast(Mapping[str, object], retained)["Subtype"] == "NChannel"


@pytest.mark.parametrize("depth", [8, 16])
def test_process_rgb_natural_values_bypass_global_tint_and_decode_before_mapping(
    depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core_pdf.impl._impl.graphics import image_samples

    def unexpected_tint(*args: object, **kwargs: object) -> None:
        raise AssertionError("process-only NChannel must not evaluate the global tint function")

    monkeypatch.setattr(image_samples, "internal_compile_pdf_function", unexpected_tint)
    space = internal_nchannel("DeviceRGB", ["First", "Second", "Third"])
    maximum = (1 << depth) - 1
    assert internal_scalar(space, [1, 0, 1]) == (255, 0, 255)
    assert tuple(internal_image(space, [maximum, 0, maximum], depth, Decode=[1, 0] * 3)) == (
        0,
        255,
        0,
    )
    assert internal_scalar(space, [1, 1, 1]) == (255, 255, 255)


@pytest.mark.parametrize("depth", [8, 16])
def test_gray_process_preserves_additive_values_and_source_color_key_mask(depth: int) -> None:
    space = internal_nchannel("DeviceGray", ["Tone"])
    maximum = (1 << depth) - 1
    assert internal_scalar(space, [0]) == (0, 0, 0)
    assert tuple(internal_image(space, [maximum], depth)) == (255,)
    assert tuple(
        internal_image(space, [maximum], depth, Decode=[1, 0], Mask=[maximum, maximum])
    ) == (0, 0)


@pytest.mark.parametrize("depth", [8, 16])
@pytest.mark.parametrize(
    "rendering",
    [
        DEFAULT_COLOR_RENDERING,
        ColorRendering(black_point_compensation="OFF"),
        ColorRendering("AbsoluteColorimetric", "ON"),
    ],
)
def test_cmyk_subset_reorders_components_and_keeps_rendering_controls(
    depth: int, rendering: ColorRendering
) -> None:
    transform = default_cmyk_transform()
    assert transform is not None
    space = internal_nchannel(
        "DeviceCMYK", ["Cyan", "Magenta", "Yellow", "Black"], names=["Black", "Cyan"]
    )
    maximum = (1 << depth) - 1
    expected = transform.apply_uint16(
        numpy.array([[65535, 0, 0, 65535]], dtype=numpy.uint16), rendering=rendering
    )[0]
    assert internal_scalar(space, [1, 1], rendering) == tuple(expected)
    numpy.testing.assert_array_equal(
        internal_image(space, [maximum, maximum], depth, rendering=rendering), expected
    )


@pytest.mark.parametrize("depth", [8, 16])
def test_nested_indexed_process_preserves_icc_profile_and_native_sample_precision(
    depth: int,
) -> None:
    profile = bytes(imagecodecs.cms_profile("linearrgb"))
    process = ["ICCBased", PdfStream({"N": 3}, profile)]
    space = internal_nchannel(process, ["R", "G", "B"])
    maximum = (1 << depth) - 1
    integers = [maximum // 2, maximum // 3, maximum // 4]
    words = numpy.rint(numpy.asarray([integers], dtype=numpy.float64) / maximum * 65535).astype(
        numpy.uint16
    )
    expected = parse_icc_transform(profile).apply_uint16(words)[0]
    assert internal_scalar(space, [value / maximum for value in integers]) == tuple(expected)
    numpy.testing.assert_array_equal(internal_image(space, integers, depth), expected)
    indexed = ["Indexed", space, 0, bytes([128, 85, 64])]
    palette_words = numpy.array([[128, 85, 64]], dtype=numpy.uint16) * 257
    palette_expected = parse_icc_transform(profile).apply_uint16(palette_words)[0]
    assert internal_scalar(indexed, [0]) == tuple(palette_expected)
    numpy.testing.assert_array_equal(internal_image(indexed, [0], depth), palette_expected)


@pytest.mark.parametrize("subtype", ["DeviceN", "NChannel"])
def test_ordinary_and_mixed_spaces_retain_global_tint(subtype: str) -> None:
    space = internal_nchannel(
        "DeviceGray",
        ["Tone"],
        names=["Tone", "Spot"],
        attributes={
            "Subtype": subtype,
            "Process": {"ColorSpace": "DeviceGray", "Components": ["Tone"]},
            "Colorants": {"Spot": ["Separation", "Spot", "DeviceRGB", internal_tint(1)]},
        },
    )
    parsed = parse_color_space(space)
    assert parsed.devicen_attributes is not None
    assert internal_scalar(space, [0, 0]) == (255, 0, 255)
    for depth in (8, 16):
        assert tuple(internal_image(space, [0, 0], depth)) == (255, 0, 255)


def test_unused_spot_definitions_do_not_make_a_process_only_space_mixed() -> None:
    space = internal_nchannel("DeviceGray", ["Tone"])
    space[4]["Colorants"] = {"Unused": ["Separation", "Unused", "DeviceRGB", internal_tint(1)]}
    assert internal_scalar(space, [1]) == (255, 255, 255)


def test_lab_process_and_misordered_rgb_mapping_recover_to_global_tint() -> None:
    for space in (
        internal_nchannel(["Lab", {"WhitePoint": [0.9642, 1, 0.8249]}], ["L", "A", "B"]),
        internal_nchannel("DeviceRGB", ["R", "G", "B"], names=["B", "R", "G"]),
    ):
        assert parse_color_space(space).devicen_attributes is None
        assert internal_scalar(space, [0, 1, 0]) == (255, 0, 255)


@pytest.mark.parametrize(
    "error", [FilterParseError, FilterUnsupportedError, PdfParseError, PdfUnsupportedError]
)
def test_unreadable_process_icc_recovers_global_tint(error: type[Exception]) -> None:
    def unreadable(*args: object, **kwargs: object) -> bytes:
        raise error("unreadable process profile")

    stream = PdfStream({"N": 3}, b"profile", decoder=unreadable)
    raw = internal_nchannel(["ICCBased", stream], ["R", "G", "B"])
    parsed = parse_color_space(raw)
    assert parsed.params["Attributes"] == raw[4]
    assert parsed.devicen_attributes is None
    assert internal_scalar(raw, [0, 0, 0]) == (255, 0, 255)


@pytest.mark.parametrize("error", [RuntimeError, PdfDecryptionError])
def test_process_icc_does_not_swallow_decoder_bugs_or_authentication_errors(
    error: type[Exception],
) -> None:
    def failed_decoder(*args: object, **kwargs: object) -> bytes:
        raise error("must propagate")

    stream = PdfStream({"N": 3}, b"profile", decoder=failed_decoder)
    raw = internal_nchannel(["ICCBased", stream], ["R", "G", "B"])
    with pytest.raises(error, match="must propagate"):
        parse_color_space(raw)


@pytest.mark.parametrize("values", [[], [0], [0, 1], [0, 0, 0, 0]])
def test_malformed_scalar_component_counts_recover_without_mapping_index_errors(
    values: list[float],
) -> None:
    raw = internal_nchannel("DeviceRGB", ["R", "G", "B"])
    assert color_operands_to_srgb(parse_color_space(raw), values) is None


@pytest.mark.parametrize("subtype", ["DeviceN", None])
def test_process_metadata_alone_does_not_change_ordinary_devicen(subtype: str | None) -> None:
    raw = internal_nchannel("DeviceRGB", ["R", "G", "B"])
    raw[4]["Subtype"] = subtype
    assert internal_scalar(raw, [0, 1, 0]) == (255, 0, 255)
    for depth in (8, 16):
        assert tuple(internal_image(raw, [0, (1 << depth) - 1, 0], depth)) == (255, 0, 255)


@pytest.mark.parametrize("kind", ["CalGray", "CalRGB"])
@pytest.mark.parametrize("depth", [8, 16])
@pytest.mark.parametrize(
    "rendering",
    [
        DEFAULT_COLOR_RENDERING,
        ColorRendering(black_point_compensation="OFF"),
        ColorRendering(black_point_compensation="ON"),
        ColorRendering("AbsoluteColorimetric", "ON"),
    ],
)
def test_calibrated_process_uses_its_dictionary_and_existing_output_controls(
    kind: str, depth: int, rendering: ColorRendering
) -> None:
    params: dict[str, object] = {
        "WhitePoint": [0.9642, 1, 0.8249],
        "BlackPoint": [0.019284, 0.02, 0.016498],
    }
    if kind == "CalGray":
        params["Gamma"] = 2
        names, values = ["Gray"], [0.25]
    else:
        params["Gamma"] = [2, 2, 2]
        params["Matrix"] = [0.9642, 0, 0, 0, 1, 0, 0, 0, 0.8249]
        names, values = ["Red", "Green", "Blue"], [0.25, 0.5, 0.75]
    process = [kind, params]
    raw = internal_nchannel(process, names)
    expected = internal_scalar(process, values, rendering)
    assert internal_scalar(raw, values, rendering) == expected
    decoded = internal_image(
        raw,
        [0] * len(names),
        depth,
        rendering=rendering,
        Decode=[bound for value in values for bound in (value, value)],
    )
    assert tuple(decoded) == expected


@pytest.mark.parametrize("depth", [8, 16])
def test_icc_process_ranges_clip_natural_values_without_rescaling_them(depth: int) -> None:
    profile = bytes(imagecodecs.cms_profile("linearrgb"))
    process = ["ICCBased", PdfStream({"N": 3, "Range": [0.25, 0.75] * 3}, profile)]
    raw = internal_nchannel(process, ["R", "G", "B"])
    clipped_words = numpy.array([[16384, 32768, 49151]], dtype=numpy.uint16)
    expected = parse_icc_transform(profile).apply_uint16(clipped_words)[0]
    assert internal_scalar(raw, [0, 0.5, 1]) == tuple(expected)
    numpy.testing.assert_array_equal(
        internal_image(raw, [0, 0, 0], depth, Decode=[0, 0, 0.5, 0.5, 1, 1]), expected
    )


@pytest.mark.parametrize("depth", [8, 16])
def test_icc_cmyk_process_accepts_reserved_subset_names_with_aliased_components(depth: int) -> None:
    transform = default_cmyk_transform()
    assert transform is not None
    process = ["ICCBased", PdfStream({"N": 4}, transform.profile)]
    raw = internal_nchannel(process, ["C", "M", "Y", "K"], names=["Black", "Magenta"])
    words = numpy.array([[0, 65535, 0, 65535]], dtype=numpy.uint16)
    expected = transform.apply_uint16(words)[0]
    assert internal_scalar(raw, [1, 1]) == tuple(expected)
    numpy.testing.assert_array_equal(internal_image(raw, [(1 << depth) - 1] * 2, depth), expected)
