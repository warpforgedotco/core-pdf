"""The shared color description leaves reader coercion and output policy in core."""

from types import SimpleNamespace

import numpy
import pytest

from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf.impl._impl.graphics.color import color_operands_to_srgb, internal_convert_image_data
from core_pdf.impl._impl.graphics.color_math import d50_xyz_to_srgb
from core_pdf.impl._impl.graphics.color_spec import (
    parse_color_space,
    recover_image_bits_per_component,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.color_math import lab_components_to_xyz
from core_pdf_spec.types import PdfName


def test_reader_keeps_image_bit_depth_coercion_and_default() -> None:
    assert recover_image_bits_per_component({}) == 8
    assert recover_image_bits_per_component({"BitsPerComponent": "4"}) == 4
    with pytest.raises(ValueError):
        recover_image_bits_per_component({"BitsPerComponent": True})


def test_reader_keeps_device_image_fast_paths() -> None:
    raw = bytes([0, 127, 255, 64, 32, 16])
    assert (
        internal_convert_image_data(
            raw, {"ColorSpace": PdfName.of("DeviceRGB"), "Width": 2, "Height": 1}
        )
        == raw
    )
    gray = bytes([0, 127])
    assert (
        internal_convert_image_data(
            gray, {"ColorSpace": PdfName.of("DeviceGray"), "Width": 2, "Height": 1}
        )
        == gray
    )


@pytest.mark.parametrize("white_point", [(0.9642, 1.0, 0.8249), (0.9505, 1.0, 1.089)])
@pytest.mark.parametrize(
    ("ranges", "components"),
    [
        (None, [[0, -100, 100], [100, 100, -100], [20, -20, 20], [80, -60, 60]]),
        ([-20, 20, -40, 40], [[0, -20, 40], [100, 20, -40], [20, -4, 8], [80, -12, 24]]),
        ([-4, 2, 3, 9], [[0, -4, 9], [100, 2, 3], [20, -1.6, 6.6], [80, -2.8, 7.8]]),
    ],
)
def test_lab_image_samples_use_actual_component_ranges(
    white_point: tuple[float, float, float],
    ranges: list[int] | None,
    components: list[list[float]],
) -> None:
    # Fixed decoded triples exercise image sample mapping independently of the
    # Lab equations covered by the standalone color-math tests.
    params: dict[str, object] = {"WhitePoint": list(white_point)}
    if ranges is not None:
        params["Range"] = ranges
    raw = bytes([0, 0, 255, 255, 255, 0, 51, 102, 153, 204, 51, 204])
    result = internal_convert_image_data(
        raw,
        {"ColorSpace": ["Lab", params], "Width": 4, "Height": 1, "BitsPerComponent": 8},
    )
    assert result is not None
    xyz = lab_components_to_xyz(numpy.array(components, dtype=numpy.float32), white_point)
    expected = numpy.clip(d50_xyz_to_srgb(xyz) * 255.0, 0, 255).astype(numpy.uint8).reshape(-1)
    actual = numpy.frombuffer(result, dtype=numpy.uint8)
    # Float32 sample decoding can straddle a final integer conversion boundary.
    numpy.testing.assert_allclose(actual, expected, rtol=0, atol=1)


def test_reader_icc_dictionary_and_channel_coercion_keep_selected_fallback() -> None:
    space = parse_color_space(["ICCBased", {"N": "3", "Range": ["0", "1"] * 3}])
    assert len(space.component_ranges) == 3
    assert space.alternate is not None
    assert space.alternate.kind == "DeviceRGB"
    assert "N" not in space.params
    assert "Range" not in space.params


def test_reader_preserves_inert_profile_bytes_and_short_palette_recovery() -> None:
    profile = PdfStream(dictionary={"N": 3}, raw_data=b"unsupported profile")
    space = parse_color_space(["Indexed", ["ICCBased", profile], "1", b"\0\0\0"])
    assert space.base is not None
    assert space.base.icc_profile == b"unsupported profile"
    assert color_operands_to_srgb(space, (1,)) is None


def test_retained_calibrated_base_does_not_require_a_new_output_backend() -> None:
    space = parse_color_space(["Indexed", ["Lab", {"Range": [-20, 20, -40, 40]}], 0, b"\xff\0\xff"])
    assert space.base is not None
    assert space.base.kind == "Lab"
    assert color_operands_to_srgb(space, (0,)) is None


def test_unknown_space_retains_generic_component_recovery() -> None:
    resolver = ObjectResolver(b"", {})
    state = TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))
    state.graphics.fill_space = parse_color_space(["Lab", {"Range": [-20, 20, -40, 40]}])
    state.op_cs((PdfName.of("Unknown"),), 0)
    state.op_sc(("2", "-1", "0.5"), 0)
    assert state.graphics.fill_space.kind == "Unknown"
    assert state.graphics.fill_color == (1, 0, 0.5)


@pytest.mark.parametrize("value", ["/DeviceRGB", b"/DeviceRGB", ["/DeviceRGB"]])
def test_reader_keeps_host_slash_prefixed_color_names(value: object) -> None:
    space = parse_color_space(value)
    assert space.kind == "DeviceRGB"
    assert len(space.component_ranges) == 3
    assert (
        internal_convert_image_data(b"\x12\x34\x56", {"ColorSpace": value, "Width": 1, "Height": 1})
        == b"\x12\x34\x56"
    )


def test_reader_preserves_literal_slash_in_pdf_color_name() -> None:
    space = parse_color_space(PdfName.of("/DeviceRGB"))
    assert space.kind == "/DeviceRGB"
    assert space.component_ranges == ()


def test_recursive_color_recovery_keeps_unsupported_base_fallback() -> None:
    raw: list[object] = ["Indexed", None, 0, b"\0"]
    raw[1] = raw
    space = parse_color_space(raw)
    assert space.base is not None
    assert space.base.component_ranges == ()
    assert color_operands_to_srgb(space, (0,)) is None


def test_reader_invalid_icc_count_retains_color_and_generic_operand_recovery() -> None:
    resolver = ObjectResolver(b"", {})
    state = TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))
    state.op_rg((1, 0, 0), 0)
    state.resources = {"ColorSpace": {"I": [PdfName.of("ICCBased"), {"N": 2}]}}
    state.op_cs((PdfName.of("I"),), 0)
    assert state.graphics.fill_space.kind == "ICCBased"
    assert state.graphics.fill_color == (1, 0, 0)
    state.op_sc(("0.25", "0.75"), 0)
    assert state.graphics.fill_color == (0.25, 0.75)
