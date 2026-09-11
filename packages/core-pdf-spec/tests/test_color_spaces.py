"""One color-space description retains the PDF rules shared by content and images."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content import interpreter
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.color import (
    indexed_color_components,
    normalize_color_components,
    tint_color_components,
)
from core_pdf_spec.s_08_graphics.color_spec import DEVICE_RGB, parse_color_space
from core_pdf_spec.s_08_graphics.image_spec import image_bits_per_component
from core_pdf_spec.types import PdfName


def internal_lab() -> list[object]:
    return ["Lab", {"WhitePoint": [1, 1, 1], "Range": [-20, 20, -40, 40]}]


def internal_tint(outputs: int = 3) -> dict[str, object]:
    return {"FunctionType": 2, "Domain": [0, 1], "N": 1, "C0": [0] * outputs, "C1": [1] * outputs}


def test_indexed_lab_lookup_scales_each_base_component_range() -> None:
    # ISO 32000-1, 8.6.6.3: palette bytes span each base component's complete range.
    space = parse_color_space(["Indexed", internal_lab(), 0, b"\xff\x00\xff"])
    assert space.base is not None
    assert space.base.kind == "Lab"
    assert indexed_color_components(space, 0) == (100, -20, 40)


def test_indexed_icc_retains_profile_alternate_and_ranges() -> None:
    profile = PdfStream(
        dictionary={"N": 3, "Range": [-1, 1, -2, 2, -3, 3], "Alternate": internal_lab()},
        raw_data=b"inert ICC bytes",
    )
    space = parse_color_space(["Indexed", ["ICCBased", profile], 0, b"\xff\x00\xff"])
    assert space.base is not None
    assert space.base.alternate is not None
    assert space.base.alternate.kind == "Lab"
    assert space.base.icc_profile == b"inert ICC bytes"
    assert "N" not in space.base.params
    assert "Range" not in space.base.params
    assert indexed_color_components(space, 0) == (1, -2, 3)


@pytest.mark.parametrize("kind", ["Separation", "DeviceN"])
def test_tint_alternate_keeps_calibrated_space_and_checks_result_count(kind: str) -> None:
    names: object = PdfName.of("Ink") if kind == "Separation" else [PdfName.of("Ink")]
    space = parse_color_space([kind, names, internal_lab(), internal_tint()])
    assert space.colorants == ("Ink",)
    assert space.alternate is not None
    assert space.alternate.kind == "Lab"
    assert tint_color_components(space, (0.5,)) == (0.5, 0.5, 0.5)
    bad = parse_color_space([kind, names, internal_lab(), internal_tint(2)])
    with pytest.raises(ValueError, match="output count"):
        tint_color_components(bad, (0.5,))


def test_indexed_allows_separation_base() -> None:
    # PDF 1.3 explicitly permits Separation and DeviceN as Indexed bases.
    base = ["Separation", PdfName.of("Ink"), "DeviceGray", internal_tint(1)]
    space = parse_color_space(["Indexed", base, 0, b"\xff"])
    assert space.base is not None
    assert space.base.kind == "Separation"
    assert indexed_color_components(space, 0) == (1,)


@pytest.mark.parametrize("base", ["Pattern", ["Indexed", "DeviceGray", 0, b"\0"]])
def test_indexed_rejects_prohibited_base_spaces(base: object) -> None:
    with pytest.raises(ValueError, match="base color space"):
        parse_color_space(["Indexed", base, 0, b"\0"])


@pytest.mark.parametrize("lookup", [b"\0\0", b"\0\0\0\0"])
def test_indexed_requires_exact_palette_length(lookup: bytes) -> None:
    with pytest.raises(ValueError, match="lookup"):
        parse_color_space(["Indexed", "DeviceRGB", 0, lookup])


def test_color_parameters_are_immutable_without_range_mirrors() -> None:
    raw = internal_lab()
    space = parse_color_space(raw)
    params = cast(dict[str, Any], raw[1])
    params["WhitePoint"][0] = 99
    params["Range"][0] = 99
    assert space.params["WhitePoint"] == (1, 1, 1)
    assert space.component_ranges == ((0, 100), (-20, 20), (-40, 40))
    assert "Range" not in space.params
    with pytest.raises(TypeError):
        cast(Any, space.params)["WhitePoint"] = (99, 1, 1)


@pytest.mark.parametrize(
    "value",
    [
        "Lab",
        "Unknown",
        ["DeviceRGB", 1],
        ["Lab", {}],
        ["Lab", {"WhitePoint": [1, 1, 1], "Range": []}],
        ["Lab", {"WhitePoint": [1, 1, 1], "Range": [1, 0, 0, 1]}],
        ["CalGray", {"WhitePoint": [1, 1, 1], "Gamma": "2"}],
        ["CalRGB", {"WhitePoint": [1, 1, 1], "Gamma": []}],
        ["CalRGB", {"WhitePoint": [1, 1, 1], "Matrix": []}],
        ["ICCBased", {"N": 3}],
        ["Pattern", "Pattern"],
        ["Separation", PdfName.of("Ink"), "Pattern", internal_tint()],
    ],
)
def test_color_parser_rejects_invalid_shapes_and_parameters(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_color_space(value)


def test_null_optional_color_parameters_use_pdf_defaults() -> None:
    space = parse_color_space(["CalRGB", {"WhitePoint": [1, 1, 1], "Gamma": None, "Matrix": None}])
    assert space.params["Gamma"] == (1, 1, 1)
    assert len(space.component_ranges) == 3


def test_color_space_cycle_is_rejected() -> None:
    raw: list[object] = ["Pattern"]
    raw.append(raw)
    with pytest.raises(ValueError, match="cycle"):
        parse_color_space(raw)


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16])
def test_image_bits_are_separate_from_color_space(bits: int) -> None:
    assert image_bits_per_component({"BitsPerComponent": bits}) == bits
    space = parse_color_space("DeviceRGB")
    assert not hasattr(space, "bits_per_component")
    assert not hasattr(space, "channels")


@pytest.mark.parametrize("bits", [None, 0, 3, True, "8", 8.0])
def test_ordinary_image_requires_valid_integer_bit_depth(bits: object) -> None:
    with pytest.raises(ValueError):
        image_bits_per_component({"BitsPerComponent": bits})


def test_mask_and_jpx_bit_depth_rules() -> None:
    # ISO 32000-1 Table 89: mask default is one; JPX ignores this dictionary entry.
    assert image_bits_per_component({"ImageMask": True}) == 1
    assert image_bits_per_component({"ImageMask": True, "BitsPerComponent": 1}) == 1
    with pytest.raises(ValueError):
        image_bits_per_component({"ImageMask": True, "BitsPerComponent": 8})
    assert image_bits_per_component({"Filter": PdfName.of("JPXDecode")}) is None
    assert (
        image_bits_per_component(
            {"Filter": [PdfName.of("JPXDecode")], "BitsPerComponent": "ignored"}
        )
        is None
    )


def internal_state() -> ContentInterpreter:
    sink = SimpleNamespace()
    return ContentInterpreter(ObjectResolver(b"", {}), cast(Any, sink), cast(Any, None))


def test_custom_color_handler_receives_raw_operands_after_validation() -> None:
    state = internal_state()
    state.graphics.fill_space = DEVICE_RGB
    operands = (2, -1, 0.5)
    observed = []
    state.operator_overrides["sc"] = lambda values, depth: observed.append((values, depth))
    before = state.graphics.fill_color
    state.execute_operation("sc", operands, 3)
    assert observed == [(operands, 3)]
    assert observed[0][0] is operands
    assert state.graphics.fill_color == before
    with pytest.raises(PdfParseError):
        state.execute_operation("sc", ("1", 0, 0), 0)
    assert len(observed) == 1


@pytest.mark.parametrize("pattern", [False, True])
def test_default_color_normalizes_components_once(
    monkeypatch: pytest.MonkeyPatch, pattern: bool
) -> None:
    state = internal_state()
    space = parse_color_space(["Pattern", internal_lab()] if pattern else internal_lab())
    state.graphics.fill_space = space
    calls = []

    def normalize(spec: Any, values: Any) -> tuple[float, ...]:
        calls.append(values)
        return normalize_color_components(spec, values)

    monkeypatch.setattr(interpreter, "normalize_color_components", normalize)
    if pattern:
        state.resources = {
            "Pattern": {
                "P": PdfStream(
                    dictionary={
                        "PatternType": 1,
                        "PaintType": 2,
                        "BBox": [0, 0, 1, 1],
                        "XStep": 1,
                        "YStep": 1,
                    }
                )
            }
        }
        operands: Any = (50, -30, 50, PdfName.of("P"))
    else:
        operands = (50, -30, 50)
    state.execute_operation("scn", operands, 0)
    assert state.graphics.fill_color == (50, -20, 40)
    assert len(calls) == 1
