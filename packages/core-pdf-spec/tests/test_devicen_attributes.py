# SPDX-License-Identifier: AGPL-3.0-only
"""ISO 32000-2:2020, 8.6.6.5 and Tables 70-71: DeviceN process/spot metadata."""

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics import color_spec
from core_pdf_spec.s_08_graphics.color_spec import (
    DeviceNAttributes,
    DeviceNProcess,
    parse_color_space,
    parse_device_n_attributes,
)
from core_pdf_spec.standards import PdfVersion, SemanticContext
from core_pdf_spec.types import PdfName


def internal_tint() -> dict[str, object]:
    return {"FunctionType": 2, "Domain": [0, 1], "N": 1, "C0": [0], "C1": [1]}


def internal_spot(name: str) -> list[object]:
    return ["Separation", name, "DeviceGray", internal_tint()]


def internal_attributes(
    space: object = "DeviceRGB", components: tuple[str, ...] = ("R", "G", "B")
) -> dict[str, object]:
    return {"Subtype": "NChannel", "Process": {"ColorSpace": space, "Components": list(components)}}


def internal_space(names: tuple[str, ...], attributes: object) -> list[object]:
    return ["DeviceN", list(names), "DeviceGray", internal_tint(), attributes]


def test_new_models_and_parser_are_public_chapter_exports() -> None:
    assert {"DeviceNProcess", "DeviceNAttributes", "parse_device_n_attributes"} <= set(
        color_spec.__all__
    )


def test_nchannel_rgb_preserves_process_space_order_and_raw_attributes() -> None:
    raw = internal_attributes()
    hints = {"PrivateHint": 42}
    raw["MixingHints"] = hints
    space = parse_color_space(internal_space(("R", "G", "B"), raw))
    attributes = space.devicen_attributes
    assert isinstance(attributes, DeviceNAttributes)
    assert attributes.subtype == "NChannel"
    assert isinstance(attributes.process, DeviceNProcess)
    assert attributes.process.color_space.kind == "DeviceRGB"
    assert attributes.process.components == ("R", "G", "B")
    assert attributes.process.component_indices == (0, 1, 2)
    assert not attributes.colorants
    assert space.params["Attributes"] == raw
    # The typed model copies arrays; raw Attributes deliberately retain opaque
    # optional hints without claiming a mixing algorithm or full Table 72 check.
    cast(dict[str, Any], raw["Process"])["Components"][0] = "Changed"
    assert attributes.process.components == ("R", "G", "B")
    assert cast(dict[str, object], space.params["Attributes"])["MixingHints"] is hints
    with pytest.raises(TypeError):
        cast(Any, attributes.colorants)["Spot"] = parse_color_space(internal_spot("Spot"))
    with pytest.raises(FrozenInstanceError):
        cast(Any, attributes.process).components = ()


@pytest.mark.parametrize("version", [None, PdfVersion(1, 3), PdfVersion(1, 6), PdfVersion(2, 0)])
def test_colorants_metadata_is_not_gated_at_pdf_16(version: PdfVersion | None) -> None:
    # Adobe PDF Reference 1.3, Table 4.20 (p.189) already defines Colorants,
    # including additional unused names. Table 70's PDF 1.6 tag is not a gate.
    raw = {"Colorants": {"Spot": internal_spot("Spot"), "Unused": internal_spot("Unused")}}
    space = parse_color_space(
        internal_space(("Spot", "Other"), raw),
        context=SemanticContext(version) if version is not None else None,
    )
    assert space.devicen_attributes is not None
    assert space.devicen_attributes.subtype == "DeviceN"
    assert space.devicen_attributes.process is None
    assert set(space.devicen_attributes.colorants) == {"Spot", "Unused"}


def test_absent_attributes_and_null_optional_entries_keep_their_defaults() -> None:
    space = parse_color_space(["DeviceN", ["Spot"], "DeviceGray", internal_tint()])
    assert space.devicen_attributes is None
    attributes = parse_device_n_attributes(
        {"Subtype": None, "Process": None, "Colorants": None, "MixingHints": None}, ("Spot",)
    )
    assert attributes.subtype == "DeviceN"
    assert attributes.process is None
    assert not attributes.colorants


@pytest.mark.parametrize("names", [("None",), ("None", "None"), ("Spot", "None", "None")])
def test_ordinary_devicen_permits_repeated_none(names: tuple[str, ...]) -> None:
    assert parse_color_space(internal_space(names, {})).colorants == names


@pytest.mark.parametrize("names", [("All",), ("Spot", "Spot"), ("None", "All")])
@pytest.mark.parametrize("with_attributes", [False, True])
def test_devicen_rejects_all_and_duplicate_actual_colorants(
    names: tuple[str, ...], with_attributes: bool
) -> None:
    raw = internal_space(names, {})
    if not with_attributes:
        raw.pop()
    with pytest.raises(ValueError, match="DeviceN colorant names"):
        parse_color_space(raw)
    # All remains the prescribed special Separation name.
    assert parse_color_space(internal_spot("All")).colorants == ("All",)


def test_nchannel_forbids_none_even_if_a_separation_describes_it() -> None:
    with pytest.raises(ValueError, match="None is not allowed"):
        parse_device_n_attributes(
            {"Subtype": "NChannel", "Colorants": {"None": internal_spot("None")}}, ("None",)
        )


@pytest.mark.parametrize(
    ("names", "indices"),
    [
        (("Cyan", "Magenta", "Yellow", "Black"), (0, 1, 2, 3)),
        (("Black", "Yellow", "Cyan", "Magenta"), (2, 3, 1, 0)),
        (("Magenta", "Yellow"), (None, 0, 1, None)),
        (("Black",), (None, None, None, 0)),
    ],
)
@pytest.mark.parametrize("icc", [False, True])
def test_cmyk_process_components_may_be_subset_and_reordered(
    names: tuple[str, ...], indices: tuple[int | None, ...], icc: bool
) -> None:
    process_space: object = (
        ["ICCBased", PdfStream(dictionary={"N": 4}, raw_data=b"profile bytes")]
        if icc
        else "DeviceCMYK"
    )
    attributes = parse_device_n_attributes(
        internal_attributes(process_space, ("Cyan", "Magenta", "Yellow", "Black")), names
    )
    assert attributes.process is not None
    assert attributes.process.component_indices == indices
    assert attributes.process.color_space.kind == ("ICCBased" if icc else "DeviceCMYK")


def test_cmyk_reserved_names_and_arbitrary_aliases_both_identify_process_components() -> None:
    raw = internal_attributes("DeviceCMYK", ("C", "M", "Y", "K"))
    # Reserved names remain CMYK process colours even without matching aliases
    # in Components. A process Colorants value is ignored before validation.
    raw["Colorants"] = {"Cyan": 42, "K": ["not a Separation"], "Unused": internal_spot("Unused")}
    attributes = parse_device_n_attributes(raw, ("K", "Cyan", "Yellow"))
    assert attributes.process is not None
    assert attributes.process.component_indices == (1, None, 2, 0)
    assert set(attributes.colorants) == {"Unused"}


def test_alias_and_reserved_name_cannot_supply_the_same_process_component_twice() -> None:
    with pytest.raises(ValueError, match="conflicting.*aliases"):
        parse_device_n_attributes(
            internal_attributes("DeviceCMYK", ("C", "M", "Y", "K")), ("C", "Cyan")
        )


@pytest.mark.parametrize(
    "components",
    [
        ("Magenta", "Cyan", "Yellow", "Black"),
        ("C", "M", "Y", "None"),
        ("C", "M", "Y", "All"),
    ],
)
def test_process_components_cannot_reassign_reserved_names(components: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="reserved name"):
        parse_device_n_attributes(internal_attributes("DeviceCMYK", components), ("Cyan",))


@pytest.mark.parametrize("names", [("R", "B", "G"), ("R", "G"), ("R", "Spot", "G", "B")])
def test_non_cmyk_process_components_must_be_complete_contiguous_and_ordered(
    names: tuple[str, ...],
) -> None:
    raw = internal_attributes()
    raw["Colorants"] = {"Spot": internal_spot("Spot")}
    with pytest.raises(ValueError, match="complete and in natural order"):
        parse_device_n_attributes(raw, names)


@pytest.mark.parametrize(
    "process_space",
    [
        "DeviceRGB",
        ["CalRGB", {"WhitePoint": [1, 1, 1]}],
        ["ICCBased", PdfStream(dictionary={"N": 3}, raw_data=b"profile bytes")],
    ],
)
def test_non_cmyk_process_can_use_aliases_and_have_adjacent_spots(process_space: object) -> None:
    # ISO 32000-2:2020, 8.6.6.5 Example 4 distinguishes an RGB process alias
    # ProcessRed from a separate spot colourant named Red.
    raw = internal_attributes(process_space, ("ProcessRed", "ProcessGreen", "ProcessBlue"))
    raw["Colorants"] = {"Red": internal_spot("Red"), "Other": internal_spot("Other")}
    attributes = parse_device_n_attributes(
        raw, ("Red", "ProcessRed", "ProcessGreen", "ProcessBlue", "Other")
    )
    assert attributes.process is not None
    assert attributes.process.component_indices == (1, 2, 3)
    assert set(attributes.colorants) == {"Red", "Other"}


@pytest.mark.parametrize("process_space", ["DeviceGray", ["CalGray", {"WhitePoint": [1, 1, 1]}]])
def test_gray_process_uses_one_naturally_ordered_component(process_space: object) -> None:
    attributes = parse_device_n_attributes(internal_attributes(process_space, ("Gray",)), ("Gray",))
    assert attributes.process is not None
    assert attributes.process.component_indices == (0,)


@pytest.mark.parametrize(
    "process_space",
    [
        ["Lab", {"WhitePoint": [1, 1, 1]}],
        "Pattern",
        ["Indexed", "DeviceGray", 0, b"\0"],
        internal_spot("Spot"),
        internal_space(("Spot",), {}),
    ],
)
def test_process_space_excludes_lab_and_all_special_color_spaces(process_space: object) -> None:
    with pytest.raises(ValueError, match="process color space"):
        parse_device_n_attributes(internal_attributes(process_space), ("R", "G", "B"))


@pytest.mark.parametrize("names", [("Cyan",), ("R", "G", "B", "Cyan")])
def test_nchannel_reserved_cmyk_names_cannot_be_declared_spots(names: tuple[str, ...]) -> None:
    raw = {"Subtype": "NChannel", "Colorants": {"Cyan": internal_spot("Cyan")}}
    if len(names) == 4:
        raw.update(internal_attributes())
    with pytest.raises(ValueError, match="process"):
        parse_device_n_attributes(raw, names)


def test_all_spot_nchannel_requires_matching_definitions_and_allows_extra_spots() -> None:
    raw = {
        "Subtype": "NChannel",
        "Colorants": {"Spot": internal_spot("Spot"), "Unused": internal_spot("Unused")},
    }
    attributes = parse_device_n_attributes(raw, ("Spot",))
    assert attributes.process is None
    assert set(attributes.colorants) == {"Spot", "Unused"}


@pytest.mark.parametrize("colorants", [None, {}, {"Spot": None}, {"Other": internal_spot("Other")}])
def test_nchannel_spot_colorants_must_all_be_described(colorants: object) -> None:
    with pytest.raises(ValueError, match="spot colorants require"):
        parse_device_n_attributes({"Subtype": "NChannel", "Colorants": colorants}, ("Spot",))


@pytest.mark.parametrize("value", ["DeviceRGB", internal_spot("Other"), 42])
def test_colorants_entries_must_be_matching_separation_spaces(value: object) -> None:
    with pytest.raises(ValueError):
        parse_device_n_attributes({"Colorants": {"Spot": value}}, ("Spot",))


def test_process_colorants_entries_are_ignored_even_when_invalid_or_cyclic() -> None:
    raw = internal_attributes()
    raw["Colorants"] = {"R": raw, "G": 42, "B": ["Separation", "Wrong"]}
    attributes = parse_device_n_attributes(raw, ("R", "G", "B"))
    assert not attributes.colorants


def test_attributes_parser_is_independent_of_the_outer_alternate_and_tint_transform() -> None:
    attributes = internal_attributes()
    broken = ["DeviceN", ["R", "G", "B"], "broken alternate", None, attributes]
    with pytest.raises(ValueError):
        parse_color_space(broken)
    assert parse_device_n_attributes(attributes, ("R", "G", "B")).process is not None


def test_process_cycle_to_parent_color_space_is_rejected() -> None:
    attributes = internal_attributes()
    space = internal_space(("R", "G", "B"), attributes)
    cast(dict[str, object], attributes["Process"])["ColorSpace"] = space
    with pytest.raises(ValueError, match="cycle"):
        parse_color_space(space)
    with pytest.raises(ValueError, match="cycle"):
        parse_device_n_attributes(attributes, ("R", "G", "B"))


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {"Subtype": "Unknown"},
        {"Subtype": 42},
        {"Process": []},
        {"Process": {}},
        {"Process": {"ColorSpace": "DeviceRGB", "Components": ["R", "G"]}},
        {"Process": {"ColorSpace": "DeviceRGB", "Components": ["R", "R", "B"]}},
        {"Process": {"ColorSpace": "DeviceRGB", "Components": ["R", "G", 42]}},
        {"Colorants": []},
        {"Colorants": {42: internal_spot("Spot")}},
        {"MixingHints": []},
    ],
)
def test_invalid_attributes_shapes_are_rejected(raw: object) -> None:
    with pytest.raises(ValueError):
        parse_device_n_attributes(raw, ("R", "G", "B"))


@pytest.mark.parametrize("names", [(), ("R", 42), (None,)])
def test_public_attributes_parser_validates_its_colorant_names(names: Any) -> None:
    with pytest.raises(ValueError, match="colorant name"):
        parse_device_n_attributes({}, names)


def test_pdf_name_objects_are_accepted_as_resolved_names() -> None:
    raw = internal_attributes()
    raw["Subtype"] = PdfName.of("NChannel")
    cast(dict[str, object], raw["Process"])["Components"] = [PdfName.of(name) for name in "RGB"]
    attributes = parse_device_n_attributes(raw, ("R", "G", "B"))
    assert attributes.process is not None
    assert attributes.process.components == ("R", "G", "B")


@pytest.mark.parametrize("version", [None, PdfVersion(1, 8), PdfVersion(9, 0)])
def test_explicit_unknown_context_does_not_guess_attribute_semantics(
    version: PdfVersion | None,
) -> None:
    with pytest.raises(PdfUnsupportedError, match="recognized PDF version"):
        parse_device_n_attributes(
            internal_attributes(), ("R", "G", "B"), context=SemanticContext(version)
        )


@pytest.mark.parametrize("version", [PdfVersion(1, 3), PdfVersion(1, 6), PdfVersion(2, 0)])
def test_nchannel_metadata_has_no_standalone_feature_availability_gate(version: PdfVersion) -> None:
    attributes = parse_device_n_attributes(
        internal_attributes(), ("R", "G", "B"), context=SemanticContext(version)
    )
    assert attributes.process is not None
    assert attributes.process.component_indices == (0, 1, 2)


def test_ordinary_devicen_can_retain_inactive_partial_rgb_process_metadata() -> None:
    raw = internal_attributes()
    raw["Subtype"] = "DeviceN"
    attributes = parse_device_n_attributes(raw, ("R",))
    assert attributes.process is not None
    assert attributes.process.component_indices == (0, None, None)
