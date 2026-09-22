import pytest

from core_pdf.impl.graphics.color_spec import (
    cs_param_floats,
    describe_color_space,
    internal_color_space_paints,
    internal_nchannel_process,
    parse_color_space,
    recover_image_bits_per_component,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.color_spec import ColorSpace, DeviceNAttributes


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ([], None),
        ([None], None),
        ("/DeviceRGB", "DeviceRGB"),
        (["/DeviceGray"], "DeviceGray"),
        (["Indexed"], "Indexed"),
        (["Indexed", None], "Indexed"),
        (["Indexed", [None]], "Indexed"),
        (["Indexed", "DeviceRGB", 1, b""], "Indexed:DeviceRGB"),
        (["ICCBased"], "ICCBased"),
        (["ICCBased", None], "ICCBased"),
        (["ICCBased", {}], "ICCBased"),
        (["ICCBased", {"Alternate": "DeviceCMYK"}], "ICCBased:DeviceCMYK"),
        (["ICCBased", PdfStream({"Alternate": "DeviceGray"}, b"broken")], "ICCBased:DeviceGray"),
        (["Indexed", ["ICCBased", {"Alternate": "DeviceRGB"}]], "Indexed:ICCBased:DeviceRGB"),
        (["Separation", "Spot", "DeviceRGB", None], "Separation"),
    ],
)
def test_color_description_keeps_nested_names_without_decoding_profiles(value, expected):
    assert describe_color_space(value) == expected


@pytest.mark.parametrize("kind", ["Indexed", "ICCBased", "Pattern"])
def test_cyclic_color_spaces_stop_at_the_recursive_base(kind):
    value: list[object] = [kind]
    if kind == "Indexed":
        value.extend([value, 1, b"\x00\xff"])
    elif kind == "ICCBased":
        value.append({"N": 1, "Alternate": value})
    else:
        value.append(value)
    assert describe_color_space(value) == kind
    parsed = parse_color_space(value)
    nested = parsed.alternate if kind == "ICCBased" else parsed.base
    assert nested is not None
    assert nested.kind == "Unknown"
    assert internal_color_space_paints(value)


@pytest.mark.parametrize("kind", ["DeviceGray", "DeviceRGB", "DeviceCMYK", "Pattern"])
@pytest.mark.parametrize("array", [False, True])
def test_builtin_color_spaces_accept_direct_and_singleton_names(kind, array):
    parsed = parse_color_space([kind] if array else kind)
    assert parsed.kind == kind
    assert (
        len(parsed.component_ranges)
        == {"DeviceGray": 1, "DeviceRGB": 3, "DeviceCMYK": 4, "Pattern": 0}[kind]
    )
    assert parsed.base is None


def test_uncolored_pattern_reuses_its_base_component_ranges():
    parsed = parse_color_space(["Pattern", "DeviceRGB"])
    assert parsed.kind == "Pattern"
    assert parsed.base is not None
    assert parsed.base.kind == "DeviceRGB"
    assert parsed.component_ranges == ((0.0, 1.0),) * 3


@pytest.mark.parametrize("raw", [True, 0, -1, None, "bad", float("inf")])
def test_image_bit_depth_rejects_invalid_values(raw):
    with pytest.raises(ValueError, match="bits-per-component"):
        recover_image_bits_per_component({"BitsPerComponent": raw})


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 8), ({}, 8), ({"BitsPerComponent": "16"}, 16), ({"BitsPerComponent": 1}, 1)],
)
def test_image_bit_depth_defaults_and_reader_numeric_coercion(value, expected):
    assert recover_image_bits_per_component(value) == expected


@pytest.mark.parametrize("raw", [None, [], [1], "bad"])
def test_short_or_nonarray_numeric_parameters_use_the_complete_default(raw):
    default = [0.0, 1.0]
    assert cs_param_floats({"Range": raw}, "Range", 2, default) == default


@pytest.mark.parametrize("raw", [["0", b"1", "ignored"], (0, 1)])
def test_numeric_parameters_coerce_only_the_required_components(raw):
    assert cs_param_floats({"Range": raw}, "Range", 2, [-1, 1]) == [0.0, 1.0]


@pytest.mark.parametrize("raw", [None, "bad", float("nan"), float("inf"), -float("inf")])
def test_numeric_color_parameters_reject_invalid_and_nonfinite_values(raw):
    with pytest.raises(ValueError, match="color space parameters"):
        cs_param_floats({"Range": [0, raw]}, "Range", 2, [0, 1])


@pytest.mark.parametrize("kind", ["Lab", "ICCBased"])
def test_component_ranges_reject_reversed_bounds(kind):
    params = {"N": 1, "Range": [1, 0]} if kind == "ICCBased" else {"Range": [1, 0, -100, 100]}
    with pytest.raises(ValueError, match="component Range"):
        parse_color_space([kind, params])


@pytest.mark.parametrize("hival", [True, -1, "bad"])
def test_indexed_color_space_rejects_invalid_high_index(hival):
    with pytest.raises(ValueError, match="hival"):
        parse_color_space(["Indexed", "DeviceGray", hival, b"\x00"])


@pytest.mark.parametrize("lookup", [b"\x00\xff", "\x00\xff", PdfStream({}, b"\x00\xff"), object()])
def test_indexed_palettes_retain_bytes_or_mark_an_unusable_lookup(lookup):
    parsed = parse_color_space(["Indexed", "DeviceGray", "1", lookup])
    assert parsed.hival == 1
    assert parsed.component_ranges == ((0.0, 1.0),)
    assert parsed.base is not None
    assert parsed.base.kind == "DeviceGray"
    assert parsed.lookup == (None if type(lookup) is object else b"\x00\xff")


@pytest.mark.parametrize("count", [True, 0, -1, None, "bad"])
def test_icc_rejects_invalid_component_counts(count):
    with pytest.raises(ValueError, match="ICCBased"):
        parse_color_space(["ICCBased", {"N": count}])


@pytest.mark.parametrize("kind", ["CalGray", "CalRGB", "Lab"])
def test_calibrated_spaces_copy_parameters_and_keep_scalars(kind):
    white = [1, 1, 1]
    source = {"WhitePoint": white, "Gamma": 2.2}
    parsed = parse_color_space([kind, source])
    white[0] = 0
    assert parsed.params["WhitePoint"] == (1, 1, 1)
    assert parsed.params["Gamma"] == 2.2
    assert len(parsed.component_ranges) == (1 if kind == "CalGray" else 3)


@pytest.mark.parametrize("names", [None, [], [None], ["None", None]])
def test_unknown_devicen_colorants_are_not_mistaken_for_no_paint(names):
    assert internal_color_space_paints(["DeviceN", names, "DeviceRGB", None])


def test_devicen_rejects_nonarray_names_during_full_parsing():
    with pytest.raises(ValueError, match="DeviceN"):
        parse_color_space(["DeviceN", "Spot", "DeviceRGB", None])


def test_nchannel_without_a_process_has_no_process_only_projection():
    space = ColorSpace("DeviceN", (), devicen_attributes=DeviceNAttributes("NChannel", None, {}))
    assert internal_nchannel_process(space) is None


@pytest.mark.parametrize("kind", ["Lab", "ICCBased"])
@pytest.mark.parametrize("bound", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_component_ranges_fail_before_color_conversion(kind, bound):
    params = (
        {"N": 1, "Range": [0, bound]} if kind == "ICCBased" else {"Range": [0, bound, -100, 100]}
    )
    with pytest.raises(ValueError, match="color space parameters"):
        parse_color_space([kind, params])
