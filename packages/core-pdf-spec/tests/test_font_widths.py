from collections.abc import Mapping

import pytest

from core_pdf_spec.s_09_fonts.widths import CompactCIDWidthMap, parse_cid_widths, parse_font_widths


@pytest.mark.parametrize("value", ["500", b"500", True, float("nan"), float("inf"), 10**400])
@pytest.mark.parametrize("form", ["compact", "array", "range"])
def test_cid_width_forms_reject_non_pdf_numbers(value: object, form: str) -> None:
    data = {"compact": [0, [value]], "array": [0, [value], 2, [600]], "range": [0, 0, value]}[form]
    with pytest.raises(ValueError, match="CID width"):
        parse_cid_widths(data)


@pytest.mark.parametrize("value", ["0", b"0", True, 0.0])
@pytest.mark.parametrize("field", ["FirstChar", "LastChar"])
def test_simple_width_range_requires_pdf_integers(value: object, field: str) -> None:
    font: dict[str, object] = {"FirstChar": 0, "LastChar": 0, "Widths": [500]}
    font[field] = value
    with pytest.raises(ValueError, match=field):
        parse_font_widths(font, "Type1")


@pytest.mark.parametrize("value", ["0", b"0", True, 0.0])
@pytest.mark.parametrize("last", [False, True])
def test_cid_width_range_requires_pdf_integers(value: object, last: bool) -> None:
    with pytest.raises(ValueError, match="CID width"):
        parse_cid_widths([0, value, 500] if last else [value, [500]])


@pytest.mark.parametrize("value", ["500", b"500", True, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["Widths", "MissingWidth", "DW", "DW2", "W2-array", "W2-range"])
def test_font_width_numbers_are_strict_in_every_dictionary_form(value: object, field: str) -> None:
    if field == "Widths":
        font = {"FirstChar": 0, "LastChar": 0, "Widths": [value]}
        subtype = "Type1"
    elif field == "MissingWidth":
        font = {"FontDescriptor": {"MissingWidth": value}}
        subtype = "Type1"
    else:
        entry = {
            "DW": {"DW": value},
            "DW2": {"DW2": [880, value]},
            "W2-array": {"W2": [0, [-1000, value, 880]]},
            "W2-range": {"W2": [0, 0, -1000, 0, value]},
        }[field]
        font = {"DescendantFonts": [entry]}
        subtype = "Type0"
    with pytest.raises(ValueError):
        parse_font_widths(font, subtype)


@pytest.mark.parametrize("value", ["0", b"0", True, 0.0])
@pytest.mark.parametrize("last", [False, True])
def test_vertical_width_indices_require_pdf_integers(value: object, last: bool) -> None:
    w2 = [0, value, -1000, 0, 880] if last else [value, [-1000, 0, 880]]
    with pytest.raises(ValueError, match="CID W2"):
        parse_font_widths({"DescendantFonts": [{"W2": w2}]}, "Type0")


def test_null_optional_widths_equal_omission_but_explicit_zero_survives() -> None:
    null_cid = {"DescendantFonts": [dict.fromkeys(("DW", "DW2", "W", "W2"))]}
    metrics = parse_font_widths(null_cid, "Type0")
    assert metrics == parse_font_widths({"DescendantFonts": [{}]}, "Type0")
    assert metrics.default_width == 1000
    assert not metrics.default_width_explicit
    assert (metrics.default_vertical_origin_y, metrics.default_vertical_displacement_y) == (
        880,
        -1000,
    )
    assert parse_font_widths(
        {
            "FontDescriptor": {"MissingWidth": None},
            "Widths": None,
            "FirstChar": None,
            "LastChar": None,
        },
        "Type1",
    ) == parse_font_widths({}, "Type1")
    explicit_defaults: list[tuple[dict[str, object], str]] = [
        ({"DescendantFonts": [{"DW": 0}]}, "Type0"),
        ({"FontDescriptor": {"MissingWidth": 0}}, "Type1"),
    ]
    for font, subtype in explicit_defaults:
        zero = parse_font_widths(font, subtype)
        assert zero.default_width == 0
        assert zero.default_width_explicit


@pytest.mark.parametrize("field", ["FirstChar", "LastChar"])
def test_null_required_simple_width_range_is_missing(field: str) -> None:
    font = {"FirstChar": 0, "LastChar": 0, "Widths": [500], field: None}
    with pytest.raises(ValueError, match="missing"):
        parse_font_widths(font, "Type1")


def test_width_mappings_keep_compact_storage_and_last_entry_precedence() -> None:
    compact = parse_cid_widths([65533, [500, 0, -250.5]])
    sparse = parse_cid_widths([65533, 65535, 500, 65534, [0, -250.5]])
    assert isinstance(compact, CompactCIDWidthMap)
    assert isinstance(sparse, dict)
    for widths in (compact, sparse):
        assert isinstance(widths, Mapping)
        assert list(widths.items()) == [(65533, 500.0), (65534, 0.0), (65535, -250.5)]
        assert widths.get(65534, 999) == 0
        assert widths.get(65536, 999) == 999
        with pytest.raises(KeyError):
            widths[65536]


def test_vertical_metrics_preserve_real_values_and_last_entry_precedence() -> None:
    metrics = parse_font_widths(
        {"DescendantFonts": [{"W2": [65534, 65535, -1000, 250, 880, 65535, [-750.5, 0, 900]]}]},
        "Type0",
    )
    assert metrics.vertical_metrics == {65534: (-1000, 250, 880), 65535: (-750.5, 0, 900)}
