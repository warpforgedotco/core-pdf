import pytest

from core_pdf.impl.fonts_widths import parse_cid_widths, parse_font_widths


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), 10**400, True, "bad", b"bad", None]
)
@pytest.mark.parametrize("form", ["compact", "array", "range"])
def test_invalid_cid_widths_are_skipped_without_losing_later_entries(value, form):
    data = {
        "compact": [0, [value, 600]],
        "array": [0, [value], 1, [600]],
        "range": [0, 0, value, 1, 1, 600],
    }[form]
    assert dict(parse_cid_widths(data)) == {1: 600.0}
    metrics = parse_font_widths({"DescendantFonts": [{"W": data}]}, "Type0")
    assert dict(metrics.widths) == {1: 600.0}
    assert metrics.default_width == 1000


@pytest.mark.parametrize("width", [500, 500.0, "500", b"500", 0, -250.5])
@pytest.mark.parametrize("form", ["compact", "array", "range"])
def test_recovery_width_forms_share_numeric_coercion(width, form):
    data = {
        "compact": [3, [width, 600]],
        "array": [3, [width], 4, [600]],
        "range": [3, 3, width, 4, 4, 600],
    }[form]
    assert dict(parse_cid_widths(data)) == {3: float(width), 4: 600.0}


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (None, {}),
        ([], {}),
        ([0, []], {}),
        ([-2, [10, 20, 30, 40]], {0: 30.0, 1: 40.0}),
        ([65534, [10, 20, 30]], {65534: 10.0, 65535: 20.0}),
        ([-3, -1, 500], {}),
        ([65536, [500]], {}),
        ([3, 1, 500], {}),
        ([-1, 1, 500], {0: 500.0, 1: 500.0}),
        ([65535, 65537, 500], {65535: 500.0}),
        (["bad", 2, [400]], {2: 400.0}),
        ([2], {}),
        ([2, 3], {}),
        ([2, "bad", 500, 4, [600]], {4: 600.0}),
        ([2, [500], 2, 2, 0], {2: 0.0}),
        ([True, 2, [500]], {2: 500.0}),
        ([1.5, 2, [500]], {2: 500.0}),
        ([b"bad", 2, [500]], {2: 500.0}),
    ],
)
def test_recovery_clips_ranges_resynchronizes_and_preserves_last_assignment(data, expected):
    assert dict(parse_cid_widths(data)) == expected


@pytest.mark.parametrize("first", ["2", b"2", 2.0])
def test_recovery_accepts_integral_cid_encodings(first):
    assert dict(parse_cid_widths([first, [500]])) == {2: 500.0}


@pytest.mark.parametrize("data", [0, "bad", {}, True])
def test_non_array_widths_are_rejected(data):
    with pytest.raises(ValueError, match="invalid CID widths array"):
        parse_cid_widths(data)


@pytest.mark.parametrize("data", [[-1, [100, 200], 3, [300]], ["-1", [100, 200], 3, [300]]])
def test_sparse_width_arrays_skip_out_of_range_cids(data):
    assert dict(parse_cid_widths(data)) == {0: 200.0, 3: 300.0}


@pytest.mark.parametrize(
    ("w2", "expected"),
    [
        (None, {}),
        ("bad", {}),
        ([], {}),
        (["bad", 2, [-900, 10, 700]], {2: (-900, 10, 700)}),
        ([-1, [-900, 10, 700, -800, 20, 600, 99]], {0: (-800, 20, 600)}),
        ([65535, 65536, -900, 10, 700], {65535: (-900, 10, 700)}),
        ([3, 2, -900, 10, 700], {}),
        ([2, "bad", -900, 10, 700], {}),
        ([2, 3, -900], {}),
        ([2, ["bad", "bad", "bad"]], {2: (-1100, 0, 0)}),
        ([2, 2, "bad", "bad", "bad"], {2: (-1100, 0, 0)}),
    ],
)
def test_vertical_recovery_clips_ranges_and_preserves_complete_metric_triples(w2, expected):
    font = {
        "MissingWidth": 500,
        "DescendantFonts": [{"DW": "0", "DW2": ["900", "-1100"], "W2": w2}],
    }
    metrics = parse_font_widths(font, "Type0")
    assert metrics.vertical_metrics == expected
    assert metrics.default_width == 0
    assert metrics.default_width_explicit
    assert metrics.default_vertical_origin_y == 900
    assert metrics.default_vertical_displacement_y == -1100


@pytest.mark.parametrize(
    ("extra", "expected", "default", "explicit"),
    [
        (
            {"FirstChar": "bad", "LastChar": "bad", "Widths": [200, "bad"]},
            {0: 200, 1: 500},
            500,
            True,
        ),
        ({"Widths": [200]}, {0: 200}, 500, True),
        ({"FirstChar": "3", "LastChar": "3", "Widths": [200, 300]}, {3: 200}, 500, True),
        ({"FontDescriptor": {"MissingWidth": "0"}, "Widths": ["bad"]}, {0: 0}, 0, True),
        ({"FontDescriptor": {}}, {}, 500, True),
        ({"MissingWidth": None, "FirstChar": "bad", "Widths": ["bad"]}, {0: 1000}, 1000, False),
    ],
)
def test_simple_font_recovery_preserves_declared_defaults_and_bounds(
    extra, expected, default, explicit
):
    metrics = parse_font_widths({"MissingWidth": 500, **extra}, "Type1")
    assert dict(metrics.widths) == expected
    assert metrics.default_width == default
    assert metrics.default_width_explicit is explicit


@pytest.mark.parametrize("descendants", [None, [], [0]])
def test_missing_descendant_recovery_uses_native_defaults(descendants):
    metrics = parse_font_widths({"DescendantFonts": descendants}, "Type0")
    assert dict(metrics.widths) == {}
    assert metrics.default_width == 1000


def test_simple_font_recovery_rejects_non_array_widths():
    with pytest.raises(ValueError, match="invalid font widths array"):
        parse_font_widths({"MissingWidth": 500, "Widths": "bad"}, "Type1")
