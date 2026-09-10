"""Reader coercions remain core-owned when spec input contracts tighten."""

from copy import deepcopy

import pytest

from core_pdf.impl._impl.graphics.color_spec import color_spec_from_value
from core_pdf.impl._impl.graphics.decode_compat import FilterParams
from core_pdf.impl._impl.graphics.filter_registry import declared_filter_names
from core_pdf.impl._impl.graphics.functions import internal_compile_pdf_function
from core_pdf.impl._impl.graphics.shading import prepare_shading
from core_pdf_spec.s_07_syntax.stream import PdfStream


def test_reader_filter_params_factory_preserves_subclass_and_recovery() -> None:
    class DerivedParams(FilterParams):
        pass

    dictionary = {
        "Columns": "2",
        "Predictor": b"12",
        "BlackIs1": 1,
        "DamagedRowsBeforeError": True,
        "Rows": None,
    }
    original = deepcopy(dictionary)
    params = DerivedParams.from_parms(dictionary)
    assert type(params) is DerivedParams
    assert params == DerivedParams(
        columns=2, predictor=12, black_is_1=True, damaged_rows_before_error=1, has_columns=True
    )
    assert dictionary == original


@pytest.mark.parametrize("value", [True, 2.0, "invalid", 0])
def test_reader_filter_params_still_reject_invalid_columns(value: object) -> None:
    with pytest.raises(ValueError, match="invalid DecodeParms Columns"):
        FilterParams.from_parms({"Columns": value})


def test_reader_preserves_filter_metadata_skipping() -> None:
    assert declared_filter_names(["FlateDecode", None, 42]) == ["FlateDecode"]


def test_reader_preserves_indexed_color_coercion() -> None:
    spec = color_spec_from_value(["Indexed", "DeviceRGB", "1", b"\x00" * 6])
    assert spec.hival == 1


def test_reader_preserves_function_coercion_and_default_domain() -> None:
    function = internal_compile_pdf_function({"FunctionType": "2", "N": "2"})
    assert function(0.5) == (0.25,)


def test_reader_preserves_shading_type_coercion() -> None:
    shading = prepare_shading(
        {
            "ShadingType": "2",
            "Coords": ["0", "0", "100", "0"],
            "Function": {"FunctionType": "2", "N": "1"},
        }
    )
    assert shading is not None
    assert shading.evaluate(0.5) == (0.5,)


def test_reader_retains_unclipped_exponential_function_outputs() -> None:
    dictionary = {
        "FunctionType": 2,
        "Domain": [0, 1],
        "N": 1,
        "C0": [-2, 8],
        "C1": [2, -8],
        "Range": [-1, 1, -3, 3],
    }
    function = internal_compile_pdf_function(dictionary)
    assert function(0) == (-2, 8)
    assert function(1) == (2, -8)
    assert dictionary["Range"] == [-1, 1, -3, 3]


def test_reader_ignores_invalid_exponential_range() -> None:
    function = internal_compile_pdf_function(
        {"FunctionType": 2, "Domain": [0, 1], "N": 1, "Range": [1, 0]}
    )
    assert function(0.5) == (0.5,)


def test_reader_retains_unclipped_parent_and_child_stitching_outputs() -> None:
    function = internal_compile_pdf_function(
        {
            "FunctionType": 3,
            "Domain": [0, 1],
            "Bounds": [],
            "Encode": [0, 1],
            "Range": [3, 7],
            "Functions": [
                {
                    "FunctionType": 2,
                    "Domain": [0, 1],
                    "N": 1,
                    "C0": [0],
                    "C1": [10],
                    "Range": [2, 8],
                }
            ],
        }
    )
    assert function(0) == (0,)
    assert function(1) == (10,)


@pytest.mark.parametrize("order", [3, 0, 1.0, "3", True, None])
def test_reader_retains_linear_sampled_interpolation_for_any_order(order: object) -> None:
    dictionary = {
        "FunctionType": 0,
        "BitsPerSample": 8,
        "Size": [4],
        "Domain": [0, 1],
        "Range": [0, 1],
        "Order": order,
    }
    function = internal_compile_pdf_function(PdfStream(dictionary, bytes([0, 255, 0, 255])))
    assert function(1 / 6) == pytest.approx((0.5,))
    assert function(0.5) == pytest.approx((0.5,))
    assert dictionary["Order"] == order


def test_reader_consumes_sampled_array_prefixes_without_mutating_source() -> None:
    dictionary = {
        "FunctionType": 0,
        "BitsPerSample": 8,
        "Size": [2],
        "Domain": ["0", "1", "unused"],
        "Range": ["0", "1", "unused"],
        "Encode": [0, 1, 123],
        "Decode": [0, 2, 456],
    }
    original = deepcopy(dictionary)
    function = internal_compile_pdf_function(PdfStream(dictionary, bytes([0, 255])))
    assert function(0.25) == (0.5,)
    assert function(0.75) == (1,)
    assert dictionary == original


@pytest.mark.parametrize("decode", [None, [-10, 10, 0, 1]])
def test_reader_retains_constant_output_for_reversed_sampled_range(decode: object) -> None:
    dictionary = {
        "FunctionType": 0,
        "BitsPerSample": 8,
        "Size": [2],
        "Domain": [0, 1],
        "Range": [2, 1, 0, 1],
        "Decode": decode,
    }
    original = deepcopy(dictionary)
    function = internal_compile_pdf_function(PdfStream(dictionary, bytes([0, 0, 255, 255])))
    assert function(0.25) == (2, 0.25)
    assert function(0.75) == (2, 0.75)
    assert dictionary == original


@pytest.mark.parametrize("decode", [[0], [0, 1, "invalid"], [0, 1, float("nan")]])
def test_reader_still_rejects_invalid_sampled_decode_arrays(decode: object) -> None:
    dictionary = {
        "FunctionType": 0,
        "BitsPerSample": 8,
        "Size": [2],
        "Domain": [0, 1],
        "Range": [0, 1],
        "Decode": decode,
    }
    with pytest.raises(ValueError):
        internal_compile_pdf_function(PdfStream(dictionary, bytes([0, 255])))


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ([0.75, 0.25], [0, 1 / 3, 2 / 3, 20 + 2 / 3, 21]),
        ([0.5, 0.5], [0, 0.5, 20, 20.5, 21]),
        ([-0.5, 0.5], [10.5, 10.75, 20, 20.5, 21]),
        ([0.5, 1.5], [0, 0.5, 10, 10.25, 10.5]),
        ([0, 0.5], [10, 10.5, 20, 20.5, 21]),
    ],
)
def test_reader_retains_first_match_stitching_for_malformed_bounds(
    bounds: list[float], expected: list[float]
) -> None:
    dictionary = {
        "FunctionType": 3,
        "Domain": [0, 1],
        "Bounds": bounds,
        "Encode": [0, 1] * 3,
        "Functions": [
            {"FunctionType": 2, "N": 1, "C0": [index * 10], "C1": [index * 10 + 1]}
            for index in range(3)
        ],
    }
    original = deepcopy(dictionary)
    function = internal_compile_pdf_function(dictionary)
    for value, result in zip([0, 0.25, 0.5, 0.75, 1], expected, strict=True):
        assert function(value) == pytest.approx((result,))
    assert dictionary == original


def test_reader_recovers_sampled_functions_inside_malformed_stitching() -> None:
    dictionary = {
        "FunctionType": 0,
        "BitsPerSample": 8,
        "Size": [4],
        "Domain": [0, 1, 123],
        "Range": [0, 1, 456],
        "Order": 3,
    }
    sampled = PdfStream(dictionary, bytes([0, 255, 0, 255]))
    function = internal_compile_pdf_function(
        {
            "FunctionType": 3,
            "Bounds": [0.75, 0.25],
            "Functions": [sampled] * 3,
            "Encode": [0, 1] * 3,
        }
    )
    assert function(0.125) == pytest.approx((0.5,))
    assert function(0.875) == pytest.approx((0.5,))
    assert dictionary["Order"] == 3


def test_reader_retains_zero_width_stitching_interval_recovery() -> None:
    function = internal_compile_pdf_function(
        {
            "FunctionType": 3,
            "Domain": [0, 0],
            "Bounds": [0],
            "Encode": [0, 1, 0.25, 1],
            "Functions": [{"FunctionType": 2}, {"FunctionType": 2}],
        }
    )
    assert function(-1) == (0.25,)
    assert function(1) == (0.25,)
