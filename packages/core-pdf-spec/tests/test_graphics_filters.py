"""Strict object-type validation at graphics/filter boundaries."""

import zlib

import pytest

from core_pdf_spec.s_07_filters.decode_spec import FilterParams, normalize_stream_decode_spec
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_filters.pipeline import decode_stream_data
from core_pdf_spec.s_08_graphics.color_spec import parse_color_space
from core_pdf_spec.s_08_graphics.pdf_function import compile_pdf_function
from core_pdf_spec.s_08_graphics.shading import parse_shading


def test_filter_params_factory_preserves_subclass_and_spec_defaults() -> None:
    class DerivedParams(FilterParams):
        pass

    params = DerivedParams.from_parms({"Columns": 2, "Predictor": 12})
    assert type(params) is DerivedParams
    assert params == DerivedParams(columns=2, predictor=12, has_columns=True)


@pytest.mark.parametrize("value", ["2", b"2", 2.0, True, 0])
def test_filter_columns_require_a_positive_pdf_integer(value: object) -> None:
    with pytest.raises(ValueError, match="invalid DecodeParms Columns"):
        FilterParams.from_parms({"Columns": value})


def test_filter_pipeline_decodes_valid_data_and_rejects_bad_names() -> None:
    assert decode_stream_data(zlib.compress(b"PDF"), {"Filter": "FlateDecode"}) == b"PDF"
    with pytest.raises(FilterParseError, match="invalid stream decode filter"):
        normalize_stream_decode_spec({"Filter": ["FlateDecode", 42]})
    with pytest.raises(FilterUnsupportedError, match="not implemented"):
        decode_stream_data(b"PDF", {"Filter": "MadeUpDecode"})


@pytest.mark.parametrize("hival", ["1", b"1", 1.0, True, -1, 256])
def test_indexed_color_requires_a_pdf_integer(hival: object) -> None:
    # ISO 32000-1, 8.6.6.3: hival is an integer in the range 0..255.
    with pytest.raises(ValueError, match="invalid hival"):
        parse_color_space(["Indexed", "DeviceRGB", hival, b"\x00" * 6])


def test_function_numeric_tokens_are_not_dictionary_numbers() -> None:
    with pytest.raises(ValueError, match="invalid PDF function type"):
        compile_pdf_function({"FunctionType": "2", "Domain": [0, 1], "N": 1})
    with pytest.raises(ValueError, match="invalid PDF function domain"):
        compile_pdf_function({"FunctionType": 2, "Domain": ["0", "1"], "N": 1})
    function = compile_pdf_function({"FunctionType": 2, "Domain": [0, 1], "N": 2})
    assert function(0.5) == (0.25,)


def test_shading_requires_a_pdf_integer_type() -> None:
    with pytest.raises(ValueError, match="unsupported shading type"):
        parse_shading({"ShadingType": "2"})


def test_exponential_function_clips_each_output_to_declared_range() -> None:
    # ISO 32000-1, Table 38: optional Range clips output values when present.
    dictionary = {
        "FunctionType": 2,
        "Domain": [0, 1],
        "N": 1,
        "C0": [-2, 8],
        "C1": [2, -8],
        "Range": [-1, 1, -3, 3],
    }
    function = compile_pdf_function(dictionary)
    assert function(-1) == (-1, 3)
    assert function(0.5) == (0, 0)
    assert function(2) == (1, -3)
    unclipped = compile_pdf_function(
        {key: value for key, value in dictionary.items() if key != "Range"}
    )
    assert unclipped(0) == (-2, 8)


@pytest.mark.parametrize(
    "range_values", [[], [0], [0, 1, 2], [1, 0], ["0", 1], [0, float("inf")], [0, 1, 0, 1]]
)
def test_exponential_function_rejects_invalid_or_mismatched_range(range_values: object) -> None:
    with pytest.raises(ValueError, match="range"):
        compile_pdf_function({"FunctionType": 2, "Domain": [0, 1], "N": 1, "Range": range_values})


def test_stitching_function_applies_child_and_parent_ranges() -> None:
    child = {
        "FunctionType": 2,
        "Domain": [0, 1],
        "N": 1,
        "C0": [0],
        "C1": [10],
        "Range": [2, 8],
    }
    dictionary = {
        "FunctionType": 3,
        "Domain": [0, 1],
        "Functions": [child],
        "Bounds": [],
        "Encode": [0, 1],
        "Range": [3, 7],
    }
    function = compile_pdf_function(dictionary)
    assert function(0) == (3,)
    assert function(0.5) == (5,)
    assert function(1) == (7,)
    del dictionary["Range"]
    function = compile_pdf_function(dictionary)
    assert function(0) == (2,)
    assert function(1) == (8,)


def test_stitching_function_checks_runtime_output_count_against_range() -> None:
    function = compile_pdf_function(
        {
            "FunctionType": 3,
            "Domain": [0, 1],
            "Functions": [{"FunctionType": 2, "Domain": [0, 1], "N": 1}],
            "Bounds": [],
            "Encode": [0, 1],
            "Range": [0, 1, 0, 1],
        }
    )
    with pytest.raises(ValueError, match="output count"):
        function(0.5)
