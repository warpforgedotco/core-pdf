"""Reader coercions remain core-owned when spec input contracts tighten."""

from core_pdf.impl._impl.graphics.color_spec import color_spec_from_value
from core_pdf.impl._impl.graphics.filter_registry import declared_filter_names
from core_pdf.impl._impl.graphics.functions import internal_compile_pdf_function
from core_pdf.impl._impl.graphics.shading import prepare_shading


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
