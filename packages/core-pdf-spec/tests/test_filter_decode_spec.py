"""Filter and DecodeParms association preserves PDF stream pipeline order."""

import pytest

from core_pdf_spec.s_07_filters.decode_spec import (
    FilterStep,
    StreamDecodeSpec,
    normalize_stream_decode_spec,
)
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_filters.pipeline import decode_stream_data
from core_pdf_spec.types import PdfName


@pytest.mark.parametrize("filters", [None, [], ()])
@pytest.mark.parametrize("params", [None, {}, [None], (None, None), 42])
def test_stream_without_filters_has_no_decode_parameters(filters: object, params: object) -> None:
    assert normalize_stream_decode_spec({"Filter": filters, "DecodeParms": params}) == (
        StreamDecodeSpec(())
    )


@pytest.mark.parametrize("dictionary", [{}, {"Filter": None, "DecodeParms": None}])
def test_absent_and_null_filter_entries_are_equivalent(dictionary: dict[str, object]) -> None:
    assert normalize_stream_decode_spec(dictionary) == StreamDecodeSpec(())


@pytest.mark.parametrize("filters", ["FlateDecode", ["FlateDecode"], ("FlateDecode",)])
def test_one_filter_retains_its_parameter_dictionary(filters: object) -> None:
    params = {"Columns": 2}
    spec = normalize_stream_decode_spec({"Filter": filters, "DecodeParms": params})
    assert tuple(step.name for step in spec.steps) == ("FlateDecode",)
    assert spec.steps[0].params is params


@pytest.mark.parametrize("params", [None, (None, None), [None, None]])
def test_multiple_filters_have_one_parameter_slot_each(params: object) -> None:
    dictionary = {"Filter": [PdfName(b"ASCII85Decode"), b"FlateDecode"], "DecodeParms": params}
    assert normalize_stream_decode_spec(dictionary) == StreamDecodeSpec(
        (FilterStep("ASCII85Decode"), FilterStep("FlateDecode"))
    )


@pytest.mark.parametrize("array_type", [list, tuple])
def test_parameter_arrays_preserve_null_slots_and_object_identity(array_type: type) -> None:
    params = {"Columns": 2}
    original = array_type([None, params])
    spec = normalize_stream_decode_spec(
        {"Filter": ["ASCII85Decode", "FlateDecode"], "DecodeParms": original}
    )
    assert spec.steps[0].params is None
    assert spec.steps[1].params is params
    assert original == array_type([None, {"Columns": 2}])


@pytest.mark.parametrize(
    ("filters", "params"),
    [
        ("FlateDecode", []),
        ("FlateDecode", (None, None)),
        (["ASCII85Decode", "FlateDecode"], {}),
        (["ASCII85Decode", "FlateDecode"], []),
        (["ASCII85Decode", "FlateDecode"], [None]),
        (["ASCII85Decode", "FlateDecode"], (None, None, None)),
    ],
)
def test_decode_parameters_require_exact_filter_cardinality(
    filters: object, params: object
) -> None:
    # ISO 32000-1, Table 5: a DecodeParms array corresponds to the Filter array.
    with pytest.raises(FilterParseError, match="^invalid stream decode parameters$"):
        normalize_stream_decode_spec({"Filter": filters, "DecodeParms": params})


def test_normalization_leaves_parameter_validation_to_the_filter() -> None:
    assert normalize_stream_decode_spec({"Filter": "FlateDecode", "DecodeParms": 42}).steps == (
        FilterStep("FlateDecode", 42),
    )


def test_external_filter_entries_only_replace_null_primary_entries() -> None:
    external = {"EarlyChange": 0}
    dictionary: dict[str, object] = {
        "Filter": None,
        "DecodeParms": None,
        "FFilter": "LZWDecode",
        "FDecodeParms": external,
    }
    spec = normalize_stream_decode_spec(dictionary)
    assert tuple(step.name for step in spec.steps) == ("LZWDecode",)
    assert spec.steps[0].params is external
    dictionary["Filter"] = "FlateDecode"
    primary: dict[str, object] = {}
    dictionary["DecodeParms"] = primary
    spec = normalize_stream_decode_spec(dictionary)
    assert tuple(step.name for step in spec.steps) == ("FlateDecode",)
    assert spec.steps[0].params is primary
    dictionary["Filter"] = []
    assert normalize_stream_decode_spec(dictionary) == StreamDecodeSpec(())


def test_filter_name_errors_precede_parameter_cardinality_errors() -> None:
    with pytest.raises(FilterParseError, match="^invalid stream decode filter$"):
        normalize_stream_decode_spec({"Filter": ["FlateDecode", 42], "DecodeParms": []})


def test_direct_decode_steps_keep_default_parameters_and_pipeline_order() -> None:
    spec = StreamDecodeSpec((FilterStep("ASCIIHexDecode"), FilterStep("RunLengthDecode")))
    assert decode_stream_data(b"0250444680>", spec) == b"PDF"


@pytest.mark.parametrize("value", [PdfName(b"/FlateDecode"), "/FlateDecode"])
def test_filter_names_preserve_decoded_leading_slashes(value: object) -> None:
    assert normalize_stream_decode_spec({"Filter": value}).steps == (FilterStep("/FlateDecode"),)
    with pytest.raises(FilterUnsupportedError, match="not implemented"):
        decode_stream_data(b"", {"Filter": value})


def test_stream_decode_spec_requires_dictionary():
    with pytest.raises(FilterParseError, match="invalid stream dictionary"):
        normalize_stream_decode_spec([])
