"""Filter and DecodeParms association preserves PDF stream pipeline order."""

import pytest

from core_pdf_spec.s_07_filters.decode_spec import StreamDecodeSpec, normalize_stream_decode_spec
from core_pdf_spec.s_07_filters.errors import FilterParseError
from core_pdf_spec.s_07_filters.pipeline import decode_stream_data
from core_pdf_spec.types import PdfName


@pytest.mark.parametrize("filters", [None, [], ()])
@pytest.mark.parametrize("params", [None, {}, [None], (None, None), 42])
def test_stream_without_filters_has_no_decode_parameters(filters: object, params: object) -> None:
    assert normalize_stream_decode_spec({"Filter": filters, "DecodeParms": params}) == (
        StreamDecodeSpec((), ())
    )


@pytest.mark.parametrize("dictionary", [{}, {"Filter": None, "DecodeParms": None}])
def test_absent_and_null_filter_entries_are_equivalent(dictionary: dict[str, object]) -> None:
    assert normalize_stream_decode_spec(dictionary) == StreamDecodeSpec((), ())


@pytest.mark.parametrize("filters", ["FlateDecode", ["FlateDecode"], ("FlateDecode",)])
def test_one_filter_retains_its_parameter_dictionary(filters: object) -> None:
    params = {"Columns": 2}
    spec = normalize_stream_decode_spec({"Filter": filters, "DecodeParms": params})
    assert spec.filters == ("FlateDecode",)
    assert spec.params[0] is params


@pytest.mark.parametrize("params", [None, (None, None), [None, None]])
def test_multiple_filters_have_one_parameter_slot_each(params: object) -> None:
    dictionary = {"Filter": [PdfName(b"ASCII85Decode"), b"/FlateDecode"], "DecodeParms": params}
    assert normalize_stream_decode_spec(dictionary) == StreamDecodeSpec(
        ("ASCII85Decode", "FlateDecode"), (None, None)
    )


@pytest.mark.parametrize("array_type", [list, tuple])
def test_parameter_arrays_preserve_null_slots_and_object_identity(array_type: type) -> None:
    params = {"Columns": 2}
    original = array_type([None, params])
    spec = normalize_stream_decode_spec(
        {"Filter": ["ASCII85Decode", "FlateDecode"], "DecodeParms": original}
    )
    assert spec.params[0] is None
    assert spec.params[1] is params
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
    assert normalize_stream_decode_spec({"Filter": "FlateDecode", "DecodeParms": 42}).params == (
        42,
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
    assert spec.filters == ("LZWDecode",)
    assert spec.params[0] is external
    dictionary["Filter"] = "FlateDecode"
    primary: dict[str, object] = {}
    dictionary["DecodeParms"] = primary
    spec = normalize_stream_decode_spec(dictionary)
    assert spec.filters == ("FlateDecode",)
    assert spec.params[0] is primary
    dictionary["Filter"] = []
    assert normalize_stream_decode_spec(dictionary) == StreamDecodeSpec((), ())


def test_filter_name_errors_precede_parameter_cardinality_errors() -> None:
    with pytest.raises(FilterParseError, match="^invalid stream decode filter$"):
        normalize_stream_decode_spec({"Filter": ["FlateDecode", 42], "DecodeParms": []})


@pytest.mark.parametrize("params", [(), (None, None)])
def test_direct_decode_spec_allows_default_or_explicit_parameter_slots(
    params: tuple[object, ...],
) -> None:
    spec = StreamDecodeSpec(("ASCIIHexDecode", "RunLengthDecode"), params)
    assert decode_stream_data(b"0250444680>", spec) == b"PDF"


def test_direct_decode_spec_rejects_mismatched_parameters() -> None:
    spec = StreamDecodeSpec(("ASCII85Decode", "FlateDecode"), (None,))
    with pytest.raises(FilterParseError, match="^invalid stream decode parameters$"):
        decode_stream_data(b"", spec)
