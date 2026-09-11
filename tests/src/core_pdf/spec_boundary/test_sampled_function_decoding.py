# SPDX-License-Identifier: AGPL-3.0-only
"""Reader stream decoding must not turn authentication failures into tint fallback."""

from typing import Any

import pytest

from core_pdf.impl._impl.graphics.functions import internal_compile_pdf_function
from core_pdf_spec.exceptions import PdfDecryptionError, PdfParseError, PdfUnsupportedError
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_syntax.stream import PdfStream


def internal_sampled(decoder: Any) -> PdfStream:
    dictionary = {
        "FunctionType": "0",
        "BitsPerSample": "8",
        "Size": ["2"],
        "Domain": [0, 1],
        "Range": [0, 1],
        "Filter": "ReaderOwnedFilter",
    }
    return PdfStream(dictionary, b"encoded", spec=dictionary, decoder=decoder)


def internal_wrap(stream: PdfStream, wrapper: str) -> object:
    if wrapper == "array":
        return [stream]
    if wrapper == "stitching":
        return {
            "FunctionType": 3,
            "Domain": [0, 1],
            "Functions": [stream],
            "Bounds": [],
            "Encode": [0, 1],
        }
    if wrapper == "recovered-stitching":
        return {
            "FunctionType": 3,
            "Domain": [0, 1],
            "Functions": [stream] * 3,
            "Bounds": [0.75, 0.25],
            "Encode": [0, 1] * 3,
        }
    return stream


@pytest.mark.parametrize("wrapper", ["direct", "array", "stitching", "recovered-stitching"])
@pytest.mark.parametrize("error_type", [PdfDecryptionError, RuntimeError])
def test_sampled_decoder_authentication_and_unexpected_failures_propagate(
    wrapper: str, error_type: type[Exception]
) -> None:
    failure = error_type("decoder failure must propagate")

    def decode(*args: object, **kwargs: object) -> bytes:
        raise failure

    with pytest.raises(error_type) as raised:
        internal_compile_pdf_function(internal_wrap(internal_sampled(decode), wrapper))
    assert raised.value is failure


@pytest.mark.parametrize("wrapper", ["direct", "array", "stitching", "recovered-stitching"])
@pytest.mark.parametrize(
    "error_type",
    [FilterParseError, FilterUnsupportedError, PdfParseError, PdfUnsupportedError, ValueError],
)
def test_known_sampled_decode_errors_retain_existing_numeric_fallback_boundary(
    wrapper: str, error_type: type[Exception]
) -> None:
    failure = error_type("controlled stream failure")

    def decode(*args: object, **kwargs: object) -> bytes:
        raise failure

    with pytest.raises(ValueError, match="invalid sampled PDF function") as raised:
        internal_compile_pdf_function(internal_wrap(internal_sampled(decode), wrapper))
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize("wrapper", ["direct", "array", "stitching"])
def test_sampled_reader_decodes_once_and_does_not_mutate_original_stream(wrapper: str) -> None:
    calls = []

    def decode(
        data: bytes | memoryview, dictionary: object, *, parent_dictionary: Any = None
    ) -> bytes:
        calls.append(None)
        assert data == b"encoded"
        assert dictionary is stream.spec
        # Preserve the reader's previous normalized-parent dictionary boundary.
        assert parent_dictionary["FunctionType"] == 0
        assert parent_dictionary["BitsPerSample"] == 8
        assert parent_dictionary["Size"] == (2,)
        return b"\0\xff"

    stream = internal_sampled(decode)
    evaluate = internal_compile_pdf_function(internal_wrap(stream, wrapper))
    assert evaluate(0.25) == (0.25,)
    assert evaluate(0.75) == (0.75,)
    assert len(calls) == 1
    assert stream.raw_data == b"encoded"
    assert stream.decoder is decode
    assert stream.spec is stream.dictionary
    assert stream.dictionary["FunctionType"] == "0"
    assert stream.dictionary["BitsPerSample"] == "8"
    assert stream.dictionary["Size"] == ["2"]


def test_malformed_sampled_payload_still_raises_a_controlled_numeric_error() -> None:
    def decode(*args: object, **kwargs: object) -> bytes:
        return b""

    with pytest.raises(ValueError, match="invalid sampled PDF function"):
        internal_compile_pdf_function(internal_sampled(decode))
