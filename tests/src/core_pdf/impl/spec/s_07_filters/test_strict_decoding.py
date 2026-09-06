# SPDX-License-Identifier: AGPL-3.0-only
"""Strict PDF decoding and application recovery remain separate contracts."""

import pytest

from core_pdf.impl._impl.graphics.decode_compat import FilterParams as CompatibleFilterParams
from core_pdf.impl._impl.graphics.stream_decoding import decode_stream_data as recover_stream
from core_pdf.impl.spec.s_07_filters.decode_spec import FilterParams
from core_pdf.impl.spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf.impl.spec.s_07_filters.pipeline import decode_stream_data


def test_ascii_hex_rejects_non_whitespace_garbage_while_application_recovers() -> None:
    source = b"6!1>"
    with pytest.raises(ValueError):
        decode_stream_data(source, {"Filter": "ASCIIHexDecode"})
    assert recover_stream(source, {"Filter": "ASCIIHexDecode"}) == b"a"


def test_mislabeled_flate_is_only_recovered_by_application_decoder() -> None:
    source = b"0 0 m 1 1 l S"
    with pytest.raises(FilterParseError):
        decode_stream_data(source, {"Filter": "FlateDecode"})
    assert recover_stream(source, {"Filter": "FlateDecode"}) == source


def test_filter_spelling_repair_is_only_an_application_policy() -> None:
    source = b"0 0 m 1 1 l S"
    with pytest.raises(FilterUnsupportedError):
        decode_stream_data(source, {"Filter": "platedecode"})
    assert recover_stream(source, {"Filter": "platedecode"}) == source


@pytest.mark.parametrize("null_filter", [None, "null"])
def test_null_filter_array_entries_are_only_skipped_by_application_policy(
    null_filter: object,
) -> None:
    dictionary = {"Filter": [null_filter, "ASCIIHexDecode"], "DecodeParms": [None, None]}
    with pytest.raises((FilterParseError, FilterUnsupportedError)):
        decode_stream_data(b"61>", dictionary)
    assert recover_stream(b"61>", dictionary) == b"a"


def test_damage_count_has_an_integer_specification_type() -> None:
    with pytest.raises(ValueError):
        FilterParams.from_parms({"DamagedRowsBeforeError": True})
    assert (
        CompatibleFilterParams.from_parms(
            {"DamagedRowsBeforeError": True}
        ).damaged_rows_before_error
        == 1
    )


def test_png_damage_tolerance_is_kept_outside_pdf_predictor_semantics() -> None:
    import zlib

    dictionary = {
        "Filter": "FlateDecode",
        "DecodeParms": {"Predictor": 12, "Columns": 2, "DamagedRowsBeforeError": 1},
    }
    source = zlib.compress(b"\x00ab\x00c")
    with pytest.raises(FilterParseError):
        decode_stream_data(source, dictionary)
    assert recover_stream(source, dictionary) == b"ab"
