# SPDX-License-Identifier: AGPL-3.0-only
"""Decode-parameter conformance rules from ISO 32000-1 clause 7.4.

Each test names the clause and table it pins.
"""

from __future__ import annotations

import pytest

from core_pdf.impl.spec.s_07_filters.decode_spec import (
    FilterParams,
    normalize_stream_decode_spec,
)
from core_pdf.impl.spec.s_07_filters.decoders import decode_ccitt_fax, decode_jbig2
from core_pdf.impl.spec.s_07_filters.errors import FilterParseError
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream

# 16 rows of Group 4 V0 codes: an all-white 16x16 image.
ALL_WHITE_G4 = b"\xff\xff"
CCITT_PARMS = {"K": -1, "Columns": 16, "Rows": 16}


def test_black_is_1_reverses_the_sample_polarity() -> None:
    """Table 11: BlackIs1 is "the reverse of the normal PDF convention"."""
    default = decode_ccitt_fax(ALL_WHITE_G4, dict(CCITT_PARMS))
    reversed_polarity = decode_ccitt_fax(ALL_WHITE_G4, {**CCITT_PARMS, "BlackIs1": True})

    # The normal convention writes 1 for white; BlackIs1 writes 1 for black.
    assert set(default) == {0xFF}
    assert set(reversed_polarity) == {0x00}


def test_black_is_1_defaults_to_false() -> None:
    """Table 11: "Default value: false"."""
    assert FilterParams.from_parms({}).black_is_1 is False
    assert decode_ccitt_fax(ALL_WHITE_G4, dict(CCITT_PARMS)) == decode_ccitt_fax(
        ALL_WHITE_G4, {**CCITT_PARMS, "BlackIs1": False}
    )


@pytest.mark.parametrize("tolerance", [0, 1, 2, 4, 100])
def test_damaged_rows_before_error_is_an_integer(tolerance: int) -> None:
    """Table 11 types it as "integer ... The number of damaged rows ... tolerated".

    Parsing it as a boolean refused every conforming file stating a count above 1.
    """
    assert (
        FilterParams.from_parms({"DamagedRowsBeforeError": tolerance}).damaged_rows_before_error
        == tolerance
    )


def test_damaged_rows_before_error_still_accepts_the_boolean_form() -> None:
    """Leniency for writers that emit `true`; it means the same as a nonzero count."""
    assert FilterParams.from_parms({"DamagedRowsBeforeError": True}).damaged_rows_before_error == 1
    assert FilterParams.from_parms({"DamagedRowsBeforeError": False}).damaged_rows_before_error == 0


def test_jbig2_globals_may_be_a_stream() -> None:
    """Table 12 types JBIG2Globals as a *stream*, which is its only legal form.

    Both spellings must reach the JBIG2 parser identically; the bytes here are
    not real JBIG2, so both raise the same parser error rather than a
    parameter error.
    """
    payload = b"\x00" * 4
    globals_bytes = b"GLOBALSBYTES"
    stream = PdfStream({}, b"", decoded_data=globals_bytes)

    with pytest.raises(FilterParseError) as from_stream:
        decode_jbig2(payload, {"JBIG2Globals": stream})
    with pytest.raises(FilterParseError) as from_bytes:
        decode_jbig2(payload, {"JBIG2Globals": globals_bytes})

    assert str(from_stream.value) == str(from_bytes.value)
    assert "globals" not in str(from_stream.value)


def test_ffilter_is_preferred_over_the_f_file_specification() -> None:
    """Table 5: on a regular stream /F is a file specification and /FFilter names its filters."""
    spec = normalize_stream_decode_spec(
        {"F": b"external.dat", "FFilter": "FlateDecode"},
    )

    assert spec.filters == ("FlateDecode",)
