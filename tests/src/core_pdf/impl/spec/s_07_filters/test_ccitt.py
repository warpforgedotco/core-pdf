from __future__ import annotations

import numpy

from core_pdf.impl.spec.s_07_filters.decode_spec import FilterParams
from core_pdf.impl.spec.s_07_filters.decoders import (
    decode_ccitt_fax,
    decode_ccitt_fax_image,
)


def test_ccitt_filter_repacks_numpy_rows_to_msb_first_bytes() -> None:
    # Four white pixels followed by four black pixels.
    encoded = bytes([0b10110110])
    params = FilterParams(columns=8, rows=1, k=0, has_columns=True)

    assert decode_ccitt_fax(encoded, params) == b"\xf0"


def test_ccitt_native_decoder_reuses_output_buffer() -> None:
    encoded = bytes([0b10110110])
    params = FilterParams(columns=8, rows=1, k=0, has_columns=True)
    output = numpy.empty((1, 8), dtype=numpy.uint8)

    decoded = decode_ccitt_fax_image(encoded, params, out=output)

    assert decoded is output
    assert decoded.tolist() == [[255, 255, 255, 255, 0, 0, 0, 0]]
