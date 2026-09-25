"""A run of PNG Up rows decodes as each row added to the one above (ISO 32000-2 7.4.4.4)."""

import random

import pytest

from core_pdf_spec.s_07_filters.predictors import png_predict


def up_by_row(rows: list[bytes]) -> bytes:
    previous = bytes(len(rows[0])) if rows else b""
    out = bytearray()
    for row in rows:
        previous = bytes((a + b) & 0xFF for a, b in zip(row, previous, strict=True))
        out += previous
    return bytes(out)


@pytest.mark.parametrize("seed", range(10))
def test_every_row_up_sums_down_the_columns(seed: int) -> None:
    rng = random.Random(seed)
    width = rng.randint(1, 8)
    rows = [rng.randbytes(width) for _ in range(rng.randint(1, 300))]
    encoded = b"".join(b"\x02" + row for row in rows)
    decoded = png_predict(encoded, columns=width, colors=1, bits_per_component=8)
    assert decoded == up_by_row(rows)


def test_a_mixed_run_still_decodes_row_by_row() -> None:
    encoded = b"\x02\x01\x02" + b"\x00\x05\x06" + b"\x02\x01\x01"
    assert png_predict(encoded, columns=2, colors=1, bits_per_component=8) == bytes(
        [1, 2, 5, 6, 6, 7]
    )


def test_no_rows_decode_to_nothing() -> None:
    assert png_predict(b"", columns=3, colors=1, bits_per_component=8) == b""
