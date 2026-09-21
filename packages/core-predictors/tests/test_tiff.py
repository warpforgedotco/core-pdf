import pytest

from core_predictors.errors import PredictorError
from core_predictors.tiff import tiff_predict, tiff_predict_bits


def pack_samples(rows: list[list[int]], bits: int) -> bytes:
    packed = bytearray()
    for row in rows:
        binary = "".join(f"{sample:0{bits}b}" for sample in row)
        binary += "0" * (-len(binary) % 8)
        packed.extend(int(binary, 2).to_bytes(len(binary) // 8, "big"))
    return bytes(packed)


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("colors", [1, 3])
@pytest.mark.parametrize("columns", [1, 3, 8, 129])
def test_tiff_differences_wrap_per_component_and_restart_each_row(
    bits: int, colors: int, columns: int
) -> None:
    modulus = 1 << bits
    rows = [
        [(modulus - 1 - index * 3 + row * 7) % modulus for index in range(columns * colors)]
        for row in range(3)
    ]
    differences = [
        [
            value if index < colors else (value - row[index - colors]) % modulus
            for index, value in enumerate(row)
        ]
        for row in rows
    ]
    encoded, expected = pack_samples(differences, bits), pack_samples(rows, bits)
    assert (
        tiff_predict(memoryview(encoded), columns=columns, colors=colors, bits_per_component=bits)
        == expected
    )


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16])
def test_tiff_kernel_keeps_complete_rows_from_a_partial_stream(bits: int) -> None:
    row_size = (27 * bits + 7) // 8
    assert tiff_predict(b"", columns=9, colors=3, bits_per_component=bits) == b""
    data = bytes(row_size + 1)
    assert tiff_predict(data, columns=9, colors=3, bits_per_component=bits) == bytes(row_size)


@pytest.mark.parametrize("bits", [8, 16])
def test_tiff_word_kernels_preserve_empty_dimensions(bits: int) -> None:
    assert tiff_predict(b"abcd", columns=0, colors=1, bits_per_component=bits) == b""


def test_tiff_sub_byte_rows_stay_byte_aligned() -> None:
    encoded = pack_samples([[1, 1, 1], [2, 3, 4]], 4)
    assert tiff_predict_bits(encoded, columns=3, colors=1, bits=4) == pack_samples(
        [[1, 2, 3], [2, 5, 9]], 4
    )


def test_tiff_rejects_depths_outside_the_specification() -> None:
    with pytest.raises(PredictorError, match="invalid TIFF predictor bits 3"):
        tiff_predict(b"", columns=1, colors=1, bits_per_component=3)
