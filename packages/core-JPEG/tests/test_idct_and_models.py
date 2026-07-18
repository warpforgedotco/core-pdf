from core_jpeg.impl.codecs.dct.idct import idct_2d
from core_jpeg.impl.models import DecodeWorkload


def transform(block: list[int]) -> list[int]:
    return idct_2d(block, [0.0] * 64)


def test_idct_zero_block_stays_zero() -> None:
    assert transform([0] * 64) == [0] * 64


def test_idct_dc_coefficient_produces_flat_block() -> None:
    block = [0] * 64
    block[0] = 80

    assert transform(block) == [10] * 64


def test_idct_clamps_output_range() -> None:
    positive = [0] * 64
    positive[0] = 10_000
    negative = [0] * 64
    negative[0] = -10_000

    assert transform(positive) == [127] * 64
    assert transform(negative) == [-128] * 64


def test_decode_workload_normalizes_features() -> None:
    workload = DecodeWorkload(codec="dct", features={"progressive", "restart-markers"})

    assert workload.features == frozenset({"progressive", "restart-markers"})
