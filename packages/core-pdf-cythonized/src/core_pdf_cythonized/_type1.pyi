# SPDX-License-Identifier: AGPL-3.0-only

def decrypt_type1(data: bytes | bytearray | memoryview, key: int) -> bytes: ...
def type1_glyph_bounds(
    charstring: bytes,
    subrs: tuple[bytes, ...],
    matrix: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float] | tuple[()] | None: ...
