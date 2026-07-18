# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import math

IDCT_SCALE = tuple(1 / math.sqrt(2) if i == 0 else 1.0 for i in range(8))
IDCT_COS = tuple(
    tuple(math.cos(((2 * x + 1) * u * math.pi) / 16.0) for u in range(8)) for x in range(8)
)
IDCT_BASIS = tuple(tuple(IDCT_SCALE[u] * IDCT_COS[x][u] for u in range(8)) for x in range(8))


def idct_2d(block: list[int], temp: list[float]) -> list[int]:
    basis = IDCT_BASIS
    cos_table = IDCT_COS
    scale = IDCT_SCALE
    for v in range(8):
        row = v * 8
        b0 = block[row]
        b1 = block[row + 1]
        b2 = block[row + 2]
        b3 = block[row + 3]
        b4 = block[row + 4]
        b5 = block[row + 5]
        b6 = block[row + 6]
        b7 = block[row + 7]
        sv = scale[v]
        bx = basis[0]
        temp[row] = sv * (
            b0 * bx[0]
            + b1 * bx[1]
            + b2 * bx[2]
            + b3 * bx[3]
            + b4 * bx[4]
            + b5 * bx[5]
            + b6 * bx[6]
            + b7 * bx[7]
        )
        bx = basis[1]
        temp[row + 1] = sv * (
            b0 * bx[0]
            + b1 * bx[1]
            + b2 * bx[2]
            + b3 * bx[3]
            + b4 * bx[4]
            + b5 * bx[5]
            + b6 * bx[6]
            + b7 * bx[7]
        )
        bx = basis[2]
        temp[row + 2] = sv * (
            b0 * bx[0]
            + b1 * bx[1]
            + b2 * bx[2]
            + b3 * bx[3]
            + b4 * bx[4]
            + b5 * bx[5]
            + b6 * bx[6]
            + b7 * bx[7]
        )
        bx = basis[3]
        temp[row + 3] = sv * (
            b0 * bx[0]
            + b1 * bx[1]
            + b2 * bx[2]
            + b3 * bx[3]
            + b4 * bx[4]
            + b5 * bx[5]
            + b6 * bx[6]
            + b7 * bx[7]
        )
        bx = basis[4]
        temp[row + 4] = sv * (
            b0 * bx[0]
            + b1 * bx[1]
            + b2 * bx[2]
            + b3 * bx[3]
            + b4 * bx[4]
            + b5 * bx[5]
            + b6 * bx[6]
            + b7 * bx[7]
        )
        bx = basis[5]
        temp[row + 5] = sv * (
            b0 * bx[0]
            + b1 * bx[1]
            + b2 * bx[2]
            + b3 * bx[3]
            + b4 * bx[4]
            + b5 * bx[5]
            + b6 * bx[6]
            + b7 * bx[7]
        )
        bx = basis[6]
        temp[row + 6] = sv * (
            b0 * bx[0]
            + b1 * bx[1]
            + b2 * bx[2]
            + b3 * bx[3]
            + b4 * bx[4]
            + b5 * bx[5]
            + b6 * bx[6]
            + b7 * bx[7]
        )
        bx = basis[7]
        temp[row + 7] = sv * (
            b0 * bx[0]
            + b1 * bx[1]
            + b2 * bx[2]
            + b3 * bx[3]
            + b4 * bx[4]
            + b5 * bx[5]
            + b6 * bx[6]
            + b7 * bx[7]
        )
    for y in range(8):
        cy = cos_table[y]
        row = y * 8
        for x in range(8):
            total = (
                temp[x] * cy[0]
                + temp[8 + x] * cy[1]
                + temp[16 + x] * cy[2]
                + temp[24 + x] * cy[3]
                + temp[32 + x] * cy[4]
                + temp[40 + x] * cy[5]
                + temp[48 + x] * cy[6]
                + temp[56 + x] * cy[7]
            )
            value = int(total * 0.25 + (0.5 if total >= 0.0 else -0.5))
            if value < -128:
                block[row + x] = -128
            elif value > 127:
                block[row + x] = 127
            else:
                block[row + x] = value
    return block
