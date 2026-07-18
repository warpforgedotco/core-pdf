# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

QE = (
    0x5601,
    0x3401,
    0x1801,
    0x0AC1,
    0x0521,
    0x0221,
    0x5601,
    0x5401,
    0x4801,
    0x3801,
    0x3001,
    0x2401,
    0x1C01,
    0x1601,
    0x5601,
    0x5401,
    0x5101,
    0x4801,
    0x3801,
    0x3401,
    0x3001,
    0x2801,
    0x2401,
    0x2201,
    0x1C01,
    0x1801,
    0x1601,
    0x1401,
    0x1201,
    0x1101,
    0x0AC1,
    0x09C1,
    0x08A1,
    0x0521,
    0x0441,
    0x02A1,
    0x0221,
    0x0141,
    0x0111,
    0x0085,
    0x0049,
    0x0025,
    0x0015,
    0x0009,
    0x0005,
    0x0001,
    0x5601,
)

NMPS = (
    1,
    2,
    3,
    4,
    5,
    38,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    29,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    45,
    46,
)

NLPS = (
    1,
    6,
    9,
    12,
    29,
    33,
    6,
    14,
    14,
    14,
    17,
    18,
    20,
    21,
    14,
    14,
    15,
    16,
    17,
    18,
    19,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    46,
)

SWITCH = (
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
)

DEFAULT_CONTEXTS = (4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 46)


class MQDecoder:
    __slots__ = ("data", "pos", "a", "c", "ct", "ctx_index", "ctx_mps")

    def __init__(
        self,
        data: bytes,
        contexts: tuple[list[int], list[int]] | None = None,
    ) -> None:
        self.data = data
        self.pos = 0
        if contexts is None:
            self.ctx_index = []
            self.ctx_mps = []
            self.reset_contexts()
        else:
            ctx_index, ctx_mps = contexts
            self.ctx_index = list(ctx_index)
            self.ctx_mps = list(ctx_mps)
        self.a = 0x8000
        self.c = 0
        self.ct = 0
        self.init_bytes()

    def reset_contexts(self) -> None:
        self.ctx_index = list(DEFAULT_CONTEXTS)
        self.ctx_mps = [0] * len(DEFAULT_CONTEXTS)

    def contexts(self) -> tuple[list[int], list[int]]:
        return list(self.ctx_index), list(self.ctx_mps)

    def byte_at(self, index: int) -> int:
        return self.data[index] if 0 <= index < len(self.data) else 0xFF

    def init_bytes(self) -> None:
        self.pos = 0
        self.c = self.byte_at(0) << 16 if self.data else 0xFF << 16
        self.byte_in()
        self.c <<= 7
        self.ct -= 7
        self.a = 0x8000

    def byte_in(self) -> None:
        current = self.byte_at(self.pos)
        nxt = self.byte_at(self.pos + 1)
        if current == 0xFF:
            if nxt > 0x8F:
                self.c += 0xFF00
                self.ct = 8
            else:
                self.pos += 1
                self.c += nxt << 9
                self.ct = 7
        else:
            self.pos += 1
            self.c += nxt << 8
            self.ct = 8

    def renormalize(self) -> None:
        while self.a < 0x8000:
            if self.ct == 0:
                self.byte_in()
            self.c <<= 1
            self.a <<= 1
            self.ct -= 1

    def decode(self, cx: int) -> int:
        ctx_index = self.ctx_index
        ctx_mps = self.ctx_mps
        idx = ctx_index[cx]
        mps = ctx_mps[cx]
        qe = QE[idx]
        self.a -= qe
        if (self.c >> 16) < qe:
            if self.a < qe:
                self.a = qe
                value = mps
                ctx_index[cx] = NMPS[idx]
            else:
                self.a = qe
                value = mps ^ 1
                if SWITCH[idx]:
                    mps ^= 1
                ctx_index[cx] = NLPS[idx]
                ctx_mps[cx] = mps
            self.renormalize()
            return value
        self.c -= qe << 16
        if self.a < 0x8000:
            if self.a < qe:
                value = mps ^ 1
                if SWITCH[idx]:
                    mps ^= 1
                ctx_index[cx] = NLPS[idx]
                ctx_mps[cx] = mps
            else:
                value = mps
                ctx_index[cx] = NMPS[idx]
            self.renormalize()
            return value
        return mps


class RawBitDecoder:
    __slots__ = ("data", "pos", "current", "bits_left")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.current = 0
        self.bits_left = 0

    def byte_at(self, index: int) -> int:
        return self.data[index] if 0 <= index < len(self.data) else 0xFF

    def decode_bit(self) -> int:
        if self.bits_left == 0:
            if self.current == 0xFF:
                nxt = self.byte_at(self.pos)
                if nxt > 0x8F:
                    self.current = 0xFF
                    self.bits_left = 8
                else:
                    self.current = nxt
                    self.pos += 1
                    self.bits_left = 7
            else:
                self.current = self.byte_at(self.pos)
                self.pos += 1
                self.bits_left = 8
        self.bits_left -= 1
        return (self.current >> self.bits_left) & 1


def default_t1_contexts() -> list[tuple[int, int]]:
    return [(index, 0) for index in DEFAULT_CONTEXTS]


def default_t1_contexts_split() -> tuple[list[int], list[int]]:
    return list(DEFAULT_CONTEXTS), [0] * len(DEFAULT_CONTEXTS)
