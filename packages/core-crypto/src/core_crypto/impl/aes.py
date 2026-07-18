# SPDX-License-Identifier: AGPL-3.0-only
from core_crypto.impl.crypto_constants import (
    AES_INV_SBOX,
    AES_RCON,
    AES_SBOX,
)


def rotate_left(x: int, n: int) -> int:
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def sub_word(w: int) -> int:
    return (
        (AES_SBOX[(w >> 24) & 0xFF] << 24)
        | (AES_SBOX[(w >> 16) & 0xFF] << 16)
        | (AES_SBOX[(w >> 8) & 0xFF] << 8)
        | AES_SBOX[w & 0xFF]
    )


def gf_mul(x: int, y: int) -> int:
    p = 0
    for ignored in range(8):
        if y & 1:
            p ^= x
        hi_bit_set = x & 0x80
        x = (x << 1) & 0xFF
        if hi_bit_set:
            x ^= 0x1B
        y >>= 1
    return p


SBOX_TABLE = bytes(AES_SBOX)
INV_SBOX_TABLE = bytes(AES_INV_SBOX)

GF_MUL_2 = bytes(gf_mul(2, x) for x in range(256))
GF_MUL_3 = bytes(gf_mul(3, x) for x in range(256))
GF_MUL_9 = bytes(gf_mul(9, x) for x in range(256))
GF_MUL_11 = bytes(gf_mul(11, x) for x in range(256))
GF_MUL_13 = bytes(gf_mul(13, x) for x in range(256))
GF_MUL_14 = bytes(gf_mul(14, x) for x in range(256))


def key_expansion(key: bytes) -> list[tuple[int, ...]]:
    key_size = len(key)
    Nk = key_size // 4
    Nr = 10 if key_size == 16 else 14
    Nb = 4

    round_key_words: list[int] = [0] * (Nb * (Nr + 1))

    for i in range(Nk):
        round_key_words[i] = int.from_bytes(key[i * 4 : (i + 1) * 4], "big")

    for i in range(Nk, Nb * (Nr + 1)):
        t = round_key_words[i - 1]
        if i % Nk == 0:
            t = sub_word(rotate_left(t, 8)) ^ AES_RCON[i // Nk]
        elif Nk > 6 and i % Nk == 4:
            t = sub_word(t)
        round_key_words[i] = round_key_words[i - Nk] ^ t

    all_bytes = b"".join(w.to_bytes(4, "big") for w in round_key_words)
    round_keys = [tuple(all_bytes[r * 16 : r * 16 + 16]) for r in range(Nr + 1)]
    return round_keys


def add_round_key(state: bytearray, round_key: tuple[int, ...]) -> None:
    for i in range(16):
        state[i] ^= round_key[i]


def sub_bytes(state: bytearray) -> None:
    state[:] = state.translate(SBOX_TABLE)


def shift_rows(state: bytearray) -> None:
    s = state[:]
    state[1], state[5], state[9], state[13] = s[5], s[9], s[13], s[1]
    state[2], state[6], state[10], state[14] = s[10], s[14], s[2], s[6]
    state[3], state[7], state[11], state[15] = s[15], s[3], s[7], s[11]


def mix_columns(state: bytearray) -> None:
    s = state[:]
    for i in range(0, 16, 4):
        s0, s1, s2, s3 = s[i : i + 4]
        state[i] = GF_MUL_2[s0] ^ GF_MUL_3[s1] ^ s2 ^ s3
        state[i + 1] = s0 ^ GF_MUL_2[s1] ^ GF_MUL_3[s2] ^ s3
        state[i + 2] = s0 ^ s1 ^ GF_MUL_2[s2] ^ GF_MUL_3[s3]
        state[i + 3] = GF_MUL_3[s0] ^ s1 ^ s2 ^ GF_MUL_2[s3]


def inv_sub_bytes(state: bytearray) -> None:
    state[:] = state.translate(INV_SBOX_TABLE)


def inv_shift_rows(state: bytearray) -> None:
    s = state[:]
    state[1], state[5], state[9], state[13] = s[13], s[1], s[5], s[9]
    state[2], state[6], state[10], state[14] = s[10], s[14], s[2], s[6]
    state[3], state[7], state[11], state[15] = s[7], s[11], s[15], s[3]


def inv_mix_columns(state: bytearray) -> None:
    s = state[:]
    for i in range(0, 16, 4):
        s0, s1, s2, s3 = s[i : i + 4]
        state[i] = GF_MUL_14[s0] ^ GF_MUL_11[s1] ^ GF_MUL_13[s2] ^ GF_MUL_9[s3]
        state[i + 1] = GF_MUL_9[s0] ^ GF_MUL_14[s1] ^ GF_MUL_11[s2] ^ GF_MUL_13[s3]
        state[i + 2] = GF_MUL_13[s0] ^ GF_MUL_9[s1] ^ GF_MUL_14[s2] ^ GF_MUL_11[s3]
        state[i + 3] = GF_MUL_11[s0] ^ GF_MUL_13[s1] ^ GF_MUL_9[s2] ^ GF_MUL_14[s3]


class AES:
    BLOCK_SIZE = 16

    def __init__(self, key: bytes) -> None:
        if len(key) not in (16, 32):
            raise ValueError(f"AES key must be 16 or 32 bytes, got {len(key)}")
        self.rounds = 10 if len(key) == 16 else 14
        self.round_keys: list[tuple[int, ...]] = key_expansion(key)

    def encrypt_inplace(self, state: bytearray) -> None:
        add_round_key(state, self.round_keys[0])
        for round_num in range(1, self.rounds):
            sub_bytes(state)
            shift_rows(state)
            mix_columns(state)
            add_round_key(state, self.round_keys[round_num])
        sub_bytes(state)
        shift_rows(state)
        add_round_key(state, self.round_keys[self.rounds])

    def decrypt_inplace(self, state: bytearray) -> None:
        add_round_key(state, self.round_keys[self.rounds])
        for round_num in range(self.rounds - 1, 0, -1):
            inv_shift_rows(state)
            inv_sub_bytes(state)
            add_round_key(state, self.round_keys[round_num])
            inv_mix_columns(state)
        inv_shift_rows(state)
        inv_sub_bytes(state)
        add_round_key(state, self.round_keys[0])

    def encrypt_block(self, block: bytes) -> bytes:
        state = bytearray(block)
        self.encrypt_inplace(state)
        return bytes(state)

    def decrypt_block(self, block: bytes) -> bytes:
        state = bytearray(block)
        self.decrypt_inplace(state)
        return bytes(state)

    def encrypt_cbc(self, iv: bytes, plaintext: bytes, padding: bool = True) -> bytes:
        if padding:
            pad_len = self.BLOCK_SIZE - (len(plaintext) % self.BLOCK_SIZE)
            padded = plaintext + bytes([pad_len] * pad_len)
        else:
            if len(plaintext) % self.BLOCK_SIZE != 0:
                raise ValueError(
                    "Plaintext length must be a multiple of block size when padding is disabled"
                )
            padded = plaintext

        ciphertext = bytearray(len(padded))
        prev = bytearray(iv)
        state = bytearray(self.BLOCK_SIZE)

        encrypt_inplace = self.encrypt_inplace

        for i in range(0, len(padded), self.BLOCK_SIZE):
            for j in range(16):
                state[j] = padded[i + j] ^ prev[j]

            encrypt_inplace(state)
            ciphertext[i : i + 16] = state
            prev[:] = state

        return bytes(ciphertext)

    def decrypt_cbc(self, iv: bytes, ciphertext: bytes, padding: bool = True) -> bytes:
        if len(ciphertext) % self.BLOCK_SIZE != 0:
            raise ValueError("Ciphertext length must be a multiple of the block size")

        plaintext = bytearray(len(ciphertext))
        prev = bytearray(iv)
        state = bytearray(self.BLOCK_SIZE)

        decrypt_inplace = self.decrypt_inplace

        for i in range(0, len(ciphertext), self.BLOCK_SIZE):
            state[:] = ciphertext[i : i + self.BLOCK_SIZE]
            block_copy = bytes(state)
            decrypt_inplace(state)

            for j in range(16):
                plaintext[i + j] = state[j] ^ prev[j]

            prev[:] = block_copy

        if padding:
            if not plaintext:
                raise ValueError("Invalid PKCS7 padding")
            pad_len = plaintext[-1]
            if pad_len > self.BLOCK_SIZE or pad_len == 0:
                raise ValueError("Invalid PKCS7 padding")
            if plaintext[-pad_len:] != bytes([pad_len] * pad_len):
                raise ValueError("Invalid PKCS7 padding")
            return bytes(plaintext[:-pad_len])

        return bytes(plaintext)
