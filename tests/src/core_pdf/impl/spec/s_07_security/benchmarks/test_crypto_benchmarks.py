# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import numpy

from core_pdf.impl.spec.s_07_security.ciphers import (
    internal_aes_cbc_decrypt,
    internal_aes_cbc_encrypt,
    internal_rc4_crypt,
)

PLAINTEXT = numpy.random.default_rng(1).bytes(64 * 1024)
RC4_KEY = bytes(16)

AES_128_KEY = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
AES_IV = bytes(16)

AES_128_CIPHERTEXT = internal_aes_cbc_encrypt(
    AES_128_KEY,
    AES_IV,
    PLAINTEXT,
    use_padding=True,
)


def test_rc4_encrypt_benchmark(benchmark) -> None:
    result = benchmark(internal_rc4_crypt, RC4_KEY, PLAINTEXT)
    assert len(result) == len(PLAINTEXT)


def test_aes_128_cbc_decrypt_benchmark(benchmark) -> None:
    result = benchmark(
        internal_aes_cbc_decrypt,
        AES_128_KEY,
        AES_IV,
        AES_128_CIPHERTEXT,
        use_padding=True,
    )
    assert result == PLAINTEXT


def test_aes_256_cbc_encrypt_benchmark(benchmark) -> None:
    result = benchmark(
        internal_aes_cbc_encrypt,
        bytes(32),
        AES_IV,
        PLAINTEXT,
        use_padding=True,
    )
    assert len(result) >= len(PLAINTEXT)
