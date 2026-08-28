# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import numpy

from core_pdf.impl.spec.s_07_security.aes import AES
from core_pdf.impl.spec.s_07_security.rc4 import CryptRC4

PLAINTEXT = numpy.random.default_rng(1).bytes(64 * 1024)
RC4_KEY = b"stream-object-key-0123456789"

AES_128_KEY = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
AES_IV = bytes(16)

AES_128 = AES(AES_128_KEY)
AES_256 = AES(bytes(32))
AES_128_CIPHERTEXT = AES_128.encrypt_cbc(AES_IV, PLAINTEXT)


def test_rc4_encrypt_benchmark(benchmark) -> None:
    result = benchmark(lambda: CryptRC4(RC4_KEY).encrypt(PLAINTEXT))
    assert len(result) == len(PLAINTEXT)


def test_aes_128_cbc_decrypt_benchmark(benchmark) -> None:
    result = benchmark(AES_128.decrypt_cbc, AES_IV, AES_128_CIPHERTEXT)
    assert result == PLAINTEXT


def test_aes_256_cbc_encrypt_benchmark(benchmark) -> None:
    result = benchmark(AES_256.encrypt_cbc, AES_IV, PLAINTEXT)
    assert len(result) >= len(PLAINTEXT)
