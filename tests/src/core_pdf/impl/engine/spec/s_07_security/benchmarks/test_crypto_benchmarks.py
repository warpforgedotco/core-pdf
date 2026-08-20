# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import numpy
import pytest

from core_pdf.impl.engine.spec.s_07_security.aes import AES, key_expansion
from core_pdf.impl.engine.spec.s_07_security.rc4 import CryptRC4

pytestmark = pytest.mark.benchmark_high_impact

PLAINTEXT = numpy.random.default_rng(1).bytes(64 * 1024)
RC4_KEY = b"stream-object-key-0123456789"

AES_128_KEY = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
AES_256_KEY = bytes(32)
AES_IV = bytes(16)

AES_128 = AES(AES_128_KEY)
AES_256 = AES(AES_256_KEY)
AES_128_CIPHERTEXT = AES_128.encrypt_cbc(AES_IV, PLAINTEXT)


def test_rc4_encrypt_benchmark(benchmark) -> None:
    result = benchmark(lambda: CryptRC4(RC4_KEY).encrypt(PLAINTEXT))
    assert len(result) == len(PLAINTEXT)


def test_aes_key_expansion_128_benchmark(benchmark) -> None:
    result = benchmark(key_expansion, AES_128_KEY)
    assert len(result) == 11


def test_aes_key_expansion_256_benchmark(benchmark) -> None:
    result = benchmark(key_expansion, AES_256_KEY)
    assert len(result) == 15


def test_aes_128_cbc_encrypt_benchmark(benchmark) -> None:
    result = benchmark(AES_128.encrypt_cbc, AES_IV, PLAINTEXT)
    assert len(result) >= len(PLAINTEXT)


def test_aes_128_cbc_decrypt_benchmark(benchmark) -> None:
    result = benchmark(AES_128.decrypt_cbc, AES_IV, AES_128_CIPHERTEXT)
    assert result == PLAINTEXT


def test_aes_256_cbc_encrypt_benchmark(benchmark) -> None:
    result = benchmark(AES_256.encrypt_cbc, AES_IV, PLAINTEXT)
    assert len(result) >= len(PLAINTEXT)
