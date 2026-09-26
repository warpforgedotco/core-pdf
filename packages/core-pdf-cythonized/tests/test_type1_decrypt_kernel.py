# SPDX-License-Identifier: AGPL-3.0-only


import random

import pytest

from core_pdf_cythonized import decrypt_type1

program = pytest.importorskip("core_adobe_fonts.type1.program")


@pytest.mark.parametrize("key", [55665, 4330, 0, 1, 0xFFFF, 12345])
def test_decryption_is_the_packages(key: int) -> None:
    rng = random.Random(key)
    for length in [0, 1, 2, 3, 4, 17, 255, 256, 4097, *(rng.randrange(1, 3000) for _ in range(20))]:
        data = bytes(rng.randrange(256) for _ in range(length))
        expected = program.decrypt_type1(data, key)
        assert decrypt_type1(data, key) == expected
        assert decrypt_type1(bytearray(data), key) == expected
        assert decrypt_type1(memoryview(data), key) == expected
        assert type(decrypt_type1(data, key)) is bytes


def test_round_trip_through_the_cipher() -> None:
    plain = b"dup 0 15 RD \\x00\\x01 NP"
    state = 4330
    cipher = bytearray()
    for byte in plain:
        encrypted = byte ^ (state >> 8)
        cipher.append(encrypted)
        state = ((encrypted + state) * 52845 + 22719) & 0xFFFF
    assert decrypt_type1(bytes(cipher), 4330) == plain
