# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re

import pytest

from core_adobe_fonts.type1.program import (
    binary_entries,
    decode_charstring,
    decode_eexec_payload,
    decrypt_type1,
    eexec_ciphertext,
)


def test_type1_byte_primitives_reject_truncation_and_preserve_unencrypted_charstrings() -> None:
    assert decode_charstring(b"\x8b\x0e", -1) == b"\x8b\x0e"
    with pytest.raises(ValueError, match="prefix"):
        decode_charstring(b"\x00", 4)
    with pytest.raises(ValueError, match="odd"):
        decode_eexec_payload(b"0000000000", 1)
    with pytest.raises(ValueError, match="truncated"):
        list(binary_entries(b"/A 5 RD abc", re.compile(rb"/(\w+) (\d+) RD ")))


def test_tolerant_binary_entries_skip_truncated_entries() -> None:
    pattern = re.compile(rb"/(\w+) (\d+) RD ")
    data = b"/A 2 RD ab /B 9 RD cd"
    assert list(binary_entries(data, pattern, skip_truncated=True)) == [(b"A", b"ab")]


def test_eexec_ciphertext_is_the_payload_decode_eexec_payload_decrypts() -> None:
    data = b"%!FontType1 currentfile eexec 0a1b2c3d4e5f6071"
    ciphertext = eexec_ciphertext(data, None)
    assert ciphertext == bytes.fromhex("0a1b2c3d4e5f6071")
    assert decode_eexec_payload(data, None) == decrypt_type1(ciphertext, 55665)[4:]


def test_tolerant_eexec_ciphertext_recovers_what_strict_rejects() -> None:
    hex_payload = b"currentfile eexec 0a1b2c3d4e5f607"
    # An odd trailing hex digit is dropped rather than rejected.
    with pytest.raises(ValueError, match="odd"):
        eexec_ciphertext(hex_payload, None)
    assert eexec_ciphertext(hex_payload, None, tolerant=True) == bytes.fromhex("0a1b2c3d4e5f60")
    # A Length1 outside the data falls back to the eexec marker.
    with pytest.raises(ValueError, match="Length1"):
        eexec_ciphertext(hex_payload, 999)
    assert eexec_ciphertext(hex_payload, 999, tolerant=True) == bytes.fromhex("0a1b2c3d4e5f60")
    # Binary data whose first four bytes happen to be hex digits is only read
    # as hex when the wider sample agrees.
    binary = b"currentfile eexec abcd\x80\x81\x82"
    assert eexec_ciphertext(binary, None, tolerant=True) == b"abcd\x80\x81\x82"
