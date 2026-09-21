# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re

import pytest

from core_adobe_fonts.type1.program import (
    binary_entries,
    decode_charstring,
    decode_eexec_payload,
)


def test_type1_byte_primitives_reject_truncation_and_preserve_unencrypted_charstrings() -> None:
    assert decode_charstring(b"\x8b\x0e", -1) == b"\x8b\x0e"
    with pytest.raises(ValueError, match="prefix"):
        decode_charstring(b"\x00", 4)
    with pytest.raises(ValueError, match="odd"):
        decode_eexec_payload(b"0000000000", 1)
    with pytest.raises(ValueError, match="truncated"):
        list(binary_entries(b"/A 5 RD abc", re.compile(rb"/(\w+) (\d+) RD ")))
