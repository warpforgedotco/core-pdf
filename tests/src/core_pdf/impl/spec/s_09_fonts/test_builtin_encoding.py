# SPDX-License-Identifier: AGPL-3.0-only
"""A font program's built-in encoding is the implicit base encoding.

Table 114: when /BaseEncoding is absent, "for a font program that is embedded
in the PDF file, the implicit base encoding shall be the font program's
built-in encoding", and /Differences describes changes from that. core-pdf
could read it back out of a Type 1 program but not out of a CFF one, so
CFF-embedded fonts silently fell back to a standard table.
"""

from __future__ import annotations

import struct

from core_pdf.impl.spec.s_09_fonts.font_program import CFFFont
from tests.helpers.cff import build_cff


def test_reads_a_format_0_custom_encoding() -> None:
    # Format 0: code array, one entry per glyph starting at glyph 1.
    encoding = bytes([0, 3]) + bytes([0x41, 0x42, 0x43])
    font = CFFFont(build_cff(encoding, ["alpha", "beta", "gamma"]))
    assert font.builtin_encoding() == {0x41: "alpha", 0x42: "beta", 0x43: "gamma"}


def test_reads_a_format_1_range_encoding() -> None:
    # Format 1: one range of three sequential codes from 0x61.
    encoding = bytes([1, 1]) + bytes([0x61, 2])
    font = CFFFont(build_cff(encoding, ["alpha", "beta", "gamma"]))
    assert font.builtin_encoding() == {0x61: "alpha", 0x62: "beta", 0x63: "gamma"}


def test_reads_supplements_appended_to_an_encoding() -> None:
    # High bit of the format byte means supplements follow; each gives a
    # second code for an already-encoded glyph, keyed by SID.
    encoding = (
        bytes([0x80, 2]) + bytes([0x41, 0x42]) + bytes([1]) + bytes([0x5A]) + struct.pack(">H", 391)
    )
    font = CFFFont(build_cff(encoding, ["alpha", "beta"]))
    assert font.builtin_encoding() == {0x41: "alpha", 0x42: "beta", 0x5A: "alpha"}
