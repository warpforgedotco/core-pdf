# SPDX-License-Identifier: AGPL-3.0-only
"""ISO 32000-1 9.3.3: word spacing applies only to the single-byte code 32."""

from __future__ import annotations

from math import isclose

from core_pdf.impl.spec.s_09_fonts.decoder import FontDecoder


def test_multibyte_cid_code_does_not_apply_word_spacing() -> None:
    decoder = FontDecoder(
        {
            "Subtype": "Type0",
            "Encoding": "Identity-H",
            "DescendantFonts": [{"Subtype": "CIDFontType0", "DW": 500}],
        }
    )

    advance = decoder.text_advance_vector(
        b"\x00 \x00A", font_size=10.0, char_space=0.0, word_space=2.0, horizontal_scale=1.0
    )

    assert isclose(advance[0], 0.1)
    assert advance[1] == 0.0
