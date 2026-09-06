# SPDX-License-Identifier: AGPL-3.0-only
"""ZapfDingbats names its glyphs a1 to a191, and the AGL does not cover them.

Without a table for these names every dingbat decodes to nothing, and the
shipped ZapfDingbats widths cannot be reached either, since those are keyed by
the character a code denotes.

The values are pinned here rather than derived, because the table was built
from a font that is not present on every platform.
"""

from __future__ import annotations

import pytest

from core_pdf.impl.spec.s_09_fonts.data.core14 import FONT_DATA
from core_pdf.impl.spec.s_09_fonts.data.zapf_dingbats import ZAPF_DINGBATS_GLYPHS
from core_pdf.impl.spec.s_09_fonts.glyphs import glyph_name_to_unicode


def test_table_covers_every_glyph_the_font_defines() -> None:
    assert len(ZAPF_DINGBATS_GLYPHS) == 201
    assert all(name.startswith("a") and name[1:].isdigit() for name in ZAPF_DINGBATS_GLYPHS)
    assert all(len(text) == 1 for text in ZAPF_DINGBATS_GLYPHS.values())


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a1", "✁"),
        ("a2", "✂"),
        ("a9", "✠"),
        ("a71", "●"),
        ("a79", "❖"),
        ("a191", "➾"),
    ],
)
def test_glyph_names_resolve(name: str, expected: str) -> None:
    assert glyph_name_to_unicode(name, zapf_dingbats=True) == expected


def test_adobe_ordering_is_not_unicode_ordering() -> None:
    # The sequence has gaps and reversals that a naive "a1 is U+2701, so aN is
    # U+2700+N" table would get wrong.
    assert glyph_name_to_unicode("a3", zapf_dingbats=True) == "✄"  # U+2704, skipping U+2703
    assert glyph_name_to_unicode("a202", zapf_dingbats=True) == "✃"  # U+2703 belongs to a202
    assert glyph_name_to_unicode("a4", zapf_dingbats=True) == "☎"  # outside the Dingbats block
    # U+2707..U+2709 run backwards through a119, a118, a117.
    assert glyph_name_to_unicode("a119", zapf_dingbats=True) == "✇"
    assert glyph_name_to_unicode("a118", zapf_dingbats=True) == "✈"
    assert glyph_name_to_unicode("a117", zapf_dingbats=True) == "✉"


def test_ornamental_brackets_use_the_dingbat_not_the_ascii_lookalike() -> None:
    # These glyphs are reachable from two codepoints; Adobe names the ornament.
    assert glyph_name_to_unicode("a89", zapf_dingbats=True) == "❨"
    assert glyph_name_to_unicode("a90", zapf_dingbats=True) == "❩"
    assert glyph_name_to_unicode("a91", zapf_dingbats=True) == "❬"
    assert glyph_name_to_unicode("a87", zapf_dingbats=True) == "❲"


def test_shipped_widths_are_reachable_from_decoded_text() -> None:
    widths = FONT_DATA["ZapfDingbats"]["widths"]
    assert len(widths) == 201
    # Keyed by the character, so a decoded dingbat finds its advance.
    assert set(widths) == set(ZAPF_DINGBATS_GLYPHS.values())
    assert widths["●"] == 791
