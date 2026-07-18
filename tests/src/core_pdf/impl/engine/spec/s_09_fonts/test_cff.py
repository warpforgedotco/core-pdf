from __future__ import annotations

import pytest
from core_font_programs.impl.cff import STANDARD_GLYPH_SIDS, CFFFont
from fontTools.cffLib import cffStandardStrings


def authoritative_standard_strings() -> list[str]:
    return list(cffStandardStrings)


def test_standard_glyph_sids_match_authoritative_cff_mapping() -> None:
    expected = {name: sid for sid, name in enumerate(authoritative_standard_strings())}

    assert expected == STANDARD_GLYPH_SIDS


@pytest.mark.parametrize(
    ("name", "sid"),
    [("sterling", 98), ("fi", 109), ("fl", 110), ("Semibold", 390)],
)
def test_standard_glyph_names_resolve_to_their_charset_glyph(name: str, sid: int) -> None:
    font = object.__new__(CFFFont)
    font.custom_string_sids = {}
    font.cid_to_gid = {sid: 7}

    assert font.glyph_id_for_name(name) == 7
