from __future__ import annotations

import pytest

import core_pdf.impl.engine.spec.s_09_fonts.cff as cff_module
from core_pdf._vendor.fontTools.cffLib import cffStandardStrings
from core_pdf.impl.engine.spec.s_09_fonts.cff import STANDARD_GLYPH_SIDS, CFFFont


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
    font = CFFFont(None)
    font.cid_to_gid = {sid: 7}

    assert font.glyph_id_for_name(name) == 7


def test_cff_geometry_is_reused_across_bbox_feature_and_bitmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    charstring = bytes([239, 239, 21, 239, 139, 5, 139, 239, 5, 39, 139, 5, 14])
    font = CFFFont(None)
    font.charstrings = [charstring]
    original = cff_module.internal_type2_glyph_geometry
    calls = 0

    def counting_geometry(
        value: bytes,
        *,
        local_subrs: tuple[bytes, ...],
        global_subrs: tuple[bytes, ...],
        collect_contours: bool,
    ) -> tuple[list[list[tuple[float, float]]], tuple[float, float, float, float] | None]:
        nonlocal calls
        calls += 1
        return original(
            value,
            local_subrs=local_subrs,
            global_subrs=global_subrs,
            collect_contours=collect_contours,
        )

    monkeypatch.setattr(cff_module, "internal_type2_glyph_geometry", counting_geometry)

    assert font.glyph_bbox_for_gid(0) == (100.0, 100.0, 200.0, 200.0)
    assert font.glyph_feature(0).contours == 1
    assert font.glyph_bitmap_for_gid(0, width=4, height=4)
    assert calls == 1
