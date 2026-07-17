from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core_pdf.impl.third_party.cff import STANDARD_GLYPH_SIDS, CFFFont

REPO_ROOT = Path(__file__).parents[5]
CFFLIB_PATH = REPO_ROOT / "src/core_pdf/impl/third_party/_vendor/fontTools/cffLib/__init__.py"


def authoritative_standard_strings() -> list[str]:
    module = ast.parse(CFFLIB_PATH.read_text())
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "cffStandardStrings"
            for target in node.targets
        )
    )
    return ast.literal_eval(assignment.value)


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
