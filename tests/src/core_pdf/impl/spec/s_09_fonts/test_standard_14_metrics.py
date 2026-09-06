"""The prescribed standard 14 metrics use literal encoded glyph characters."""

from __future__ import annotations

import pytest

from core_pdf.impl.spec.s_09_fonts.data.base_encodings import STANDARD_ENCODING
from core_pdf.impl.spec.s_09_fonts.data.core14 import FONT_DATA
from core_pdf.impl.spec.s_09_fonts.metrics import standard_14_widths


@pytest.mark.parametrize(
    ("font", "character", "width"),
    [
        ("Helvetica", "H", 722),
        ("Times-Roman", "A", 722),
        ("Courier", "A", 600),
        ("Times-Italic", " ", 250),
    ],
)
def test_standard_14_glyph_metrics(font: str, character: str, width: int) -> None:
    widths = standard_14_widths(font, STANDARD_ENCODING)
    assert widths is not None
    assert widths.width_for(ord(character), -1) == width


def test_standard_14_table_contains_only_prescribed_font_names() -> None:
    assert len(FONT_DATA) == 14
    assert len({id(entry) for entry in FONT_DATA.values()}) == 14
    assert len({id(entry["widths"]) for entry in FONT_DATA.values()}) == 9
    for name, entry in FONT_DATA.items():
        assert entry["widths"], name
        assert entry["props"]["FontName"] == name
        assert all(isinstance(width, int) for width in entry["widths"].values())


@pytest.mark.parametrize("name", ["Arial", "CourierNew", "TimesNewRoman"])
def test_nonstandard_font_names_do_not_select_substitute_metrics(name: str) -> None:
    assert standard_14_widths(name, STANDARD_ENCODING) is None
