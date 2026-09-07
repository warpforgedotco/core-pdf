from pathlib import Path
from typing import Any

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .support import FIXTURES_ROOT
from .test_pymupdf_geometry import internal_geometry_expected

real_pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.compat_differential
internal_PDFS = (
    FIXTURES_ROOT / "PyMuPDF/tests/resources/small-table.pdf",
    FIXTURES_ROOT / "PyMuPDF/docs/samples/mupdf-title.pdf",
    FIXTURES_ROOT / "PyMuPDF/tests/resources/test_4043.pdf",
)


def internal_text_snapshot(module: Any, source: bytes, **kwargs: Any) -> dict[str, Any]:
    with module.open(stream=source, filetype="pdf") as document:
        page = document[0]
        return {kind: page.get_text(kind, **kwargs) for kind in ("text", "words", "blocks")}


@pytest.mark.parametrize("pdf_path", internal_PDFS, ids=lambda path: path.name)
def test_text_words_and_blocks_share_capture_grouping(pdf_path: Path) -> None:
    source = pdf_path.read_bytes()
    assert internal_text_snapshot(compat_pymupdf, source) == internal_geometry_expected(
        internal_text_snapshot(real_pymupdf, source)
    )


@pytest.mark.parametrize(
    "fontname",
    [
        "helv",
        "hebo",
        "heit",
        "hebi",
        "tiro",
        "tibo",
        "tiit",
        "tibi",
        "cour",
        "cobo",
        "coit",
        "cobi",
        "symb",
        "zadb",
    ],
)
def test_standard_font_text_metrics(fontname: str) -> None:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        page = fixture.new_page(width=612, height=792)
        page.insert_text(
            (50, 60), "First words, separated.\nSecond line", fontname=fontname, fontsize=11
        )
        page.insert_text((50, 120), "Last paragraph", fontname=fontname, fontsize=16)
        fixture.select([len(fixture) - 1])
        source = fixture.tobytes()
    assert internal_text_snapshot(compat_pymupdf, source) == internal_geometry_expected(
        internal_text_snapshot(real_pymupdf, source)
    )


@pytest.mark.parametrize(
    "clip",
    [
        (0, 0, 612, 792),
        (100, 190, 200, 216),
        (110, 190, 180, 215),
        (0, 0, 10, 10),
    ],
)
def test_character_clipping(clip: tuple[int, int, int, int]) -> None:
    source = internal_PDFS[0].read_bytes()
    assert internal_text_snapshot(compat_pymupdf, source, clip=clip) == internal_geometry_expected(
        internal_text_snapshot(real_pymupdf, source, clip=clip)
    )


def test_word_delimiters() -> None:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        page = fixture.new_page()
        page.insert_text((50, 50), "word1,word2 - word3. word4?word5.")
        fixture.select([len(fixture) - 1])
        source = fixture.tobytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        for delimiters in (None, "", ",.?-", ","):
            assert actual[0].get_text("words", delimiters=delimiters) == internal_geometry_expected(
                expected[0].get_text("words", delimiters=delimiters)
            )


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("flags", [0, 8, 64, 195])
def test_rotated_hidden_text_and_spacing_flags(rotation: int, flags: int) -> None:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        page = fixture.new_page(width=612, height=792)
        page.insert_text((300, 400), "Visible text", rotate=rotation)
        page.insert_text((300, 500), "Hidden text", rotate=rotation, render_mode=3)
        page.insert_text((50, 50), "One")
        page.insert_text((74, 50), "two")
        fixture.select([len(fixture) - 1])
        source = fixture.tobytes()
    assert internal_text_snapshot(
        compat_pymupdf, source, flags=flags
    ) == internal_geometry_expected(internal_text_snapshot(real_pymupdf, source, flags=flags))
