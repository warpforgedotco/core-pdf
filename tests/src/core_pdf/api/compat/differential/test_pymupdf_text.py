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


def internal_ligature_source() -> bytes:
    source = FIXTURES_ROOT / "PyMuPDF/tests/resources/2201.00069.pdf"
    with real_pymupdf.open(source) as fixture:
        fixture.select([0])
        page = fixture[0]
        content = page.get_contents()[0]
        page.set_contents(content)
        fixture.update_stream(content, b"BT /F68 12 Tf 50 700 Td <411b421c431d441e451f46> Tj ET")
        return fixture.tobytes()


@pytest.mark.parametrize("flags", [0, 1, 2, 3, 194, 195])
@pytest.mark.parametrize("clip", [None, (60, 130, 90, 145)])
def test_ligature_preservation_and_expansion(flags: int, clip: object) -> None:
    source = internal_ligature_source()
    assert internal_text_snapshot(
        compat_pymupdf, source, flags=flags, clip=clip
    ) == internal_geometry_expected(
        internal_text_snapshot(real_pymupdf, source, flags=flags, clip=clip)
    )


@pytest.mark.parametrize("flags", [0, 1])
def test_ligature_word_delimiter_geometry(flags: int) -> None:
    source = internal_ligature_source()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        for delimiters in ("f", "i", "ﬁ", "BDE"):
            assert actual[0].get_text(
                "words", flags=flags, delimiters=delimiters
            ) == internal_geometry_expected(
                expected[0].get_text("words", flags=flags, delimiters=delimiters)
            )


def internal_unicode_source(codepoints: list[int]) -> bytes:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        page = fixture.new_page(width=2000)
        page.insert_text((50, 50), " ".join(chr(65 + i) for i in range(len(codepoints))))
        font = page.get_fonts()[0][0]
        cmap = fixture.get_new_xref()
        fixture.update_object(cmap, "<<>>")
        pairs = " ".join(f"<{65 + i:02x}> <{point:04x}>" for i, point in enumerate(codepoints))
        fixture.update_stream(
            cmap,
            (
                "/CIDInit /ProcSet findresource begin 12 dict begin begincmap "
                "1 begincodespacerange <00> <ff> endcodespacerange "
                f"{len(codepoints)} beginbfchar {pairs} endbfchar "
                "endcmap CMapName currentdict /CMap defineresource pop end end"
            ).encode(),
        )
        fixture.xref_set_key(font, "ToUnicode", f"{cmap} 0 R")
        fixture.select([len(fixture) - 1])
        return fixture.tobytes()


@pytest.mark.parametrize("flags", [0, 2, 195])
def test_unicode_whitespace_flags_and_word_boundaries(flags: int) -> None:
    source = internal_unicode_source(
        [
            8,
            9,
            10,
            11,
            12,
            13,
            32,
            0x85,
            0xA0,
            0x1680,
            *range(0x2000, 0x200B),
            0x2028,
            0x2029,
            0x202F,
            0x205F,
            0x3000,
        ]
    )
    assert internal_text_snapshot(
        compat_pymupdf, source, flags=flags
    ) == internal_geometry_expected(internal_text_snapshot(real_pymupdf, source, flags=flags))


@pytest.mark.parametrize("flags", [0, 195])
def test_invalid_control_mappings_fall_back_to_font_encoding(flags: int) -> None:
    source = internal_unicode_source([*range(8), *range(14, 32), *range(127, 160)])
    assert internal_text_snapshot(
        compat_pymupdf, source, flags=flags
    ) == internal_geometry_expected(internal_text_snapshot(real_pymupdf, source, flags=flags))


@pytest.mark.parametrize("name", ["test_2791_content.pdf", "test_3376.pdf", "001003ED.pdf"])
def test_unresolved_control_cids_are_preserved(name: str) -> None:
    source = (FIXTURES_ROOT / "PyMuPDF/tests/resources" / name).read_bytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        assert actual[0].get_text() == expected[0].get_text()


@pytest.mark.parametrize("character", ["A", " "])
@pytest.mark.parametrize("offset", [0, 0.5, 1.19, 1.21])
@pytest.mark.parametrize("rotation", [0, 90])
def test_overlapping_repeated_characters_are_suppressed(
    character: str, offset: float, rotation: int
) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page()
        for displacement in (0, offset):
            page.insert_text((100 + displacement, 200), character, fontsize=12, rotate=rotation)
        source = document.tobytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        for kind in ("text", "words", "blocks", "rawdict"):
            assert actual[0].get_text(kind) == internal_geometry_expected(
                expected[0].get_text(kind)
            )
