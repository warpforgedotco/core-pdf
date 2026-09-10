import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_geometry import internal_geometry_expected
from .test_pymupdf_text import internal_PDFS, real_pymupdf

pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("char_space", [-4, -1, 1.5, 3.2, 3.4, 4])
@pytest.mark.parametrize("word_space", [0, 3])
def test_spacing_changes_cursor_without_expanding_glyph_boxes(
    rotation: int, char_space: float, word_space: float
) -> None:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        page = fixture.new_page()
        page.insert_text((150, 200), "Char space word", rotate=rotation)
        for xref in page.get_contents():
            content = fixture.xref_stream(xref)
            content = content.replace(b"BT", f"BT\n{char_space} Tc {word_space} Tw".encode())
            fixture.update_stream(xref, content)
        fixture.select([len(fixture) - 1])
        source = fixture.tobytes()
    with (
        real_pymupdf.open(stream=source) as reference,
        compat_pymupdf.open(stream=source) as actual,
    ):
        ref_page, page = reference[0], actual[0]
        for kind in ["rawdict", "words", "text"]:
            assert page.get_text(kind) == internal_geometry_expected(ref_page.get_text(kind))
        assert [tuple(r) for r in page.search_for("Char")] == internal_geometry_expected(
            [tuple(r) for r in ref_page.search_for("Char")]
        )
        assert [
            tuple(tuple(p) for p in q) for q in page.search_for("Char", quads=True)
        ] == internal_geometry_expected(
            [tuple(tuple(p) for p in q) for q in ref_page.search_for("Char", quads=True)]
        )


@pytest.mark.parametrize("query", ["Chapter", "One", "1.8.2", "Omission", "This"])
def test_embedded_font_search_with_spacing_and_crop_translation(query: str) -> None:
    from .support import FIXTURES_ROOT

    path = FIXTURES_ROOT / "PyMuPDF/tests/resources/test-E+A.pdf"
    with real_pymupdf.open(path) as reference, compat_pymupdf.open(path) as actual:
        assert [tuple(r) for r in actual[0].search_for(query)] == internal_geometry_expected(
            [tuple(r) for r in reference[0].search_for(query)]
        )


@pytest.mark.parametrize("reset_matrix", [False, True])
def test_relative_line_origins_and_explicit_matrix_resets(reset_matrix: bool) -> None:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        page = fixture.new_page()
        page.insert_text((50, 50), "Relative")
        xref = page.get_contents()[0]
        content = ["BT /helv 11 Tf 1 0 0 1 50.1234 780.1234 Tm"]
        y = 780.1234
        for line in range(35):
            if line:
                y -= 12.3456
                if reset_matrix and line % 7 == 0:
                    content.append(f"1 0 0 1 50.1234 {y:.4f} Tm")
                else:
                    content.append("0 -12.3456 Td")
            content.append("(Relative) Tj")
        content.append("ET")
        fixture.update_stream(xref, "\n".join(content).encode())
        fixture.select([len(fixture) - 1])
        source = fixture.tobytes()
    with (
        real_pymupdf.open(stream=source) as reference,
        compat_pymupdf.open(stream=source) as actual,
    ):
        assert actual[0].get_text("rawdict") == internal_geometry_expected(
            reference[0].get_text("rawdict")
        )


@pytest.mark.parametrize("first_text", ["Paragraph start", "* List item", "1. Numbered item"])
@pytest.mark.parametrize("indent", [0.0, 0.5, 0.6, 12.0])
@pytest.mark.parametrize("line_distance", [8.0, 11.0, 15.0, 16.0])
def test_indented_lines_start_paragraphs_without_splitting_lists(
    first_text: str, indent: float, line_distance: float
) -> None:
    with real_pymupdf.open() as fixture:
        page = fixture.new_page()
        page.insert_text((50, 50), first_text, fontsize=10)
        page.insert_text((50 + indent, 50 + line_distance), "Following text", fontsize=10)
        source = fixture.tobytes()
    with (
        real_pymupdf.open(stream=source) as reference,
        compat_pymupdf.open(stream=source) as actual,
    ):
        for kind in ("words", "blocks", "rawdict"):
            assert actual[0].get_text(kind) == internal_geometry_expected(
                reference[0].get_text(kind)
            )


@pytest.mark.parametrize("gap", [2, 4, 6])
@pytest.mark.parametrize("following_font", ["helv", "cour"])
def test_synthetic_spaces_take_the_following_text_style(gap: int, following_font: str) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((50, 50), "Red", fontsize=12, color=(1, 0, 0))
        x = 50 + real_pymupdf.get_text_length("Red", fontsize=12) + gap
        page.insert_text((x, 50), "Blue", fontname=following_font, fontsize=12, color=(0, 0, 1))
        source = document.tobytes()
    with (
        real_pymupdf.open(stream=source) as expected,
        compat_pymupdf.open(stream=source) as actual,
    ):
        assert actual[0].get_text("rawdict") == internal_geometry_expected(
            expected[0].get_text("rawdict")
        )


@pytest.mark.parametrize(
    "preceding",
    ["A", "\u06f9", "\u0700", "\u1fff", "\u2000", "\u20cf", "\u20d0", "⌉", "∑", "漢", "\u00a0"],
)
@pytest.mark.parametrize("flags", [128, 130, 138])
def test_synthetic_spaces_respect_preceding_unicode_ranges(preceding: str, flags: int) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((50, 50), "Xtimes.", fontsize=10)
        font = page.get_fonts()[0][0]
        cmap = document.get_new_xref()
        document.update_object(cmap, "<< >>")
        document.update_stream(
            cmap,
            b"begincmap 1 begincodespacerange <00> <FF> endcodespacerange "
            + f"1 beginbfchar <58> <{ord(preceding):04X}> endbfchar endcmap".encode(),
        )
        document.xref_set_key(font, "ToUnicode", f"{cmap} 0 R")
        contents = page.get_contents()[0]
        document.update_stream(contents, b"BT /helv 10 Tf 50 750 Td [(X) -400 (times.)] TJ ET")
        source = document.tobytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        for kind in ("text", "words", "blocks", "rawdict"):
            assert actual[0].get_text(kind, flags=flags) == internal_geometry_expected(
                expected[0].get_text(kind, flags=flags)
            )
