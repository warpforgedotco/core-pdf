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
