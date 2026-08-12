from pathlib import Path

from core_pdf.api.compat.pdfplumber import PDF, open, utils

FIXTURE = Path("tests/fixtures/SCORE-Bench/src/executive-summary-2022-p1-7-p007.pdf")


def test_pdfplumber_page_objects_and_coordinates() -> None:
    with open(FIXTURE) as pdf:
        page = pdf.pages[0]
        assert page.page_number == 1
        assert page.width > 0
        assert page.height > 0
        assert page.chars
        assert all(char["top"] <= char["bottom"] for char in page.chars)
        assert all(char["doctop"] >= 0 for char in page.chars)
        assert set(page.objects) >= {"char"}


def test_pdfplumber_views_words_search_and_selection() -> None:
    with PDF.open(FIXTURE, pages=(1,)) as pdf:
        page = pdf.pages[0]
        words = page.extract_words(return_chars=True)
        assert words
        assert words[0]["chars"]
        assert page.search("2022", regex=False)
        assert len(pdf.pages) == 1
        assert page.crop((0, 0, page.width / 2, page.height)).chars
        assert page.within_bbox((0, 0, page.width, page.height)).chars


def test_pdfplumber_edges_and_serialization() -> None:
    with open(FIXTURE) as pdf:
        page = pdf.pages[0]
        assert len(page.edges) >= len(page.lines)
        assert all(edge["orientation"] in {"h", "v"} for edge in page.edges)
        assert pdf.to_dict()["pages"][0]["page_number"] == 1


def test_pdfplumber_page_image_writes_png(tmp_path: Path) -> None:
    with open(FIXTURE) as pdf:
        target = tmp_path / "page.png"
        pdf.pages[0].to_image(resolution=20).save(target)
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_pdfplumber_table_debug_surface() -> None:
    with open(FIXTURE) as pdf:
        finder = pdf.pages[0].debug_tablefinder()
        assert finder.page.page_number == 1
        assert isinstance(finder.edges, list)
        assert isinstance(finder.intersections, dict)
        assert isinstance(finder.tables, list)


def test_pdfplumber_table_finder_honors_explicit_lines() -> None:
    with open(FIXTURE) as pdf:
        page = pdf.pages[0]
        finder = page.debug_tablefinder(
            {
                "vertical_strategy": "explicit",
                "horizontal_strategy": "explicit",
                "explicit_vertical_lines": [10, 20],
                "explicit_horizontal_lines": [30, 40],
            }
        )
        assert {edge["orientation"] for edge in finder.edges} == {"v", "h"}
        assert len(finder.edges) == 4


def test_pdfplumber_structure_tree_maps_native_structure() -> None:
    structure_fixture = Path("tests/fixtures/pdfplumber/tests/pdfs/pdf_structure.pdf")
    with open(structure_fixture) as pdf:
        tree = pdf.pages[0].structure_tree
    assert tree
    assert tree[0]["role"] == "Document"
    assert any(child["role"] == "Table" for child in tree[0]["children"])


def test_pdfplumber_lifecycle_layout_and_serialization() -> None:
    with open(FIXTURE) as pdf:
        page = pdf.pages[0]
        assert page.layout is not None
        assert page.extract_text_lines()
        assert pdf.to_json()
        assert "object_type" in pdf.to_csv()
        page.close()


def test_pdfplumber_text_line_and_word_options() -> None:
    with open(FIXTURE) as pdf:
        page = pdf.pages[0]
        lines = page.extract_text_lines(return_chars=True)
        words = page.extract_words(return_chars=True, split_at_punctuation="-")
    assert lines
    assert lines[0]["chars"]
    assert words
    assert words[0]["chars"]


def test_pdfplumber_utils_facade() -> None:
    with open(FIXTURE) as pdf:
        chars = pdf.pages[0].chars
        words = utils.extract_words(chars, return_chars=True)
    assert words
    assert utils.obj_to_bbox(chars[0]) == (
        chars[0]["x0"],
        chars[0]["top"],
        chars[0]["x1"],
        chars[0]["bottom"],
    )
    assert utils.text.extract_text(chars)


def test_pdfplumber_image_objects_expose_source_fields() -> None:
    with open(FIXTURE) as pdf:
        images = pdf.pages[0].images
    if images:
        assert images[0]["srcsize"]
        assert "stream" in images[0]
