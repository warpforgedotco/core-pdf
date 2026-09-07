from typing import Any

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_text import internal_PDFS, real_pymupdf

pytestmark = pytest.mark.compat_differential


def internal_selection_source(rotation: int = 0) -> bytes:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        page = fixture.new_page()
        page.insert_text((150, 200), "First words\nSecond line\nThird line", rotate=rotation)
        fixture.select([len(fixture) - 1])
        return fixture.tobytes()


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("fraction", [0, 0.01, 0.25, 0.5, 1])
def test_textbox_character_intersection(rotation: int, fraction: float) -> None:
    source = internal_selection_source(rotation)
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        ref_page, page = expected[0], actual[0]
        bbox = ref_page.get_text("words")[0][:4]
        rect = (bbox[0], bbox[1], bbox[0] + (bbox[2] - bbox[0]) * fraction, bbox[3])
        assert page.get_textbox(rect) == ref_page.get_textbox(rect)
        assert page.get_textpage().extractTextbox(rect) == ref_page.get_textpage().extractTextbox(
            rect
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ((0, 0), (600, 800)),
        ((160, 200), (175, 215)),
        ((175, 215), (160, 200)),
        ((152, 200), (152, 200)),
        ((150, 0), (150, 190)),
        ((0, 0), (0, 215)),
        ((0, 200), (600, 200)),
        ((600, 200), (0, 215)),
    ],
)
def test_selection_reading_order(start: tuple[int, int], end: tuple[int, int]) -> None:
    results = []
    source = internal_selection_source()
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(stream=source) as document:
            page = document[0]
            snapshot = page.get_textpage()
        results.append(snapshot.extractSelection(start, end))
    assert results[0] == results[1]


def test_textbox_snapshot_reuse() -> None:
    results: list[Any] = []
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(stream=internal_selection_source()) as document:
            page = document[0]
            snapshot = page.get_textpage(clip=(150, 180, 180, 210))
            results.append(page.get_textbox((0, 0, 600, 800), textpage=snapshot))
        results.append(snapshot.extractTextbox((0, 0, 600, 800)))
    assert results[:2] == results[2:]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize(
    "query", ["First", "WORDS", "st words", "words Second", "line", "missing", ""]
)
@pytest.mark.parametrize("quads", [False, True])
def test_search_character_geometry(rotation: int, query: str, quads: bool) -> None:
    from .test_pymupdf_geometry import internal_geometry_expected

    def coordinates(values: Any) -> Any:
        if values is None:
            return None
        return [tuple(tuple(p) for p in value) if quads else tuple(value) for value in values]

    results = []
    source = internal_selection_source(rotation)
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(stream=source) as document:
            page = document[0]
            results.append(coordinates(page.search_for(query, quads=quads)))
            snapshot = page.get_textpage()
        results.append(coordinates(snapshot.search(query, hit_max=1, quads=quads)))
    assert results[:2] == internal_geometry_expected(results[2:])


@pytest.mark.parametrize("query", ["abc", "abc abc", "  ", "abc  Ä", "ä", "Ä", "ss", "ABC\nabc"])
def test_search_repeated_matches_whitespace_and_non_ascii(query: str) -> None:
    from .test_pymupdf_geometry import internal_geometry_expected

    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        page = fixture.new_page()
        page.insert_text((50, 50), "abcabc ABC abc  Ää ä ß\nNext line")
        fixture.select([len(fixture) - 1])
        source = fixture.tobytes()
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(stream=source) as document:
            results.append([tuple(rect) for rect in document[0].search_for(query)])
    assert results[0] == internal_geometry_expected(results[1])


def test_search_reuses_snapshot_clip() -> None:
    from .test_pymupdf_geometry import internal_geometry_expected

    results = []
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(stream=internal_selection_source()) as document:
            page = document[0]
            snapshot = page.get_textpage(clip=(150, 180, 180, 210))
            results.append(
                [
                    tuple(rect)
                    for rect in page.search_for("First", textpage=snapshot, clip=(0, 0, 1, 1))
                ]
            )
    assert results[0] == internal_geometry_expected(results[1])


@pytest.mark.parametrize("method", ["get_textbox", "search_for"])
def test_selection_rejects_another_pages_snapshot(method: str) -> None:
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(internal_PDFS[0]) as first, module.open(internal_PDFS[0]) as second:
            page = first[0]
            snapshot = page.get_textpage()
            argument = (0, 0, 600, 800) if method == "get_textbox" else "text"
            with pytest.raises(ValueError) as error:
                getattr(second[0], method)(argument, textpage=snapshot)
            results.append(str(error.value))
    assert results[0] == results[1]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_selection_follows_rotated_lines(rotation: int) -> None:
    source = internal_selection_source(rotation)
    with real_pymupdf.open(stream=source) as reference:
        lines = [
            line for block in reference[0].get_text("rawdict")["blocks"] for line in block["lines"]
        ]
        points = []
        for line, offset in [(lines[0], 2), (lines[1], 4)]:
            char = line["spans"][0]["chars"][offset]
            x0, y0, x1, y1 = char["bbox"]
            dx, dy = line["dir"]
            points.append(((x0 + x1) / 2 - dx * 0.1, (y0 + y1) / 2 - dy * 0.1))
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(stream=source) as document:
            page = document[0]
            results.append(page.get_textpage().extractSelection(*points))
    assert results[0] == results[1]
