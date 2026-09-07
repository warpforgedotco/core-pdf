from typing import Any

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_geometry import internal_geometry_expected
from .test_pymupdf_text import internal_ligature_source, internal_PDFS, real_pymupdf

pytestmark = pytest.mark.compat_differential


def internal_snapshot(textpage: Any) -> dict[str, Any]:
    return {
        "rect": tuple(textpage.rect),
        "text": textpage.extractText(),
        "text_alias": textpage.extractTEXT(),
        "words": textpage.extractWORDS(),
        "blocks": textpage.extractBLOCKS(),
    }


@pytest.mark.parametrize("flags", [0, 1, 2, 195])
@pytest.mark.parametrize("clip", [None, (60, 130, 90, 145)])
def test_textpage_freezes_flags_and_clip(flags: int, clip: object) -> None:
    source = internal_ligature_source()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        ref_page, page = expected[0], actual[0]
        ref_tp, tp = ref_page.get_textpage(clip, flags), page.get_textpage(clip, flags)
        assert internal_snapshot(tp) == internal_geometry_expected(internal_snapshot(ref_tp))
        for kind in ("text", "words", "blocks"):
            assert page.get_text(
                kind, textpage=tp, flags=1 - flags, clip=(0, 0, 1, 1)
            ) == internal_geometry_expected(
                ref_page.get_text(kind, textpage=ref_tp, flags=1 - flags, clip=(0, 0, 1, 1))
            )
        assert tp.extractWORDS(delimiters="fﬁ") == internal_geometry_expected(
            ref_tp.extractWORDS(delimiters="fﬁ")
        )


@pytest.mark.parametrize("close_document", [False, True])
def test_textpage_survives_source_release(close_document: bool) -> None:
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        document = module.open(internal_PDFS[0])
        page = document[0]
        textpage = page.get_textpage()
        if close_document:
            document.close()
        del page
        results.append(internal_snapshot(textpage))
        if not close_document:
            document.close()
    assert results[0] == internal_geometry_expected(results[1])


@pytest.mark.parametrize(
    "matrix",
    [
        (2, 2),
        (1, 0, 0, 1, 10, 20),
        (90,),
        (0, 1, -1, 0, 600, 0),
        (1, 0.25, 0.1, 1, 0, 0),
        (2, 1),
        (0, 0),
    ],
)
@pytest.mark.parametrize("clip", [None, (100, 190, 200, 216)])
def test_textpage_matrix_and_output_clipping(matrix: tuple[float, ...], clip: object) -> None:
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(internal_PDFS[0]) as document:
            page = document[0]
            results.append(internal_snapshot(page.get_textpage(clip, 0, module.Matrix(*matrix))))
    assert results[0] == internal_geometry_expected(results[1])


@pytest.mark.parametrize("same_document", [False, True])
def test_textpage_page_ownership(same_document: bool) -> None:
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(internal_PDFS[0]) as document, module.open(internal_PDFS[0]) as other:
            page = document[0]
            textpage = page.get_textpage()
            with pytest.raises(ValueError) as error:
                (document if same_document else other)[0].get_text(textpage=textpage)
            results.append((str(error.value), textpage.parent == page))
    assert results[0] == results[1]


def test_textpage_returns_independent_values() -> None:
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(internal_PDFS[0]) as document:
            page = document[0]
            textpage = page.get_textpage()
            box = textpage.rect
            box.x0 = 1000
            words = textpage.extractWORDS()
            words.clear()
            results.append(internal_snapshot(textpage))
    assert results[0] == internal_geometry_expected(results[1])


@pytest.mark.parametrize(
    "clip", [(110, 190, 180, 215), (100, 190, 180, 216), (106, 200, 140, 215), (120, 205, 121, 206)]
)
def test_reused_textpage_word_clip_requires_half_word_area(clip: tuple[int, ...]) -> None:
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(internal_PDFS[0]) as document:
            page = document[0]
            results.append(page.get_text("words", textpage=page.get_textpage(), clip=clip))
    assert results[0] == internal_geometry_expected(results[1])


@pytest.mark.parametrize("flags", [0, 64, 195])
def test_textpage_crop_bounds_and_empty_blocks(flags: int) -> None:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        page = fixture.new_page(width=200, height=200)
        page.insert_text((20, 20), "inside")
        page.insert_text((150, 150), "outside")
        page.set_cropbox(real_pymupdf.Rect(0, 0, 100, 100))
        fixture.select([len(fixture) - 1])
        source = fixture.tobytes()
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(stream=source) as document:
            page = document[0]
            results.append((internal_snapshot(page.get_textpage(flags=flags)), page.get_text()))
    assert results[0] == internal_geometry_expected(results[1])


def test_textpage_does_not_change_after_page_edit() -> None:
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(internal_PDFS[0]) as document:
            page = document[0]
            textpage = page.get_textpage()
            page.insert_text((50, 50), "Later text")
            results.append(internal_snapshot(textpage))
    assert results[0] == internal_geometry_expected(results[1])
