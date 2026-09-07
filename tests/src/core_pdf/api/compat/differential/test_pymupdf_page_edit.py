from pathlib import Path

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_geometry import internal_geometry_expected
from .test_pymupdf_text import internal_PDFS, real_pymupdf

pytestmark = pytest.mark.compat_differential


def internal_page_state(page: object) -> object:
    return {
        name: tuple(getattr(page, name))
        for name in (
            "mediabox",
            "cropbox",
            "bleedbox",
            "trimbox",
            "artbox",
            "rect",
            "transformation_matrix",
            "rotation_matrix",
            "derotation_matrix",
        )
    }


@pytest.mark.parametrize("path", internal_PDFS, ids=lambda p: p.name)
@pytest.mark.parametrize("rotation", [-450, -90, -1, 0, 45, 90, 180, 270, 360, 450])
def test_page_rotation_persistence(path: Path, rotation: int) -> None:
    with real_pymupdf.open(path) as reference, compat_pymupdf.open(path) as actual:
        actual_page = actual[0]
        reference_page = reference[0]
        retained_page = actual[0]
        for page in (reference_page, actual_page):
            snapshot = page.get_textpage()
            old_text = snapshot.extractText()
            page.set_rotation(rotation)
            assert snapshot.extractText() == old_text
        assert actual_page.rotation == reference_page.rotation
        assert retained_page.rotation == reference_page.rotation
        assert internal_page_state(retained_page) == internal_page_state(actual_page)
        assert internal_page_state(actual_page) == internal_geometry_expected(
            internal_page_state(reference_page)
        )
        assert actual_page.get_text("rawdict") == internal_geometry_expected(
            reference_page.get_text("rawdict")
        )
        assert actual.xref_object(actual_page.xref) == reference.xref_object(reference_page.xref)
        with real_pymupdf.open(stream=actual.tobytes(no_new_id=True)) as reopened:
            assert internal_page_state(reopened[0]) == internal_page_state(reference_page)
            assert reopened[0].get_text("rawdict") == reference_page.get_text("rawdict")


@pytest.mark.parametrize("path", internal_PDFS, ids=lambda p: p.name)
@pytest.mark.parametrize("name", ["cropbox", "bleedbox", "trimbox", "artbox"])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_page_box_persistence(path: Path, name: str, rotation: int) -> None:
    with real_pymupdf.open(path) as reference, compat_pymupdf.open(path) as actual:
        pages = [reference[0], actual[0]]
        for page in pages:
            page.set_rotation(rotation)
            page.set_mediabox((10, 20, 410, 620))
            getattr(page, f"set_{name}")((20.25, 30.5, 390.75, 580.125))
        assert internal_page_state(pages[1]) == internal_geometry_expected(
            internal_page_state(pages[0])
        )
        assert actual.xref_object(pages[1].xref) == reference.xref_object(pages[0].xref)
        with real_pymupdf.open(stream=actual.tobytes(no_new_id=True)) as reopened:
            assert internal_page_state(reopened[0]) == internal_page_state(pages[0])
            assert reopened[0].get_text("rawdict") == pages[0].get_text("rawdict")
        for page in pages:
            page.set_mediabox((0, 0, 400, 600))
        assert actual.xref_object(pages[1].xref) == reference.xref_object(pages[0].xref)
        assert internal_page_state(pages[1]) == internal_geometry_expected(
            internal_page_state(pages[0])
        )


@pytest.mark.parametrize(
    ("method", "value"),
    [
        ("set_rotation", 90.0),
        ("set_mediabox", (0, 0, 0, 0)),
        ("set_mediabox", (100, 100, 0, 0)),
        ("set_cropbox", (-1, 0, 10, 20)),
        ("set_cropbox", (0, 0, 0, 0)),
        ("set_bleedbox", (-1, 0, 10, 20)),
        ("set_trimbox", (0, 0, 100000, 100000)),
        ("set_artbox", (100, 100, 0, 0)),
    ],
)
def test_page_box_errors(method: str, value: object) -> None:
    results = []
    for engine in (real_pymupdf, compat_pymupdf):
        with engine.open(internal_PDFS[0]) as document:
            with pytest.raises(Exception) as error:
                getattr(document[0], method)(value)
            results.append((type(error.value).__name__, str(error.value)))
    assert results[0] == results[1]


@pytest.mark.parametrize(
    "method",
    ["set_rotation", "set_mediabox", "set_cropbox", "set_bleedbox", "set_trimbox", "set_artbox"],
)
def test_page_edit_closed_document(method: str) -> None:
    results = []
    for engine in (real_pymupdf, compat_pymupdf):
        document = engine.open(internal_PDFS[0])
        page = document[0]
        document.close()
        with pytest.raises(Exception) as error:
            getattr(page, method)(90 if method == "set_rotation" else (0, 0, 100, 100))
        results.append((type(error.value).__name__, str(error.value)))
    assert results[0] == results[1]


def test_page_edit_inherited_boxes() -> None:
    with real_pymupdf.open() as source:
        page = source.new_page()
        parent = int(source.xref_get_key(page.xref, "Parent")[1].split()[0])
        source.update_object(page.xref, f"<< /Type /Page /Parent {parent} 0 R >>")
        source.xref_set_key(parent, "MediaBox", "[10 20 410 620]")
        source.xref_set_key(parent, "CropBox", "[20 40 390 590]")
        source.xref_set_key(parent, "Rotate", "90")
        data = source.tobytes(no_new_id=True)
    with real_pymupdf.open(stream=data) as reference, compat_pymupdf.open(stream=data) as actual:
        pages = (reference[0], actual[0])
        for method, value in [
            ("set_mediabox", (0, 0, 500, 700)),
            ("set_rotation", 180),
            ("set_rotation", 22.5),
            ("set_cropbox", (20, 30, 400, 600)),
        ]:
            for page in pages:
                getattr(page, method)(value)
            assert internal_page_state(pages[1]) == internal_geometry_expected(
                internal_page_state(pages[0])
            )
            assert actual.xref_object(pages[1].xref) == reference.xref_object(pages[0].xref)
            with real_pymupdf.open(stream=actual.tobytes(no_new_id=True)) as reopened:
                assert internal_page_state(reopened[0]) == internal_page_state(pages[0])
