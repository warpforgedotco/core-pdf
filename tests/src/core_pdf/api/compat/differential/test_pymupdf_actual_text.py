import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .support import FIXTURES_ROOT
from .test_pymupdf_geometry import internal_geometry_expected

real_pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("flags", [0, 4, 195, 199])
@pytest.mark.parametrize("layout", ["image", "inline", "multiple", "nested", "mixed", "text"])
@pytest.mark.parametrize("replacement", ["Replacement", "OK", ""])
def test_actual_text_scopes_with_images(flags: int, layout: str, replacement: str) -> None:
    with real_pymupdf.open(FIXTURES_ROOT / "pdf20examples/pdf20-utf8-test.pdf") as document:
        image = b"q 20 0 0 30 100 400 cm /Im0 Do Q "
        body = image
        if layout == "inline":
            body = b"q 20 0 0 30 100 400 cm BI /W 1 /H 1 /BPC 8 /CS /RGB ID \xff\x00\x00 EI Q "
        elif layout == "multiple":
            body += b"q 10 0 0 20 120 300 cm /Im0 Do Q "
        elif layout == "nested":
            body = b"/Span << /MCID 0 >> BDC " + image + b"EMC "
        elif layout in ("mixed", "text"):
            text = b"BT /T1 12 Tf 100 380 Td (Native) Tj ET "
            body = body + text if layout == "mixed" else text
        value = real_pymupdf.get_pdf_str(replacement).encode()
        content = b"/Span << /ActualText " + value + b" >> BDC " + body + b"EMC"
        page = document[0]
        stream = page.get_contents()[0]
        page.set_contents(stream)
        document.update_stream(stream, content)
        document.xref_set_key(document.pdf_catalog(), "StructTreeRoot", "null")
        source = document.tobytes()
    with (
        real_pymupdf.open(stream=source) as expected,
        compat_pymupdf.open(stream=source) as actual,
    ):
        for kind in ("text", "words", "blocks", "rawdict"):
            assert actual[0].get_text(kind, flags=flags) == internal_geometry_expected(
                expected[0].get_text(kind, flags=flags)
            )


@pytest.mark.parametrize("structure_parents", ["null", "/Invalid", "0"])
@pytest.mark.parametrize("parent_tree", ["null", "/Invalid"])
def test_image_mcid_without_usable_structure_tree(structure_parents: str, parent_tree: str) -> None:
    with real_pymupdf.open(FIXTURES_ROOT / "pdf20examples/pdf20-utf8-test.pdf") as document:
        page = document[0]
        stream = page.get_contents()[0]
        page.set_contents(stream)
        document.update_stream(
            stream,
            b"/Span << /MCID 0 >> BDC q 20 0 0 30 100 400 cm /Im0 Do Q EMC "
            b"BT /T1 12 Tf 100 380 Td (Visible text) Tj ET",
        )
        document.xref_set_key(page.xref, "StructParents", structure_parents)
        document.xref_set_key(
            document.pdf_catalog(),
            "StructTreeRoot",
            f"<< /Type /StructTreeRoot /ParentTree {parent_tree} >>",
        )
        source = document.tobytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        ref_page, page = expected[0], actual[0]
        for kind in ("text", "words", "blocks", "rawdict"):
            assert page.get_text(kind) == internal_geometry_expected(ref_page.get_text(kind))
