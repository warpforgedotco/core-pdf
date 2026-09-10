import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_geometry import internal_geometry_expected

real_pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize(
    "glyph_name", ["summationdisplay", "summationtext", "integraldisplay", "A"]
)
@pytest.mark.parametrize("to_unicode", [False, True])
@pytest.mark.parametrize("flags", [0, 128, 195])
def test_simple_glyph_names_use_reader_unicode_policy(
    glyph_name: str, to_unicode: bool, flags: int
) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((100, 100), "X")
        font = page.get_fonts()[0][0]
        document.xref_set_key(font, "Encoding", f"<< /Differences [88 /{glyph_name}] >>")
        document.xref_set_key(font, "FirstChar", "88")
        document.xref_set_key(font, "LastChar", "88")
        document.xref_set_key(font, "Widths", "[600]")
        if to_unicode:
            cmap = document.get_new_xref()
            document.update_object(cmap, "<< >>")
            document.update_stream(
                cmap,
                b"begincmap 1 begincodespacerange <00> <FF> endcodespacerange "
                b"1 beginbfchar <58> <0059> endbfchar endcmap",
            )
            document.xref_set_key(font, "ToUnicode", f"{cmap} 0 R")
        source = document.tobytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        for kind in ("text", "words", "blocks", "rawdict"):
            assert actual[0].get_text(kind, flags=flags) == internal_geometry_expected(
                expected[0].get_text(kind, flags=flags)
            )


@pytest.mark.parametrize(
    "replacement", ["Replacement", "OK", "Native suffix", "Prefix Native", "One two three", ""]
)
@pytest.mark.parametrize(
    "layout", ["single", "consecutive", "nested", "repeated", "shared", "inline"]
)
def test_structure_actual_text_preserves_marked_content_scopes(
    replacement: str, layout: str
) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((100, 100), "Native")
        font_name = page.get_fonts()[0][4]
        tree, element = document.get_new_xref(), document.get_new_xref()
        document.update_object(
            tree,
            f"<< /Type /StructTreeRoot /K [{element} 0 R] "
            f"/ParentTree << /Nums [0 [{element} 0 R {element} 0 R]] >> >>",
        )
        document.update_object(
            element,
            f"<< /Type /StructElem /S /Span /P {tree} 0 R /Pg {page.xref} 0 R "
            f"/K [0 1] /ActualText {real_pymupdf.get_pdf_str(replacement)} >>",
        )
        document.xref_set_key(document.pdf_catalog(), "StructTreeRoot", f"{tree} 0 R")
        document.xref_set_key(page.xref, "StructParents", "0")
        body = b"(Native) Tj"
        if layout == "consecutive":
            body = b"(Na) Tj (tive) Tj"
        elif layout == "nested":
            body = b"(Na) Tj /Span BMC (ti) Tj EMC (ve) Tj"
        inline = "/ActualText (Native) " if layout == "inline" else ""
        content = (
            f"BT /{font_name} 12 Tf 100 700 Td /Span << /MCID 0 {inline}>> BDC ".encode()
            + body
            + b" EMC "
        )
        if layout in ("repeated", "shared"):
            mcid = 0 if layout == "repeated" else 1
            content += f"0 -30 Td /Span << /MCID {mcid} >> BDC (Native) Tj EMC ".encode()
        content += b"ET"
        document.update_stream(page.get_contents()[0], content)
        source = document.tobytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        for kind in ("text", "words", "blocks", "rawdict"):
            assert actual[0].get_text(kind) == internal_geometry_expected(
                expected[0].get_text(kind)
            )
