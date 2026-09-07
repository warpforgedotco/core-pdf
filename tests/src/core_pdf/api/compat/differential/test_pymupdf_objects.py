from pathlib import Path

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_text import internal_PDFS, real_pymupdf

pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("path", internal_PDFS, ids=lambda path: path.name)
@pytest.mark.parametrize("compressed", [False, True])
def test_native_object_inspection(path: Path, compressed: bool) -> None:
    with real_pymupdf.open(path) as reference, compat_pymupdf.open(path) as actual:
        assert actual.xref_length() == reference.xref_length()
        assert actual.pdf_catalog() == reference.pdf_catalog()
        assert actual.pdf_trailer(compressed=compressed) == reference.pdf_trailer(
            compressed=compressed
        )
        for xref in range(1, reference.xref_length()):
            assert actual.xref_object(xref, compressed=compressed) == reference.xref_object(
                xref, compressed=compressed
            ), xref
            assert actual.xref_get_keys(xref) == reference.xref_get_keys(xref)
            for key in reference.xref_get_keys(xref):
                assert actual.xref_get_key(xref, key) == reference.xref_get_key(xref, key), (
                    xref,
                    key,
                )
            for method in [
                "xref_is_stream",
                "xref_is_font",
                "xref_is_image",
                "xref_is_xobject",
                "xref_stream_raw",
                "xref_stream",
            ]:
                assert getattr(actual, method)(xref) == getattr(reference, method)(xref), (
                    xref,
                    method,
                )


@pytest.mark.parametrize(
    "object_text",
    [
        "42",
        "true",
        "-.123456789",
        "(literal\\ntext\\t\\(test\\))",
        "<FEFF004103A9>",
        "[1 2 /Name#20Here (abc) <00FF> false null]",
        "<< /Nested << /A 1 /B [3 4] >> /Float .123456789 /Zero -0.0 >>",
        "<< /A /Name#C3#A9 /B /Name#E9 /S <410042> /K#C3#A9 1 >>",
        "<< /Array [" + " ".join(str(i) for i in range(60)) + "] /Empty [] /Dict <<>> >>",
    ],
)
@pytest.mark.parametrize("compressed", [False, True])
def test_object_syntax_and_values(object_text: str, compressed: bool) -> None:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        xref = fixture.get_new_xref()
        fixture.update_object(xref, object_text)
        source = fixture.tobytes()
    with (
        real_pymupdf.open(stream=source) as reference,
        compat_pymupdf.open(stream=source) as actual,
    ):
        assert actual.xref_object(xref, compressed=compressed) == reference.xref_object(
            xref, compressed=compressed
        )
        assert actual.xref_get_keys(xref) == reference.xref_get_keys(xref)
        for key in [
            *reference.xref_get_keys(xref),
            "Nested/A",
            "Nested/B/0",
            "Missing",
            "/Nested",
            "Nested/Missing/X",
        ]:
            assert actual.xref_get_key(xref, key) == reference.xref_get_key(xref, key)


@pytest.mark.parametrize("xref", [-2, -1, 0, 1000])
@pytest.mark.parametrize(
    "method",
    [
        "xref_object",
        "xref_get_keys",
        "xref_get_key",
        "xref_stream",
        "xref_stream_raw",
        "xref_is_stream",
        "xref_is_font",
    ],
)
def test_invalid_and_trailer_references(xref: int, method: str) -> None:
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(internal_PDFS[0]) as document:
            try:
                result = (
                    getattr(document, method)(xref, "Root")
                    if method == "xref_get_key"
                    else getattr(document, method)(xref)
                )
                results.append(result)
            except (ValueError, RuntimeError) as error:
                results.append((type(error).__name__, str(error)))
    assert results[0] == results[1]


def test_empty_document_object_structure() -> None:
    with real_pymupdf.open() as reference, compat_pymupdf.open() as actual:
        assert actual.xref_length() == reference.xref_length()
        assert actual.pdf_catalog() == reference.pdf_catalog()
        assert actual.pdf_trailer() == reference.pdf_trailer()
        for xref in [1, 2]:
            assert actual.xref_object(xref) == reference.xref_object(xref)


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("xref_length", ()),
        ("pdf_catalog", ()),
        ("pdf_trailer", ()),
        ("xref_object", (1,)),
        ("xref_get_keys", (1,)),
        ("xref_get_key", (1, "Type")),
        ("xref_stream", (1,)),
        ("xref_stream_raw", (1,)),
        ("xref_is_stream", (1,)),
        ("xref_is_image", (1,)),
        ("xref_is_font", (1,)),
        ("xref_is_xobject", (1,)),
    ],
)
def test_object_access_after_document_close(method: str, args: tuple[object, ...]) -> None:
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        document = module.open(internal_PDFS[0])
        document.close()
        with pytest.raises(ValueError) as error:
            getattr(document, method)(*args)
        results.append(str(error.value))
    assert results[0] == results[1]


@pytest.mark.parametrize("path", internal_PDFS, ids=lambda path: path.name)
def test_object_access_after_text_capture(path: Path) -> None:
    with real_pymupdf.open(path) as reference, compat_pymupdf.open(path) as actual:
        reference[0].get_text("dict")
        actual[0].get_text("dict")
        for xref in range(1, reference.xref_length()):
            assert actual.xref_object(xref) == reference.xref_object(xref)


@pytest.mark.parametrize(
    "key", ["Root/Pages/Count", "Root/Pages/Kids/0", "Root/Pages/Missing", "Root"]
)
def test_indirect_dictionary_paths(key: str) -> None:
    with (
        real_pymupdf.open(internal_PDFS[0]) as reference,
        compat_pymupdf.open(internal_PDFS[0]) as actual,
    ):
        assert actual.xref_get_key(-1, key) == reference.xref_get_key(-1, key)


@pytest.mark.parametrize("compress", [False, True])
def test_form_stream_inspection(compress: bool) -> None:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        xref = fixture.get_new_xref()
        fixture.update_object(xref, "<< /Type /XObject /Subtype /Form /BBox [0 0 10 10] >>")
        fixture.update_stream(xref, b"q 0 0 10 10 re f Q\n", compress=compress)
        source = fixture.tobytes()
    with (
        real_pymupdf.open(stream=source) as reference,
        compat_pymupdf.open(stream=source) as actual,
    ):
        for method in [
            "xref_is_xobject",
            "xref_is_image",
            "xref_is_stream",
            "xref_object",
            "xref_stream",
            "xref_stream_raw",
        ]:
            assert getattr(actual, method)(xref) == getattr(reference, method)(xref)


@pytest.mark.parametrize("data", [b"Hello world\xe9", b"ab\x01cd", b"a\nb", b"a\bb", b"\\\\\\\\"])
@pytest.mark.parametrize("ascii_only", [False, True])
def test_ascii_object_string_policy(data: bytes, ascii_only: bool) -> None:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        xref = fixture.get_new_xref()
        fixture.update_object(xref, "<< /Data [<" + data.hex() + ">] >>")
        source = fixture.tobytes()
    with (
        real_pymupdf.open(stream=source) as reference,
        compat_pymupdf.open(stream=source) as actual,
    ):
        for compressed in [False, True]:
            assert actual.xref_object(
                xref, compressed=compressed, ascii=ascii_only
            ) == reference.xref_object(xref, compressed=compressed, ascii=ascii_only)
