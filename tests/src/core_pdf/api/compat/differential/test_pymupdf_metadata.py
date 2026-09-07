from pathlib import Path

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_text import internal_PDFS, real_pymupdf

pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("path", [None, *internal_PDFS], ids=lambda p: p.name if p else "empty")
@pytest.mark.parametrize(
    "metadata",
    [
        {"title": "New title", "author": "Author"},
        {"title": "Résumé € Ελληνικά 😀", "subject": "(parentheses) \\ slash"},
        {"title": "\x00\x01\b\t\n\f\r\x80\x81\x9f"},
        {"title": None, "author": "null", "subject": "none"},
        {"creationDate": "D:20260101000000Z", "trapped": "True"},
        {"format": "ignored", "encryption": "ignored"},
        {},
        None,
    ],
)
def test_metadata_update_and_persistence(
    path: Path | None, metadata: dict[str, object] | None
) -> None:
    with real_pymupdf.open(path) as reference, compat_pymupdf.open(path) as actual:
        for document in (reference, actual):
            if path is None:
                document.new_page()
            page = document[0]
            before = page.get_text()
            document.set_metadata({"title": "Previous", "keywords": "keep me"})
            previous_metadata = document.metadata
            previous_values = dict(previous_metadata)
            document.set_metadata(metadata)
            assert previous_metadata == previous_values
            assert previous_metadata is not document.metadata
            assert page.get_text() == before
        assert actual.metadata == reference.metadata
        assert actual.xref_length() == reference.xref_length()
        assert actual.xref_get_key(-1, "Info") == reference.xref_get_key(-1, "Info")
        kind, value = reference.xref_get_key(-1, "Info")
        if kind == "xref":
            xref = int(value.split()[0])
            assert actual.xref_object(xref) == reference.xref_object(xref)
        for engine in (real_pymupdf, compat_pymupdf):
            with engine.open(stream=actual.tobytes(no_new_id=True)) as reopened:
                assert reopened.metadata == reference.metadata


@pytest.mark.parametrize("metadata", [{"unknown": "value"}, [], "title", {"title": 123}])
def test_metadata_errors(metadata: object) -> None:
    results = []
    for engine in (real_pymupdf, compat_pymupdf):
        with engine.open(internal_PDFS[0]) as document:
            with pytest.raises(Exception) as error:
                document.set_metadata(metadata)
            results.append((type(error.value).__name__, str(error.value)))
    assert results[0] == results[1]


def test_metadata_closed_document() -> None:
    results = []
    for engine in (real_pymupdf, compat_pymupdf):
        document = engine.open(internal_PDFS[0])
        document.close()
        with pytest.raises(Exception) as error:
            document.set_metadata({"title": "closed"})
        results.append((type(error.value).__name__, str(error.value)))
    assert results[0] == results[1]


@pytest.mark.parametrize("path", [None, *internal_PDFS], ids=lambda p: p.name if p else "empty")
@pytest.mark.parametrize("xml", ["<x>Metadata α 😀</x>", "", "<x>" + "a" * 500 + "</x>"])
def test_xml_metadata_persistence(path: Path | None, xml: str) -> None:
    with real_pymupdf.open(path) as reference, compat_pymupdf.open(path) as actual:
        for document in (reference, actual):
            if path is None:
                document.new_page()
        for value in ("<previous/>", xml):
            for document in (reference, actual):
                document.set_xml_metadata(value)
            assert actual.get_xml_metadata() == reference.get_xml_metadata()
            assert actual.xref_xml_metadata() == reference.xref_xml_metadata()
            xref = reference.xref_xml_metadata()
            assert actual.xref_object(xref) == reference.xref_object(xref)
            assert actual.xref_stream_raw(xref) == reference.xref_stream_raw(xref)
            for engine in (real_pymupdf, compat_pymupdf):
                with engine.open(stream=actual.tobytes(no_new_id=True)) as reopened:
                    assert reopened.get_xml_metadata() == reference.get_xml_metadata()
        for payload in (b"a\x00b", b"\xffa", b"\xef\xbb\xbf<x/>"):
            for document in (reference, actual):
                document.update_stream(document.xref_xml_metadata(), payload)
            assert actual.get_xml_metadata() == reference.get_xml_metadata()
        for document in (reference, actual):
            document.del_xml_metadata()
            assert document.get_xml_metadata() == ""
            assert document.xref_xml_metadata() == 0
            with real_pymupdf.open(stream=document.tobytes(no_new_id=True)) as reopened:
                assert reopened.get_xml_metadata() == ""
        assert actual.xref_object(actual.pdf_catalog()) == reference.xref_object(
            reference.pdf_catalog()
        )


@pytest.mark.parametrize(
    "method", ["get_xml_metadata", "set_xml_metadata", "del_xml_metadata", "xref_xml_metadata"]
)
def test_xml_metadata_closed_document(method: str) -> None:
    results = []
    for engine in (real_pymupdf, compat_pymupdf):
        document = engine.open(internal_PDFS[0])
        document.close()
        with pytest.raises(Exception) as error:
            getattr(document, method)(*(["<x/>"] if method == "set_xml_metadata" else []))
        results.append((type(error.value).__name__, str(error.value)))
    assert results[0] == results[1]
