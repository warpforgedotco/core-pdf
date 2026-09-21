from typing import Any

import pytest

from core_pdf.impl._impl.document.document import PdfDocument
from core_pdf_spec.s_07_syntax.stream import PdfStream


@pytest.mark.parametrize("unicode_name", [None, b"", b"unicode.txt"])
@pytest.mark.parametrize("legacy_name", [None, b"", b"legacy.txt"])
@pytest.mark.parametrize("unicode_stream", [False, True])
def test_attachment_name_and_stream_precedence_preserve_identity(
    text_pdf_bytes, unicode_name, legacy_name, unicode_stream
):
    with PdfDocument(text_pdf_bytes) as document:
        legacy = PdfStream({}, b"legacy payload")
        unicode = PdfStream({}, b"unicode payload")
        streams = {"F": legacy}
        if unicode_stream:
            streams["UF"] = unicode
        spec = {"UF": unicode_name, "F": legacy_name, "EF": streams}
        document.catalog()["Names"] = {"EmbeddedFiles": {"Names": [b"entry", spec]}}
        (record,) = document.embedded_files()
        assert record.name == "entry"
        assert record.filename == (unicode_name or legacy_name or b"entry").decode()
        assert record.filespec is spec
        assert record.stream is (unicode if unicode_stream else legacy)
        assert record.data == (b"unicode payload" if unicode_stream else b"legacy payload")


@pytest.mark.parametrize("invalid", [None, 1, {}, {"EF": 1}, {"EF": {}}, {"EF": {"F": b"invalid"}}])
@pytest.mark.parametrize("recover", [False, True])
def test_attachment_recovery_retains_valid_name_tree_siblings(text_pdf_bytes, invalid, recover):
    with PdfDocument(text_pdf_bytes) as document:
        good = {"F": b"good.txt", "EF": {"F": PdfStream({}, b"data")}}
        document.catalog()["Names"] = {"EmbeddedFiles": {"Names": [b"bad", invalid, b"good", good]}}
        document.xref_was_recovered = recover
        if recover:
            records = document.embedded_files()
            assert [(r.filename, r.data) for r in records] == [("good.txt", b"data")]
        else:
            with pytest.raises(ValueError, match="invalid embedded file"):
                document.embedded_files()


@pytest.mark.parametrize("names", [None, 1, {}, {"EmbeddedFiles": None}, {"EmbeddedFiles": {}}])
def test_absent_attachment_trees_return_empty_projection(text_pdf_bytes, names):
    with PdfDocument(text_pdf_bytes) as document:
        document.catalog()["Names"] = names
        assert document.embedded_files() == []


@pytest.mark.parametrize("tree", [1, [], b"invalid"])
def test_invalid_attachment_tree_type_is_rejected(text_pdf_bytes, tree):
    with PdfDocument(text_pdf_bytes) as document:
        document.catalog()["Names"] = {"EmbeddedFiles": tree}
        with pytest.raises(ValueError, match="invalid EmbeddedFiles name tree"):
            document.embedded_files()


def test_attachment_aliases_share_original_filespec_and_stream(text_pdf_bytes):
    with PdfDocument(text_pdf_bytes) as document:
        stream = PdfStream({}, b"same bytes")
        spec: dict[str, Any] = {"EF": {"F": stream}}
        document.catalog()["Names"] = {"EmbeddedFiles": {"Names": [b"a", spec, b"b", spec]}}
        a, b = document.embedded_files()
        assert a.name == a.filename == "a"
        assert b.name == b.filename == "b"
        assert a.filespec is b.filespec is spec
        assert a.stream is b.stream is stream
        assert a.data == b.data == b"same bytes"
