from io import BytesIO
from typing import Any

import pytest

from core_pdf.impl.document_document import PdfDocument, check_security_aliases
from core_pdf.impl.exceptions import (
    PdfDocumentClosedError,
    PdfEmptySourceError,
    PdfSourceError,
    PdfUnsupportedError,
)
from core_pdf_spec.s_07_syntax.types import PdfDict


@pytest.mark.parametrize(
    ("literal", "escaped"), [("Encrypt", "Encr#79pt"), ("AuthCode", "Auth#43ode"), ("ID", "I#44")]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_security_alias_collisions_are_rejected_regardless_of_order(
    text_pdf_bytes, literal, escaped, reverse
):
    pairs = [(literal, {}), (escaped, {})]
    with PdfDocument(text_pdf_bytes) as document:
        with pytest.raises(PdfUnsupportedError, match="Ambiguous security dictionary"):
            check_security_aliases(dict(reversed(pairs) if reverse else pairs), document.resolver)


@pytest.mark.parametrize("outer", ["Encrypt", "Encr#79pt", "AuthCode", "Auth#43ode"])
@pytest.mark.parametrize("nested", [False, True])
def test_security_dictionary_alias_checks_reach_nested_dictionaries(text_pdf_bytes, outer, nested):
    values: PdfDict = {"Length": 40, "L#65ngth": 128}
    if nested:
        values = {"CF": {"StdCF": values}}
    with PdfDocument(text_pdf_bytes) as document:
        with pytest.raises(PdfUnsupportedError, match="Ambiguous security dictionary"):
            check_security_aliases({outer: values}, document.resolver)


@pytest.mark.parametrize("outer", ["Encrypt", "AuthCode", "Root"])
def test_security_alias_walk_is_cycle_safe_and_ignores_unrelated_trailer_entries(
    text_pdf_bytes, outer
):
    values: PdfDict = {"Length": 40}
    values["Self"] = values
    trailer: PdfDict = {outer: values, "Unrelated": {"Foo": 1, "F#6fo": 2}}
    with PdfDocument(text_pdf_bytes) as document:
        check_security_aliases(trailer, document.resolver)


@pytest.mark.parametrize("kind", ["bytes", "bytearray", "memoryview", "literal", "reader"])
def test_source_forms_preserve_page_count_and_borrowed_reader_position(text_pdf_bytes, kind):
    reader = BytesIO(text_pdf_bytes)
    reader.seek(7)
    values = {
        "bytes": text_pdf_bytes,
        "bytearray": bytearray(text_pdf_bytes),
        "memoryview": memoryview(text_pdf_bytes),
        "literal": text_pdf_bytes.decode("latin-1"),
        "reader": reader,
    }
    with PdfDocument(values[kind]) as document:
        assert document.page_count() == 1
        assert not document.closed
    assert document.closed
    document.close()
    assert not reader.closed
    assert reader.tell() == 7
    with pytest.raises(PdfDocumentClosedError):
        document.__enter__()


def test_borrowed_file_remains_open_after_document_closes(text_pdf_bytes, tmp_path):
    path = tmp_path / "source.pdf"
    path.write_bytes(text_pdf_bytes)
    with path.open("rb") as reader:
        reader.seek(4)
        with PdfDocument(reader) as document:
            assert document.page_count() == 1
        assert reader.tell() == 4
        assert not reader.closed


@pytest.mark.parametrize("borrowed", [False, True])
def test_empty_file_source_has_a_clear_error(tmp_path, borrowed):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    if borrowed:
        with path.open("rb") as reader:
            with pytest.raises(PdfSourceError, match="empty"):
                PdfDocument(reader)
            assert not reader.closed
    else:
        with pytest.raises(PdfSourceError, match="empty"):
            PdfDocument(path)


class EmptyReader:
    def read(self, size: int = -1) -> bytes:
        return b""


@pytest.mark.parametrize(
    "source",
    [b"", bytearray(), memoryview(b""), BytesIO(b""), EmptyReader()],
    ids=["bytes", "bytearray", "memoryview", "BytesIO", "reader"],
)
def test_every_empty_source_fails_as_an_empty_file_does(source: Any) -> None:
    with pytest.raises(PdfEmptySourceError, match="PDF source is empty"):
        PdfDocument(source)


def test_unsupported_source_has_a_clear_error():
    value: Any = object()
    with pytest.raises(PdfSourceError, match="not supported"):
        PdfDocument(value)


@pytest.mark.parametrize("error_type", [OSError, TypeError, ValueError])
def test_unavailable_file_descriptor_falls_back_to_reader(text_pdf_bytes, error_type):
    class Reader(BytesIO):
        def fileno(self):
            raise error_type("unavailable")

    reader = Reader(text_pdf_bytes)
    reader.seek(7)
    with PdfDocument(reader) as document:
        assert document.page_count() == 1
    assert reader.tell() == 7
    assert not reader.closed


@pytest.mark.parametrize("error_type", [OSError, TypeError, ValueError])
def test_unavailable_position_falls_back_to_sequential_read(text_pdf_bytes, error_type):
    class Reader(BytesIO):
        def tell(self):
            raise error_type("unavailable")

    reader = Reader(text_pdf_bytes)
    with PdfDocument(reader) as document:
        assert document.page_count() == 1
    assert not reader.closed


@pytest.mark.parametrize("buffer_type", [bytes, bytearray, memoryview])
def test_read_only_source_accepts_binary_buffers(text_pdf_bytes, buffer_type):
    class Reader:
        def read(self, size=-1):
            return buffer_type(text_pdf_bytes if size < 0 else text_pdf_bytes[:size])

    with PdfDocument(Reader()) as document:
        assert document.page_count() == 1


def test_read_failure_preserves_position_and_borrowed_reader_ownership(text_pdf_bytes):
    class Reader(BytesIO):
        def read(self, *args):
            raise OSError("read failed")

    reader = Reader(text_pdf_bytes)
    reader.seek(7)
    with pytest.raises(PdfSourceError, match="read failed") as failure:
        PdfDocument(reader)
    assert isinstance(failure.value.__cause__, OSError)
    assert reader.tell() == 7
    assert not reader.closed
