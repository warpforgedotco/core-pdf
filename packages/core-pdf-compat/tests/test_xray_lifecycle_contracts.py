from types import SimpleNamespace
from typing import Any

import pytest

from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf_compat import xray


@pytest.fixture
def opened_document(monkeypatch):
    class Document:
        closed = False
        decipher = None
        pages = (SimpleNamespace(page_number=1), SimpleNamespace(page_number=3))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

    document = Document()
    monkeypatch.setattr(xray.internal_XrayDocument, "open", lambda *args, **kwargs: document)
    monkeypatch.setattr(xray, "_validate_mupdf_structure", lambda document: None)
    return document


@pytest.mark.parametrize("failure", [KeyError, TypeError, ValueError, xray.TTLibError])
def test_page_failure_preserves_later_findings_and_closes_document(
    opened_document, monkeypatch, failure
):
    caches = []
    recovered = []

    def extract(page, cache):
        caches.append(cache)
        if page.page_number == 1:
            cache["font"] = {b"a": b"b"}
            raise failure("broken page")
        assert cache["font"] == {b"a": b"b"}
        return [{"text": "later secret"}]

    def recover(page):
        recovered.append(page.page_number)
        return [{"text": "recovered secret"}]

    monkeypatch.setattr(xray, "_page_redactions", extract)
    monkeypatch.setattr(xray, "_raw_highlight_redactions", recover)
    expected = {3: [{"text": "later secret"}]}
    if failure is xray.TTLibError:
        expected[1] = [{"text": "recovered secret"}]
    assert xray.inspect(b"input") == expected
    assert caches[0] is caches[1]
    assert recovered == ([1] if failure is xray.TTLibError else [])
    assert opened_document.closed


@pytest.mark.parametrize("failure", [RecursionError, xray.TTLibError])
def test_fatal_recovery_failure_discards_partial_findings_and_closes_document(
    opened_document, monkeypatch, failure
):
    def extract(page, cache):
        if page.page_number == 3:
            raise xray.TTLibError("broken font")
        return [{"text": "partial result"}]

    def recover(page):
        raise failure("cannot recover")

    monkeypatch.setattr(xray, "_page_redactions", extract)
    monkeypatch.setattr(xray, "_raw_highlight_redactions", recover)
    assert xray.inspect(b"input") == {}
    assert opened_document.closed


@pytest.mark.parametrize("boundary", ["validation", "extraction"])
def test_unexpected_failure_propagates_after_closing_document(
    opened_document, monkeypatch, boundary
):
    def fail(*args):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(
        xray, "_validate_mupdf_structure" if boundary == "validation" else "_page_redactions", fail
    )
    with pytest.raises(RuntimeError, match="unexpected failure"):
        xray.inspect(b"input")
    assert opened_document.closed


def test_password_requirement_stops_extraction_and_closes_document(opened_document, monkeypatch):
    monkeypatch.setattr(xray, "_requires_password", lambda document: True)
    with pytest.raises(PdfUnsupportedError, match="document closed or encrypted"):
        xray.inspect(b"input")
    assert opened_document.closed


@pytest.mark.parametrize(
    ("raw", "empty"),
    [
        (b"/Type /ObjStm", True),
        (b"/Type /ObjStm /Type /Pages", True),
        (b"/Type /ObjStm /Type /Page", False),
        (b"/Type /Page", False),
        (b"", False),
    ],
)
def test_unsupported_object_stream_recovery_requires_absence_of_page_objects(
    monkeypatch, raw, empty
):
    error = PdfUnsupportedError("unsupported")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(xray.internal_XrayDocument, "open", fail)
    if empty:
        assert xray.inspect(raw) == {}
    else:
        with pytest.raises(PdfUnsupportedError) as caught:
            xray.inspect(raw)
        assert caught.value is error


@pytest.mark.parametrize(
    "kind", ["bytes", "path", "string-path", "missing", "https", "unsupported"]
)
def test_source_byte_recovery_reads_only_supported_local_sources(tmp_path, kind):
    path = tmp_path / "source.pdf"
    path.write_bytes(b"pdf source")
    sources = {
        "bytes": b"pdf source",
        "path": path,
        "string-path": str(path),
        "missing": tmp_path / "absent",
        "https": "https://example.com/source.pdf",
        "unsupported": object(),
    }
    expected = b"pdf source" if kind in {"bytes", "path", "string-path"} else None
    assert xray._source_bytes(sources[kind]) == expected


@pytest.mark.parametrize(
    ("text", "retained"),
    [("12/03/2020", False), ("1-2-99", False), ("secret 12/03/2020", True), ("account", True)],
)
def test_date_only_findings_are_suppressed_without_losing_other_text(
    opened_document, monkeypatch, text, retained
):
    monkeypatch.setattr(xray, "_page_redactions", lambda page, cache: [{"text": text}])
    assert xray.inspect(b"input") == (
        {1: [{"text": text}], 3: [{"text": text}]} if retained else {}
    )
    assert opened_document.closed


@pytest.mark.parametrize(
    ("revision", "stored_hash", "required"),
    [(4, b"different", False), (5, b"empty", False), (6, b"different", True)],
)
def test_password_gate_checks_empty_password_only_for_newer_handlers(
    revision, stored_hash, required
):
    calls = []

    class Handler:
        r = revision
        u_validation_salt = b"salt"
        u_hash = stored_hash

        def decipher(self):
            pass

        def password_hash(self, password, salt):
            calls.append((password, salt))
            return b"empty"

    handler = Handler()
    document: Any = SimpleNamespace(decipher=handler.decipher)
    assert xray._requires_password(document) is required
    assert calls == ([(b"", b"salt")] if revision >= 5 else [])
