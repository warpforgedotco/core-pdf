from contextlib import closing
from typing import cast

import pytest

from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf.impl._impl.document.recovery.xref import iter_indirect_object_headers
from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_07_syntax.xref import PdfXRefEntry, key_for
from core_pdf.impl.types import PdfReference, PdfString


def test_finds_indirect_object_header_in_full_backing_bytes() -> None:
    data = memoryview(b"prefix\n12 3 obj\nvalue")

    assert list(iter_indirect_object_headers(data, 0, len(data))) == [(len(b"prefix\n"), 12, 3)]


def test_rejects_object_keyword_cut_off_at_search_end() -> None:
    data = memoryview(b"prefix\n12 3 object\nvalue")
    search_end = data.tobytes().index(b"obj") + len(b"obj")

    assert list(iter_indirect_object_headers(data, 0, search_end)) == []


def test_rejects_header_whose_token_starts_before_search_window() -> None:
    data = memoryview(b"\n12 3 obj\nvalue")

    assert list(iter_indirect_object_headers(data, 2, len(data))) == []
    assert list(
        iter_indirect_object_headers(data, 2, len(data), allow_prefix_before_start=True)
    ) == [(1, 12, 3)]


def test_finds_header_from_sliced_memoryview_without_full_source_buffer() -> None:
    wrapped = memoryview(b"outsideprefix\n12 3 obj\nvalueoutside")
    data = wrapped[len(b"outside") : -len(b"outside")]

    assert list(iter_indirect_object_headers(data, 0, len(data))) == [(len(b"prefix\n"), 12, 3)]


def test_resolves_demanded_object_missing_from_damaged_xref() -> None:
    data = b"%PDF-1.7\n154 0 obj\n<< /Type /Font >>\nendobj\n"
    with closing(ObjectResolver(data, {}, {}, recover_missing=True)) as resolver:
        resolved = resolver.resolve(PdfReference(154))

    assert isinstance(resolved, dict)
    assert str(cast(PdfDict, resolved)["Type"]) == "Font"


def test_missing_object_recovery_keeps_generation_and_latest_revision_order() -> None:
    data = b"1 0 obj\n(old)\nendobj\n1 1 obj\n(other generation)\nendobj\n1 0 obj\n(new)\nendobj\n"
    with closing(ObjectResolver(data, {}, {}, recover_missing=True)) as resolver:
        assert resolver.resolve(PdfReference(1, 0)) == PdfString(b"new")
        assert resolver.resolve(PdfReference(1, 1)) == PdfString(b"other generation")


def test_object_missing_from_damaged_object_stream_resolves_to_none() -> None:
    # Object stream 5 declares only object 6; the xref claims 7 also lives
    # there. The missing entry must resolve to None, not leak a default value.
    header = b"6 0 "
    body = b"<< /Type /Font >>"
    stream_content = header + body
    prefix = b"%PDF-1.7\n"
    data = (
        prefix
        + b"5 0 obj\n<< /Type /ObjStm /N 1 /First "
        + str(len(header)).encode()
        + b" /Length "
        + str(len(stream_content)).encode()
        + b" >>\nstream\n"
        + stream_content
        + b"\nendstream\nendobj\n"
    )
    xref = {
        key_for(5, 0): PdfXRefEntry(len(prefix)),
        key_for(6, 0): PdfXRefEntry(0, object_stream=5, index_in_stream=0),
        key_for(7, 0): PdfXRefEntry(0, object_stream=5, index_in_stream=1),
    }
    with closing(ObjectResolver(data, xref, {})) as resolver:
        present = resolver.resolve(PdfReference(6))
        missing = resolver.resolve(PdfReference(7))

    assert isinstance(present, dict)
    assert str(cast(PdfDict, present)["Type"]) == "Font"
    assert missing is None


def test_damaged_xref_recovery_does_not_swallow_unsupported_security_error() -> None:
    data = b"1 0 obj\n(encrypted)\nendobj\n"

    def internal_reject_decipher(
        object_number: int,
        generation_number: int,
        value: bytes,
        dictionary: PdfDict | None,
    ) -> bytes:
        del object_number, generation_number, value, dictionary
        raise PdfUnsupportedError("unsupported security configuration")

    with closing(
        ObjectResolver(
            data,
            {},
            {},
            decipher=internal_reject_decipher,
            recover_missing=True,
        )
    ) as resolver:
        with pytest.raises(PdfUnsupportedError, match="unsupported security configuration"):
            resolver.resolve(PdfReference(1))
