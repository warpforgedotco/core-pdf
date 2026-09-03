from contextlib import closing
from typing import cast

import pytest

from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.primitives import PdfReference, PdfString
from core_pdf.impl.spec.s_07_syntax.resolver import (
    ObjectResolver,
    internal_find_indirect_object_header,
)
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_07_syntax.xref import PdfXRefEntry, key_for


def test_finds_indirect_object_header_in_full_backing_bytes() -> None:
    data = memoryview(b"prefix\n12 3 obj\nvalue")

    assert internal_find_indirect_object_header(data, 0, len(data)) == len(b"prefix\n")


def test_rejects_object_keyword_cut_off_at_search_end() -> None:
    data = memoryview(b"prefix\n12 3 object\nvalue")
    search_end = data.tobytes().index(b"obj") + len(b"obj")

    assert internal_find_indirect_object_header(data, 0, search_end) is None


def test_rejects_header_whose_token_starts_before_search_window() -> None:
    data = memoryview(b"\n12 3 obj\nvalue")

    assert internal_find_indirect_object_header(data, 2, len(data)) is None


def test_finds_header_from_sliced_memoryview_without_full_source_buffer() -> None:
    wrapped = memoryview(b"outsideprefix\n12 3 obj\nvalueoutside")
    data = wrapped[len(b"outside") : -len(b"outside")]

    assert internal_find_indirect_object_header(data, 0, len(data)) == len(b"prefix\n")


def test_deep_resolve_terminates_on_a_cyclic_reference_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A damaged file can encode 1 0 R -> 2 0 R -> 1 0 R.

    deep_resolve used to recurse into itself for a PdfReference while only ever
    recording container ids in ``seen``, so such a chain ran to RecursionError.
    """
    with closing(ObjectResolver(b"", {}, {})) as resolver:
        monkeypatch.setattr(
            type(resolver),
            "resolve",
            lambda self, ref: (
                PdfReference(2, 0)
                if type(ref) is PdfReference and ref.object_number == 1
                else PdfReference(1, 0)
                if type(ref) is PdfReference
                else ref
            ),
        )
        assert resolver.deep_resolve(PdfReference(1, 0)) == PdfReference(1, 0)


def test_resolve_str_does_not_expand_composite_object_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def internal_reject_deep_resolution(*args: object) -> object:
        del args
        raise AssertionError("composite value was deep-resolved")

    monkeypatch.setattr(ObjectResolver, "deep_resolve", internal_reject_deep_resolution)
    with closing(ObjectResolver(b"", {}, {})) as resolver:
        assert resolver.resolve_str([PdfReference(1), "XYZ"]) is None
        assert resolver.resolve_str(PdfString(b"https://example.invalid")) == (
            "https://example.invalid"
        )


def test_resolves_demanded_object_missing_from_damaged_xref() -> None:
    data = b"%PDF-1.7\n154 0 obj\n<< /Type /Font >>\nendobj\n"
    with closing(ObjectResolver(data, {}, {}, recover_missing=True)) as resolver:
        resolved = resolver.resolve(PdfReference(154))

    assert isinstance(resolved, dict)
    assert str(cast(PdfDict, resolved)["Type"]) == "Font"


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
