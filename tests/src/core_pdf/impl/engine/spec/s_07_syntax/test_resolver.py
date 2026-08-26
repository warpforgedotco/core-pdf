from typing import cast

from core_pdf.impl.engine.spec.s_07_syntax.resolver import ObjectResolver
from core_pdf.impl.engine.spec.s_07_syntax.xref import PdfXRefEntry, key_for
from core_pdf.impl.primitives import PdfReference
from core_pdf.impl.types import PdfDict


def test_resolves_demanded_object_missing_from_damaged_xref() -> None:
    data = b"%PDF-1.7\n154 0 obj\n<< /Type /Font >>\nendobj\n"
    resolver = ObjectResolver(data, {}, {}, recover_missing=True)

    try:
        resolved = resolver.resolve(PdfReference(154))
    finally:
        resolver.close()

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
    resolver = ObjectResolver(data, xref, {})

    try:
        present = resolver.resolve(PdfReference(6))
        missing = resolver.resolve(PdfReference(7))
    finally:
        resolver.close()

    assert isinstance(present, dict)
    assert str(cast(PdfDict, present)["Type"]) == "Font"
    assert missing is None
