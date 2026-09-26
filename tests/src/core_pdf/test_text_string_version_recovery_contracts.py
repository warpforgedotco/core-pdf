from __future__ import annotations

from collections.abc import Callable

import pytest

from core_pdf.impl.recovery_resolver import ObjectResolver
from core_pdf.impl.recovery_text_strings import decode_pdf_text_string
from core_pdf_spec.s_07_syntax_primitives.text_string import (
    decode_pdf_text_string as decode_spec_text_string,
)
from core_pdf_spec.standards import PdfVersion, SemanticContext, recognized_version

VERSIONS = (
    None,
    *(PdfVersion(1, minor) for minor in range(8)),
    PdfVersion(1, 8),
    PdfVersion(2, 0),
    PdfVersion(2, 1),
    PdfVersion(3, 0),
)

PAYLOADS = (
    b"",
    b"plain",
    b"price \xa0",
    b"\x80\x99\x9e\xa0",
    b"\xfe\xff\x00H\x00i",
    b"\xfe\xff\x00H\x00\xa0",
    b"\xfe\xff\x00",
    b"\xef\xbb\xbfcaf\xc3\xa9",
    b"\xef\xbb\xbf\xa0",
    b"\xef\xbb\xbf\xff",
    b"\xff\xfeH\x00i\x00",
    b"\xff\xfeH\x00\xa0\x00",
    b"\xff\xfeH",
)


def version_guarded_decode(data: bytes, context: SemanticContext | None) -> str:
    if data.startswith(b"\xff\xfe"):
        try:
            return data[2:].decode("utf-16-le")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-16LE data") from exc
    if context is not None and (
        (version := recognized_version(context)) is None
        or (version < PdfVersion(2, 0) and data.startswith(b"\xef\xbb\xbf"))
        or (version < PdfVersion(1, 2) and data.startswith(b"\xfe\xff"))
        or (version < PdfVersion(1, 3) and not data.startswith(b"\xfe\xff") and 0xA0 in data)
    ):
        context = None
    return decode_spec_text_string(data, context=context)


def outcome[T](decode: Callable[[T], str], data: T) -> tuple[str, str]:
    try:
        return ("ok", decode(data))
    except ValueError as error:
        return (type(error).__name__, str(error))


@pytest.mark.parametrize("version", VERSIONS, ids=str)
@pytest.mark.parametrize("data", PAYLOADS, ids=repr)
def test_resolver_decodes_text_as_the_version_guarded_path_did(
    version: PdfVersion | None, data: bytes
) -> None:
    context = None if version is None else SemanticContext(version)
    resolver = ObjectResolver(b"", {}, semantic_context=context)
    expected = outcome(lambda value: version_guarded_decode(value, context), data)

    assert outcome(resolver.decode_text, data) == expected
    assert outcome(decode_pdf_text_string, data) == expected
    assert outcome(decode_pdf_text_string, memoryview(data)) == expected
