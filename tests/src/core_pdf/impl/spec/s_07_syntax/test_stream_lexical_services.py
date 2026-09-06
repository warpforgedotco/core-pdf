# SPDX-License-Identifier: AGPL-3.0-only
"""Length, decryption, and decoding share the same stream parser lifecycle."""

import pytest

from core_pdf.impl._impl.document.recovery.lexer import PdfLexer as ReaderLexer
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.types import PdfReference


@pytest.mark.parametrize("lexer_type", [PdfLexer, ReaderLexer])
def test_indirect_length_and_stream_services_keep_object_identity(
    lexer_type: type[PdfLexer],
) -> None:
    length_calls: list[PdfReference] = []
    decrypt_calls: list[tuple[int, int, bytes]] = []
    decode_calls: list[bytes] = []

    def resolve_length(reference: PdfReference) -> int:
        length_calls.append(reference)
        return 3

    def decrypt(number: int, generation: int, data: bytes, dictionary: PdfDict | None) -> bytes:
        decrypt_calls.append((number, generation, data))
        assert dictionary is not None
        assert dictionary["Length"] == PdfReference(12, 0)
        return b"plain"

    def decode(
        data: bytes | memoryview,
        dictionary: object,
        *,
        parent_dictionary: object = None,
    ) -> bytes:
        decode_calls.append(bytes(data))
        return bytes(data) + b" decoded"

    lexer = lexer_type(
        b"7 2 obj << /Length 12 0 R >> stream\nabc\nendstream\nendobj 42",
        reference_resolver=resolve_length,
        decipher=decrypt,
        stream_decoder=decode,
    )
    stream = lexer.parse_indirect_object()
    assert isinstance(stream, PdfStream)
    assert bytes(stream.raw_data) == b"plain"
    assert stream.data == b"plain decoded"
    assert lexer.parse_object() == 42
    assert length_calls == [PdfReference(12, 0)]
    assert decrypt_calls == [(7, 2, b"abc")]
    assert decode_calls == [b"plain"]
