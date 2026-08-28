# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_07_syntax.objects import PdfObjectStream
from core_pdf.impl.engine.spec.s_07_syntax.stream import PdfStream


def internal_object_stream(values: tuple[int, ...]) -> PdfObjectStream:
    body_parts: list[bytes] = []
    header_parts: list[bytes] = []
    offset = 0
    for object_number, value in enumerate(values, start=1):
        encoded = str(value).encode("ascii")
        header_parts.append(f"{object_number} {offset}".encode("ascii"))
        body_parts.append(encoded)
        offset += len(encoded) + 1
    header = b" ".join(header_parts) + b" "
    body = b" ".join(body_parts)
    return PdfObjectStream(
        PdfStream(
            {"Type": "ObjStm", "N": len(values), "First": len(header)},
            decoded_data=header + body,
        )
    )


def test_object_stream_serializes_access_to_its_shared_lexer(monkeypatch: Any) -> None:
    values = tuple(range(100, 132))
    object_stream = internal_object_stream(values)
    original_parse_object_at = PdfLexer.parse_object_at

    def interleaved_parse_object_at(lexer: PdfLexer, position: int) -> object:
        if lexer is not object_stream.lexer:
            return original_parse_object_at(lexer, position)
        lexer.rewind(position)
        time.sleep(0.001)
        return lexer.parse_object()

    monkeypatch.setattr(PdfLexer, "parse_object_at", interleaved_parse_object_at)
    with ThreadPoolExecutor(max_workers=8) as executor:
        resolved = tuple(executor.map(object_stream.get, range(1, len(values) + 1)))

    assert resolved == values


def test_object_stream_caches_resolved_values() -> None:
    object_stream = internal_object_stream((101,))

    assert object_stream.get(1) == 101
    object_stream.raw_body = b"999"
    assert object_stream.get(1) == 101
