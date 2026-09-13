# SPDX-License-Identifier: AGPL-3.0-only
"""Cross-reference identity checks and owned object-stream parser lifetimes."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from core_pdf_spec.exceptions import PdfDecryptionError, PdfParseError, PdfUnsupportedError
from core_pdf_spec.s_07_syntax.objects import PdfObjectStream
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import CachedPdfObject, PdfDict
from core_pdf_spec.s_07_syntax.xref import PdfXRefEntry, key_for
from core_pdf_spec.standards import PdfVersion, SemanticContext
from core_pdf_spec.types import PdfName, PdfReference, PdfString


def object_stream() -> PdfStream:
    return PdfStream({"Type": PdfName.of("ObjStm"), "N": 2, "First": 8}, b"1 0 3 5 null 42")


@pytest.mark.parametrize("header", [b"2 0 obj", b"1 1 obj"])
def test_resolver_rejects_wrong_header_before_deciphering(header: bytes) -> None:
    calls: list[tuple[int, int]] = []

    def decipher(number: int, generation: int, data: bytes, dictionary: PdfDict | None) -> bytes:
        calls.append((number, generation))
        return data

    resolver = ObjectResolver(
        header + b" (value) endobj", {key_for(1): PdfXRefEntry(0)}, decipher=decipher
    )
    try:
        with pytest.raises(PdfParseError):
            resolver.resolve(PdfReference(1))
        assert calls == []
        assert key_for(1) not in resolver.objects
    finally:
        resolver.close()


def test_matching_header_deciphers_under_the_requested_identity() -> None:
    calls: list[tuple[int, int]] = []

    def decipher(number: int, generation: int, data: bytes, dictionary: PdfDict | None) -> bytes:
        calls.append((number, generation))
        return data + b"!"

    resolver = ObjectResolver(
        b"1 2 obj (value) endobj", {key_for(1, 2): PdfXRefEntry(0, 2)}, decipher=decipher
    )
    try:
        result = resolver.resolve(PdfReference(1, 2))
        assert isinstance(result, PdfString)
        assert result.data == b"value!"
        assert calls == [(1, 2)]
    finally:
        resolver.close()


@pytest.mark.parametrize("index", [None, -1, 1, 2, True])
def test_compressed_xref_requires_matching_ordinal(index: int | None) -> None:
    resolver = ObjectResolver(
        b"", {key_for(1): PdfXRefEntry(0, object_stream=2, index_in_stream=index)}
    )
    resolver.objects[key_for(2)] = object_stream()
    try:
        with pytest.raises(PdfParseError, match="compressed object"):
            resolver.resolve(PdfReference(1))
        assert key_for(1) not in resolver.objects
    finally:
        resolver.close()


@pytest.mark.parametrize("container", [None, 17, {}])
def test_in_use_compressed_xref_requires_a_stream_container(container: CachedPdfObject) -> None:
    resolver = ObjectResolver(
        b"", {key_for(1): PdfXRefEntry(0, object_stream=2, index_in_stream=0)}
    )
    resolver.objects[key_for(2)] = container
    try:
        with pytest.raises(PdfParseError, match="container"):
            resolver.resolve(PdfReference(1))
        assert resolver.resolve(PdfReference(99)) is None
    finally:
        resolver.close()


def test_compressed_null_is_a_valid_resolved_object() -> None:
    resolver = ObjectResolver(
        b"",
        {
            key_for(1): PdfXRefEntry(0, object_stream=2, index_in_stream=0),
            key_for(3): PdfXRefEntry(0, object_stream=2, index_in_stream=1),
        },
    )
    resolver.objects[key_for(2)] = object_stream()
    try:
        assert resolver.resolve(PdfReference(1)) is None
        assert key_for(1) in resolver.objects
        assert resolver.resolve(PdfReference(3)) == 42
        assert resolver.get_object_stream(2) is resolver.get_object_stream(2)
    finally:
        resolver.close()


def test_compressed_reference_requires_generation_zero() -> None:
    resolver = ObjectResolver(
        b"", {key_for(1, 1): PdfXRefEntry(0, object_stream=2, index_in_stream=0)}
    )
    resolver.objects[key_for(2)] = object_stream()
    try:
        with pytest.raises(PdfParseError, match="compressed object"):
            resolver.resolve(PdfReference(1, 1))
    finally:
        resolver.close()


@pytest.mark.parametrize("error_type", [PdfDecryptionError, PdfUnsupportedError, RuntimeError])
def test_object_stream_decoder_errors_propagate(error_type: type[Exception]) -> None:
    error = error_type("decoder failed")

    def decoder(
        data: bytes | memoryview, dictionary: object, *, parent_dictionary: object = None
    ) -> bytes:
        raise error

    stream = object_stream().replace(decoder=decoder)
    resolver = ObjectResolver(
        b"", {key_for(1): PdfXRefEntry(0, object_stream=2, index_in_stream=0)}
    )
    resolver.objects[key_for(2)] = stream
    try:
        with pytest.raises(error_type) as raised:
            resolver.resolve(PdfReference(1))
        assert raised.value is error
        assert resolver.object_streams == {}
    finally:
        resolver.close()


def test_competing_object_stream_factories_close_the_losing_parser() -> None:
    barrier = Barrier(2, timeout=5)
    candidates: list[PdfObjectStream] = []

    class Resolver(ObjectResolver):
        def create_object_stream(self, stream: PdfStream) -> PdfObjectStream:
            candidate = super().create_object_stream(stream)
            candidates.append(candidate)
            barrier.wait()
            return candidate

    resolver = Resolver(b"", {})
    resolver.objects[key_for(2)] = object_stream()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            parsers = list(executor.map(resolver.get_object_stream, [2, 2]))
        assert parsers[0] is parsers[1]
        assert parsers[0] is not None
        assert len(candidates) == 2
        winner = parsers[0]
        loser = next(candidate for candidate in candidates if candidate is not winner)
        assert winner.raw_body
        assert loser.raw_body == b""
        assert loser.index == {}
    finally:
        resolver.close()
    assert all(candidate.raw_body == b"" for candidate in candidates)


def test_context_change_and_close_release_cached_parsers_without_releasing_caller_view() -> None:
    source = memoryview(b"caller source")
    resolver = ObjectResolver(source, {})
    resolver.objects[key_for(2)] = object_stream()
    parser = resolver.get_object_stream(2)
    assert parser is not None
    try:
        resolver.semantic_context = SemanticContext(PdfVersion.parse("1.7"))
        assert parser.raw_body == b""
        assert resolver.object_streams == {}
        assert resolver.objects == {}
        resolver.objects[key_for(2)] = object_stream()
        current = resolver.get_object_stream(2)
        assert current is not None
        assert current.raw_body
        resolver.close()
        resolver.close()
        assert current.raw_body == b""
        assert bytes(source) == b"caller source"
    finally:
        resolver.close()
        source.release()
