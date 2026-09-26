"""The resolver reuses its file lexers between objects, one per reader at a time."""

import re

from core_pdf import PdfDocument
from core_pdf.impl.document.recovery.resolver import LEXER_POOL_LIMIT
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.types import PdfReference
from tests.src.core_pdf.test_document_pass_contracts import multi_page_pdf


def test_a_released_lexer_serves_the_next_object() -> None:
    with PdfDocument(multi_page_pdf()) as document:
        resolver = document.resolver
        first = resolver.get_lexer()
        resolver.release_lexer(first)
        assert resolver.get_lexer() is first


def test_a_lexer_in_use_is_not_handed_out_again() -> None:
    with PdfDocument(multi_page_pdf()) as document:
        resolver = document.resolver
        outer = resolver.get_lexer()
        inner = resolver.get_lexer()
        assert inner is not outer
        resolver.release_lexer(inner)
        resolver.release_lexer(outer)


def test_a_reused_lexer_takes_the_resolvers_current_decipher() -> None:
    with PdfDocument(multi_page_pdf()) as document:
        resolver = document.resolver
        resolver.release_lexer(resolver.get_lexer())

        def decipher(*_: object) -> bytes:
            return b""

        original = resolver.decipher
        resolver.decipher = decipher
        try:
            assert resolver.get_lexer().decipher is decipher
        finally:
            resolver.decipher = original


def test_the_pool_is_bounded_and_emptied_with_the_caches() -> None:
    with PdfDocument(multi_page_pdf()) as document:
        resolver = document.resolver
        lexers = [resolver.get_lexer() for _ in range(LEXER_POOL_LIMIT + 2)]
        for lexer in lexers:
            resolver.release_lexer(lexer)
        assert len(resolver.lexer_pool) == LEXER_POOL_LIMIT
        assert lexers[-1].data_len == 0  # closed, not pooled
        resolver.detach_parsed_caches()
        assert resolver.lexer_pool == []
        assert lexers[0].data_len == 0


def address(value: object) -> str:
    return re.sub(r"0x[0-9a-f]+", "", repr(getattr(value, "dictionary", value)))


def test_objects_read_through_reused_lexers_are_the_same() -> None:
    data = multi_page_pdf()
    with PdfDocument(data) as pooled, PdfDocument(data) as fresh:
        fresh.resolver.lexer_pool.clear()
        for number in range(1, 10):
            reference = PdfReference(number, 0)
            left = pooled.resolver.resolve(reference)
            fresh.resolver.detach_parsed_caches()
            right = fresh.resolver.resolve(reference)
            assert type(left) is type(right)
            assert address(left) == address(right)
            if isinstance(left, PdfStream) and isinstance(right, PdfStream):
                assert left.data == right.data
