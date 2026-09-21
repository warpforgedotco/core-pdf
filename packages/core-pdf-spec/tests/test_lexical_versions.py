# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf_spec.s_07_content.inline_images import validate_inline_images
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.operations import ContentOperands, iter_content_operations
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.objects import PdfObjectStream, parse_object_stream_header
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.xref import PdfXRefEntry, key_for
from core_pdf_spec.s_07_syntax_primitives.scanning import (
    looks_like_indirect_object_header,
    skip_name,
    skip_pdf_ignored,
)
from core_pdf_spec.s_07_syntax_primitives.tokens import lexical_rules
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.standards import PdfVersion, SemanticContext
from core_pdf_spec.types import PdfName, PdfReference, PdfString


def internal_parse(data: bytes, version: str) -> object:
    lexer = PdfLexer(data, semantic_context=SemanticContext(PdfVersion.parse(version)))
    try:
        return lexer.parse_object()
    finally:
        lexer.close()


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_legacy_names_keep_literal_number_signs_in_keys_and_values(version: str) -> None:
    assert internal_parse(b"<< /A#42 /C#44 /AB /other /# /#zz >>", version) == {
        "A#42": PdfName.of("C#44"),
        "AB": PdfName.of("other"),
        "#": PdfName.of("#zz"),
    }


@pytest.mark.parametrize("version", ["1.2", "1.3", "1.7", "2.0"])
def test_modern_names_decode_escapes_and_reject_malformed_ones(version: str) -> None:
    assert internal_parse(b"<< /A#42 /C#44 >>", version) == {"AB": PdfName.of("CD")}
    for name in (b"/#", b"/#zz", b"/#00"):
        with pytest.raises(PdfParseError):
            internal_parse(name, version)


@pytest.mark.parametrize("version", ["1.0", "1.1", "1.2"])
@pytest.mark.parametrize("data", [b"[12\0 34]", b"<4\0 1>", b"\0true"])
def test_earlier_whitespace_does_not_silently_consume_nul(version: str, data: bytes) -> None:
    with pytest.raises(PdfParseError):
        internal_parse(data, version)


@pytest.mark.parametrize("version", ["1.3", "1.7", "2.0"])
def test_nul_whitespace_works_in_arrays_hex_strings_and_keywords(version: str) -> None:
    assert internal_parse(b"[12\0 34]", version) == [12, 34]
    assert internal_parse(b"<4\0 1>", version) == PdfString(b"A", is_literal=False)
    assert internal_parse(b"\0true", version) is True


@pytest.mark.parametrize("version", ["1.0", "1.2", "1.3", "2.0"])
def test_comments_literal_data_and_standard_whitespace_are_shared(version: str) -> None:
    assert internal_parse(b"% comment\0with NUL\n[1\t2\r3\n4\f5 6]", version) == [1, 2, 3, 4, 5, 6]
    assert internal_parse(b"(data\0stays)", version) == PdfString(b"data\0stays", is_literal=True)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(("version", "end"), [("1.2", 0), ("1.3", 1)])
def test_scanners_use_selected_rules_in_contiguous_and_strided_views(
    strided: bool, version: str, end: int
) -> None:
    raw = b"\0rest"
    data = (
        memoryview(b"".join(bytes((byte, 255)) for byte in raw))[::2]
        if strided
        else memoryview(raw)
    )
    rules = lexical_rules(SemanticContext(PdfVersion.parse(version)))
    try:
        assert skip_pdf_ignored(data, 0, len(data), rules=rules) == end
        assert skip_pdf_ignored(b"%abc\0\n\0rest", 0, 11, rules=rules) == 6 + end
        assert skip_name(b"/Name\0next", 0, 10, rules=rules) == (10 if end == 0 else 5)
        header = memoryview(b"1\0 0 obj ")
        assert looks_like_indirect_object_header(header, 0, len(header), rules=rules) == bool(end)
        header.release()
    finally:
        data.release()


def test_skip_ignored_respects_the_supplied_buffer_limit() -> None:
    assert skip_pdf_ignored(b"%comment\n  suffix", 0, 4) == 4


@pytest.mark.parametrize("version", [None, PdfVersion(1, 8), PdfVersion(2, 1)])
def test_unknown_lexical_context_fails_without_changing_an_existing_lexer(
    version: PdfVersion | None,
) -> None:
    with pytest.raises(PdfUnsupportedError, match="recognized PDF version"):
        PdfLexer(b"/A#42", semantic_context=SemanticContext(version))
    lexer = PdfLexer(b"/A#42")
    try:
        with pytest.raises(PdfUnsupportedError):
            lexer.semantic_context = SemanticContext(version)
        assert lexer.parse_object() == PdfName.of("AB")
    finally:
        lexer.close()


def test_lexer_context_change_affects_subsequent_objects() -> None:
    lexer = PdfLexer(b"/A#42 /A#42", semantic_context=SemanticContext(PdfVersion(1, 1)))
    try:
        assert lexer.parse_object() == PdfName.of("A#42")
        lexer.semantic_context = SemanticContext(PdfVersion(1, 2))
        assert lexer.parse_object() == PdfName.of("AB")
    finally:
        lexer.close()


def test_resolver_reparses_cached_names_when_bootstrap_context_changes() -> None:
    data = b"1 0 obj << /A#42 /C#44 >> endobj"
    resolver = ObjectResolver(data, {key_for(1): PdfXRefEntry(0)})
    try:
        reference = PdfReference(1, 0)
        bootstrap = resolver.resolve(reference)
        assert bootstrap == {"AB": PdfName.of("CD")}
        resolver.semantic_context = SemanticContext(PdfVersion(1, 1))
        legacy = resolver.resolve(reference)
        assert legacy == {"A#42": PdfName.of("C#44")}
        assert bootstrap == {"AB": PdfName.of("CD")}
        resolver.semantic_context = SemanticContext(PdfVersion(1, 2))
        assert resolver.resolve(reference) == bootstrap
        assert resolver.resolve(reference) is not bootstrap
    finally:
        resolver.close()


def test_context_change_rebuilds_cached_object_stream_parsers() -> None:
    body = b"2 0 /A#42"
    data = (
        b"1 0 obj << /Type /ObjStm /N 1 /First 4 /Length "
        + str(len(body)).encode()
        + b" >> stream\n"
        + body
        + b"\nendstream endobj"
    )
    resolver = ObjectResolver(
        data,
        {
            key_for(1): PdfXRefEntry(0),
            key_for(2): PdfXRefEntry(0, object_stream=1, index_in_stream=0),
        },
    )
    try:
        assert resolver.resolve(PdfReference(2, 0)) == PdfName.of("AB")
        resolver.semantic_context = SemanticContext(PdfVersion(1, 1))
        assert resolver.resolve(PdfReference(2, 0)) == PdfName.of("A#42")
    finally:
        resolver.close()


@pytest.mark.parametrize(("version", "expected"), [("1.1", "A#42"), ("1.5", "AB")])
def test_decoded_object_stream_body_keeps_the_supplied_context(version: str, expected: str) -> None:
    stream = PdfStream(
        raw_data=b"2 0 /A#42", dictionary={"Type": PdfName.of("ObjStm"), "N": 1, "First": 4}
    )
    parser = PdfObjectStream(stream, semantic_context=SemanticContext(PdfVersion.parse(version)))
    try:
        assert parser.get(2) == PdfName.of(expected)
    finally:
        parser.close()


def test_object_stream_header_uses_selected_whitespace() -> None:
    with pytest.raises(PdfParseError):
        parse_object_stream_header(b"2\0 0 /A", 5, 1, context=SemanticContext(PdfVersion(1, 2)))
    assert parse_object_stream_header(
        b"2\0 0 /A", 5, 1, context=SemanticContext(PdfVersion(1, 3))
    ) == [(2, 0)]


@pytest.mark.parametrize(("version", "expected"), [("1.1", "A#42"), ("2.0", "AB")])
def test_nested_content_streams_share_the_document_name_rules(version: str, expected: str) -> None:
    tags: list[str] = []

    class Sink:
        def __getattr__(self, name: str) -> Any:
            return lambda *args, **kwargs: None

    resolver = ObjectResolver(b"", {})
    context = SemanticContext(PdfVersion.parse(version))
    state = ContentInterpreter(
        resolver, cast(Any, Sink()), cast(Any, None), semantic_context=context
    )

    def record_tag(operands: ContentOperands, depth: int) -> None:
        tags.append(str(operands[0]))
        state.op_BMC(operands, depth)

    state.operator_overrides["BMC"] = record_tag
    child = PdfStream(
        raw_data=b"/A#42 BMC EMC",
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 1, 1], "Resources": {}},
    )
    try:
        state.stream_executor.consume(
            PdfStream(raw_data=b"/A#42 BMC /Child Do EMC"),
            {"XObject": {"Child": child}},
            IDENTITY_MATRIX,
            0,
        )
        assert tags == [expected, expected]
    finally:
        resolver.close()


def test_content_tokens_and_inline_image_boundaries_use_selected_whitespace() -> None:
    raw = b"BI /W 1 /H 1 /BPC 8 /CS /G ID\0x EI"
    with pytest.raises(PdfParseError, match="data separator"):
        validate_inline_images(raw, context=SemanticContext(PdfVersion(1, 2)))
    validate_inline_images(raw, context=SemanticContext(PdfVersion(1, 3)))
    lexer = PdfLexer(b"q\0Q", semantic_context=SemanticContext(PdfVersion(1, 2)))
    try:
        assert list(iter_content_operations(lexer)) == [("q\0Q", ())]
    finally:
        lexer.close()


def test_unconfigured_interpreter_retains_a_custom_factory_context() -> None:
    context = SemanticContext(PdfVersion(1, 1))
    state = ContentInterpreter(
        cast(Any, None),
        cast(Any, None),
        cast(Any, None),
        lexer_factory=lambda data: PdfLexer(data, semantic_context=context),
    )
    lexer = state.create_lexer(b"/A#42")
    try:
        assert lexer.parse_object() == PdfName.of("A#42")
    finally:
        lexer.close()
