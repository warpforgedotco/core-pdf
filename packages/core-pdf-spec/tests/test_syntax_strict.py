# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_document.metadata import catalog_metadata_stream
from core_pdf_spec.s_07_document.page import iter_page_nodes, page_rotation
from core_pdf_spec.s_07_syntax.inherited_values import collect_inherited_values
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.objects import PdfObjectStream, parse_object_stream_header
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.resources import resolve_resource_dict
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.trees import (
    MalformedTreeNode,
    iter_name_tree_items,
    iter_number_tree_items,
)
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import XRefScanner, decode_xref_rows, key_for
from core_pdf_spec.types import PdfName, PdfReference, PdfString


@pytest.mark.parametrize(
    "data",
    [
        b"<6z1>",
        b"/A#XX",
        b"<<bad /Type /Page>>",
        b"<< /Bad ] /Type /Page >>",
        # Numeric arrays reject number spellings PDF does not have.
        b"[1_000 2]",
        b"[1.2e3 4]",
        b"[1_000 % ignored\n2]",
    ],
)
def test_lexer_rejects_malformed_values(data: bytes) -> None:
    lexer = PdfLexer(data)
    try:
        with pytest.raises(PdfParseError):
            lexer.parse_object()
    finally:
        lexer.close()


@pytest.mark.parametrize(
    "data",
    [
        b"1 0 obj endobj",
        b"1 0 obj 2 endobx",
        b"1 0 obj << /Length 9 >> stream\na\nendstream\nendobj",
    ],
)
def test_lexer_rejects_malformed_indirect_objects(data: bytes) -> None:
    lexer = PdfLexer(data)
    try:
        with pytest.raises(PdfParseError):
            lexer.parse_indirect_object()
        assert lexer.current_obj_num is None
        assert lexer.current_gen_num is None
    finally:
        lexer.close()


def test_pdf_string_defaults_and_odd_hex_padding() -> None:
    lexer = PdfLexer(b"[(a\\q) (a\n\rb) <6 1 2>]")
    try:
        assert [value.data for value in lexer.parse_object()] == [b"aq", b"a\n\nb", b"a "]
    finally:
        lexer.close()


@pytest.mark.parametrize(
    ("prefix", "suffix"), [(b"", b""), (b"[", b"]"), (b"[% comment\n", b"]"), (b"[(x) ", b"]")]
)
@pytest.mark.parametrize("sliced_view", [False, True])
def test_real_number_overflow_reports_implementation_limit(
    prefix: bytes, suffix: bytes, sliced_view: bool
) -> None:
    data = prefix + b"9" * 400 + b".0" + suffix
    lexer = PdfLexer(memoryview(b"x" + data)[1:] if sliced_view else data)
    try:
        with pytest.raises(PdfParseError, match="real number exceeds implementation limits"):
            lexer.parse_object()
    finally:
        lexer.close()


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"[]", []),
        (b"[+1 -2 .5 -3.]", [1, -2, 0.5, -3.0]),
        (b"[1 % ignored\n2]", [1, 2]),
        (b"[[1] 2 0 R]", [[1], PdfReference(2)]),
    ],
)
def test_numeric_arrays_and_general_arrays_leave_following_token(
    data: bytes, expected: list[object]
) -> None:
    lexer = PdfLexer(data + b" /Next")
    try:
        assert lexer.parse_object() == expected
        assert lexer.pos == len(data)
        assert lexer.parse_object() == PdfName.of("Next")
    finally:
        lexer.close()


@pytest.mark.parametrize("type_entry", [b" /Type /Sig", b" /Type /DocTimeStamp", b""])
def test_signature_contents_remain_unencrypted_when_type_follows(type_entry: bytes) -> None:
    calls: list[bytes] = []

    def decipher(number: int, generation: int, data: bytes, dictionary: PdfDict | None) -> bytes:
        assert (number, generation) == (7, 2)
        calls.append(data)
        return data + b"!"

    lexer = PdfLexer(
        b"7 2 obj << /Contents <6162> /ByteRange [0 2]" + type_entry + b" /Name (x) >> endobj",
        decipher=decipher,
    )
    try:
        parsed = lexer.parse_indirect_object()
        assert parsed["Contents"].data == b"ab"
        assert parsed["Name"].data == b"x!"
        assert calls == [b"x"]
        assert lexer.current_obj_num is None
    finally:
        lexer.close()


def test_nonsignature_contents_decipher_after_dictionary_classification() -> None:
    calls: list[bytes] = []

    def decipher(number: int, generation: int, data: bytes, dictionary: PdfDict | None) -> bytes:
        calls.append(data)
        return data + b"!"

    lexer = PdfLexer(
        b"1 0 obj << /Contents <61> /Name (b) /Type /Example >> endobj", decipher=decipher
    )
    try:
        assert lexer.parse_indirect_object()["Contents"].data == b"a!"
        assert calls == [b"b", b"a"]
    finally:
        lexer.close()


def test_object_stream_rejects_partial_header_and_duplicate_offsets() -> None:
    with pytest.raises(PdfParseError):
        parse_object_stream_header(b"1 0 ", 4, 2)
    stream = PdfStream({"Type": PdfName.of("ObjStm"), "N": 2, "First": 8}, b"1 0 2 0 (x)")
    with pytest.raises(PdfParseError):
        PdfObjectStream(stream)


def test_object_stream_parses_exact_offsets() -> None:
    stream = PdfStream({"Type": PdfName.of("ObjStm"), "N": 2, "First": 8}, b"1 0 2 4 (x) 17")
    objects = PdfObjectStream(stream)
    try:
        assert isinstance(objects.get(1), PdfString)
        assert objects.get(1).data == b"x"
        assert objects.get(2) == 17
    finally:
        objects.close()


def test_xref_unknown_type_is_null_and_missing_reference_resolves_null() -> None:
    entries = decode_xref_rows(bytes([7, 0, 0]), [1, 1, 1], [1, 1], 2)
    assert not entries[key_for(1)].in_use
    resolver = ObjectResolver(b"", entries)
    try:
        assert resolver.resolve(PdfReference(1)) is None
        assert resolver.resolve(PdfReference(2)) is None
    finally:
        resolver.close()


def test_xref_rejects_invalid_generation_and_truncated_rows() -> None:
    with pytest.raises(PdfParseError, match="generation"):
        decode_xref_rows(b"\x01\x00\x01\x00\x00", [1, 1, 3], [0, 1], 1)
    with pytest.raises(PdfParseError, match="length"):
        decode_xref_rows(b"\x01\x00", [1, 1, 1], [0, 1], 1)
    with pytest.raises(PdfParseError, match="Size"):
        XRefScanner.parse_table_section(b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< >>", 0)


@pytest.mark.parametrize("tree", [{"Nums": [0, "ok", 1]}, {"Nums": ["bad", 1]}, {"Kids": [17]}])
def test_number_tree_rejects_malformed_nodes_and_entries(tree: dict) -> None:
    with pytest.raises(ValueError):
        list(iter_number_tree_items(tree, lambda value: value))


def test_number_tree_reports_a_null_kid_with_its_node() -> None:
    # A null root is an empty tree (ISO 32000-2, 7.9.7); a null kid is not a
    # dictionary, so strict traversal rejects it and names the node it saw.
    tree = {"Kids": [None, {"Nums": [1, "one"]}]}
    assert list(iter_number_tree_items(None, lambda value: value)) == []
    with pytest.raises(ValueError, match="invalid number tree node"):
        list(iter_number_tree_items(tree, lambda value: value))
    reports: list[str] = []
    items = list(iter_number_tree_items(tree, lambda value: value, on_malformed=reports.append))
    assert items == [(1, "one")]
    (report,) = reports
    assert isinstance(report, MalformedTreeNode)
    assert report == "invalid number tree node"
    assert report.node is None


def test_a_key_that_fails_to_decode_is_a_malformed_entry() -> None:
    def decode(value: object) -> str:
        if value == b"bad":
            raise ValueError("undecodable name tree key")
        return str(value)

    tree = {"Names": [b"bad", 1, "good", 2]}
    with pytest.raises(ValueError, match="undecodable name tree key"):
        list(iter_name_tree_items(tree, lambda value: value, decode))
    reports: list[str] = []
    items = list(
        iter_name_tree_items(tree, lambda value: value, decode, on_malformed=reports.append)
    )
    assert items == [("good", 2)]
    assert reports == ["undecodable name tree key"]


def test_inheritance_resource_and_page_tree_damage_is_rejected() -> None:
    with pytest.raises(ValueError, match="parent"):
        collect_inherited_values(
            {"MediaBox": [0, 0, 10, 10], "Parent": 7}, ("MediaBox",), lambda value: value
        )
    resolver = ObjectResolver(b"", {})
    try:
        with pytest.raises(PdfParseError):
            resolve_resource_dict(17, resolver)
        assert resolve_resource_dict(PdfReference(999), resolver) is None
    finally:
        resolver.close()
    with pytest.raises(ValueError):
        list(iter_page_nodes({"Type": PdfName.of("Pages"), "Kids": [17]}, lambda value: value))
    assert page_rotation(None) == 0


@pytest.mark.parametrize(
    "data",
    [
        b"barekeyword",
        b"/A#00",
        b"0 0 obj 1 endobj",
        b"1 0 obj << /Length 1 >> stream\ra\nendstream\nendobj",
    ],
)
def test_strict_lexical_grammar_rejects_reader_extensions(data: bytes) -> None:
    lexer = PdfLexer(data)
    try:
        with pytest.raises(PdfParseError):
            if b" obj " in data:
                lexer.parse_indirect_object()
            else:
                lexer.parse_object()
    finally:
        lexer.close()


def test_metadata_rejects_malformed_catalog_but_preserves_null_and_absence() -> None:
    resolver = ObjectResolver(b"", {})
    try:
        with pytest.raises(ValueError, match="Root"):
            catalog_metadata_stream(resolver, 17)
        assert catalog_metadata_stream(resolver, {}) is None
        stream = PdfStream({}, b"<x/>")
        assert catalog_metadata_stream(resolver, {"Metadata": stream}) is stream
        with pytest.raises(ValueError, match="Metadata"):
            catalog_metadata_stream(resolver, {"Metadata": 17})
    finally:
        resolver.close()
