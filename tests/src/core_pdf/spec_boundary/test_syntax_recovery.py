# SPDX-License-Identifier: AGPL-3.0-only
"""Reader recovery remains core-owned across the standalone syntax boundary."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from core_pdf.impl._impl.document.metadata import resolve_metadata_stream
from core_pdf.impl._impl.document.page_tree import collect_inherited_values, iter_page_nodes
from core_pdf.impl._impl.document.recovery.lexer import PdfLexer
from core_pdf.impl._impl.document.recovery.objects import PdfObjectStream
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf.impl._impl.document.recovery.resources import resolve_resource_dict
from core_pdf.impl._impl.document.recovery.trees import (
    iter_name_tree_items,
    iter_number_tree_items,
)
from core_pdf.impl._impl.document.recovery.xref import XRefScanner
from core_pdf.impl.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer as StrictLexer
from core_pdf_spec.s_07_syntax.objects import PdfObjectStream as StrictObjectStream
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import XRefScanner as StrictXRefScanner
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.types import PdfName, PdfReference, PdfString


@pytest.mark.parametrize(
    "data", [b"<6z1>", b"/A#XX", b"<<bad /Type /Page>>", b"<< /Bad ] /Type /Page >>"]
)
def test_reader_recovers_values_rejected_by_strict_lexer(data: bytes) -> None:
    strict = StrictLexer(data)
    reader = PdfLexer(data)
    try:
        with pytest.raises(PdfParseError):
            strict.parse_object()
        recovered = reader.parse_object()
        if data.startswith(b"<6"):
            assert recovered.data == b"a\x00"
        elif data.startswith(b"/"):
            assert recovered == PdfName.of("A#XX")
        else:
            assert recovered["Type"] == PdfName.of("Page")
    finally:
        strict.close()
        reader.close()


@pytest.mark.parametrize(
    ("data", "expected"), [(b"1 0 obj endobj", None), (b"1 0 obj 2 endobx", 2)]
)
def test_reader_preserves_indirect_object_repairs(data: bytes, expected: object) -> None:
    lexer = PdfLexer(data)
    try:
        assert lexer.parse_indirect_object() == expected
    finally:
        lexer.close()


def test_reader_recovers_stream_length_without_changing_decipher_timing() -> None:
    calls: list[bytes] = []

    def decipher(number: int, generation: int, data: bytes, dictionary: PdfDict | None) -> bytes:
        assert (number, generation) == (1, 0)
        calls.append(data)
        return data

    lexer = PdfLexer(b"1 0 obj << /Length 99 >> stream\na\nendstream\nendobj", decipher=decipher)
    try:
        stream = lexer.parse_indirect_object()
        assert bytes(stream.raw_data) == b"a\n"
        # Existing reader resynchronization clears the object context.
        assert calls == []
        assert lexer.current_obj_num is None
    finally:
        lexer.close()


def test_reader_preserves_legacy_string_line_endings() -> None:
    lexer = PdfLexer(b"(a\n\rb)")
    try:
        assert lexer.parse_object().data == b"a\nb"
    finally:
        lexer.close()


@pytest.mark.parametrize(
    "make_buffer",
    [
        pytest.param(bytes, id="bytes"),
        pytest.param(bytearray, id="bytearray"),
        pytest.param(memoryview, id="memoryview"),
        pytest.param(lambda data: memoryview(b"xx" + data + b"yy")[2:-2], id="view-slice"),
        pytest.param(
            lambda data: memoryview(b"".join(bytes((byte, 0)) for byte in data))[::2],
            id="strided-view",
        ),
    ],
)
@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"[]", []),
        (b"[+1 -2 .5 -3.]", [1, -2, 0.5, -3.0]),
        (b"[1_000 2]", [1000, 2]),
        (b"[1.2e3 4]", [1200.0, 4]),
        (b"[1_000 % ignored\n2]", [1000, 2]),
        (b"[[1] 2 0 R]", [[1], PdfReference(2)]),
    ],
)
def test_reader_numeric_arrays_preserve_buffer_and_cursor_behavior(
    make_buffer: Callable[[bytes], bytes | bytearray | memoryview],
    data: bytes,
    expected: list[object],
) -> None:
    lexer = PdfLexer(make_buffer(data + b" /Next"))
    try:
        assert lexer.parse_object() == expected
        assert lexer.pos == len(data)
        assert lexer.parse_object() == PdfName.of("Next")
    finally:
        lexer.close()


def test_reader_recovers_partial_object_stream_header() -> None:
    stream = PdfStream({"Type": PdfName.of("ObjStm"), "N": 2, "First": 4}, b"1 0 (x)")
    with pytest.raises(PdfParseError):
        StrictObjectStream(stream)
    objects = PdfObjectStream(stream)
    try:
        assert objects.get(1).data == b"x"
    finally:
        objects.lexer.close()


def test_reader_skips_invalid_xref_generation_and_keeps_following_row() -> None:
    data = b"\x01\x00\x01\x00\x00\x01\x09\x00\x00\x00"
    stream = PdfStream({"Type": PdfName.of("XRef"), "Size": 2, "W": [1, 1, 3]}, data)
    with pytest.raises(PdfParseError):
        StrictXRefScanner.parse_stream(stream)
    entries, _ = XRefScanner.parse_stream(stream)
    assert key_for(0) not in entries
    assert entries[key_for(1)].offset == 9


def test_reader_repairs_missing_trailer_keyword_and_size() -> None:
    data = b"xref\n0 1\n0000000000 65535 f \n<< >>"
    with pytest.raises(PdfParseError):
        StrictXRefScanner.parse_table_section(data, 0)
    entries, trailer, _, _ = XRefScanner.parse_table_section(data, 0)
    assert trailer["Size"] == 1
    assert not entries[key_for(0, 65535)].in_use


def test_reader_preserves_partial_tree_entries_inheritance_and_resources() -> None:
    assert list(
        iter_number_tree_items({"Nums": [0, "ok", "bad", 2, 3]}, lambda value: value, recover=True)
    ) == [(0, "ok")]
    assert collect_inherited_values(
        {"MediaBox": [0, 0, 10, 10], "Parent": 7}, ("MediaBox",), lambda value: value
    ) == {"MediaBox": [0, 0, 10, 10]}
    resolver = ObjectResolver(b"", {}, {})
    try:
        assert resolve_resource_dict(17, resolver) is None
    finally:
        resolver.close()
    leaf = {"Type": PdfName.of("Page")}
    pages = iter_page_nodes(
        {"Type": PdfName.of("Pages"), "Kids": [17, leaf]},
        lambda value: value,
        on_invalid_child=lambda value: True,
    )
    assert [node.dictionary for node in pages] == [leaf]


def test_reader_deciphers_valid_stream_in_its_object_context() -> None:
    calls: list[bytes] = []

    def decipher(number: int, generation: int, data: bytes, dictionary: PdfDict | None) -> bytes:
        assert (number, generation) == (1, 0)
        calls.append(data)
        return data

    lexer = PdfLexer(b"1 0 obj << /Length 1 >> stream\na\nendstream\nendobj", decipher=decipher)
    try:
        assert bytes(lexer.parse_indirect_object().raw_data) == b"a"
        assert calls == [b"a"]
    finally:
        lexer.close()


@pytest.mark.parametrize(
    "data",
    [
        b"barekeyword",
        b"/A#00",
        b"0 0 obj 1 endobj",
        b"1 0 obj << /Length 1 >> stream\ra\nendstream\nendobj",
    ],
)
def test_reader_preserves_lexical_extensions(data: bytes) -> None:
    lexer = PdfLexer(data)
    try:
        result = lexer.parse_indirect_object() if b" obj " in data else lexer.parse_object()
        if data == b"barekeyword":
            assert result == "barekeyword"
        elif data == b"/A#00":
            assert str(result) == "A\x00"
        elif data.startswith(b"0"):
            assert result == 1
        else:
            assert bytes(result.raw_data) == b"a"
    finally:
        lexer.close()


def test_reader_recovers_nearby_object_stream_offset() -> None:
    stream = PdfStream({"Type": PdfName.of("ObjStm"), "N": 1, "First": 4}, b"1 5 [(xy)]")
    strict = StrictObjectStream(stream)
    reader = PdfObjectStream(stream)
    try:
        with pytest.raises(PdfParseError):
            strict.get(1)
        recovered = reader.get(1)
        assert isinstance(recovered, PdfString)
        assert recovered.data == b"xy"
        assert reader.get(PdfReference(1)) is recovered
        missing = object()
        assert reader.get(2, missing) is missing
        with pytest.raises(ValueError, match="invalid object number"):
            reader.get(-1)
    finally:
        strict.lexer.close()
        reader.lexer.close()


def test_reader_recovers_nearby_indirect_object_offset() -> None:
    from core_pdf_spec.s_07_syntax.resolver import ObjectResolver as StrictResolver
    from core_pdf_spec.s_07_syntax.xref import PdfXRefEntry
    from core_pdf_spec.types import PdfReference

    data = b"1 0 obj (value) endobj"
    entries = {key_for(1): PdfXRefEntry(3)}
    strict = StrictResolver(data, entries, {})
    reader = ObjectResolver(data, entries, {})
    try:
        with pytest.raises(PdfParseError):
            strict.resolve(PdfReference(1))
        value = reader.resolve(PdfReference(1))
        assert isinstance(value, PdfString)
        assert value.data == b"value"
    finally:
        strict.close()
        reader.close()


def test_reader_does_not_suppress_custom_tree_decoder_failure() -> None:
    def fail(value: object) -> int | None:
        raise ValueError("decoder failure")

    with pytest.raises(ValueError, match="decoder failure"):
        list(
            iter_number_tree_items(
                {"Nums": [0, 1]}, lambda value: value, decode_number=fail, recover=True
            )
        )


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("recover_entries", [False, True])
def test_reader_tree_entry_recovery_does_not_enable_node_recovery(
    recover: bool, recover_entries: bool
) -> None:
    entries = iter_number_tree_items(
        {"Nums": [0, "first", "invalid", "skip", 1, "second", 2]},
        lambda value: value,
        recover=recover,
        recover_entries=recover_entries,
    )
    if recover or recover_entries:
        assert list(entries) == [(0, "first"), (1, "second")]
    else:
        with pytest.raises(ValueError, match="Nums array"):
            list(entries)

    nodes = iter_number_tree_items(
        {"Kids": [17, {"Nums": [1, "child"]}]},
        lambda value: value,
        recover=recover,
        recover_entries=recover_entries,
    )
    if recover:
        assert list(nodes) == [(1, "child")]
    else:
        with pytest.raises(ValueError, match="tree node"):
            list(nodes)


@pytest.mark.parametrize("recover", [False, True])
def test_reader_tree_cycles_and_depth_keep_their_recovery_policy(recover: bool) -> None:
    tree: dict[str, object] = {"Nums": [0, "root"]}
    tree["Kids"] = [tree, {"Nums": [1, "child"]}]
    cycle_items = iter_number_tree_items(tree, lambda value: value, recover=recover)
    if recover:
        assert list(cycle_items) == [(0, "root"), (1, "child")]
    else:
        with pytest.raises(ValueError, match="cycle"):
            list(cycle_items)

    depth_items = iter_number_tree_items(tree, lambda value: value, recover=recover, max_depth=0)
    if recover:
        assert list(depth_items) == [(0, "root")]
    else:
        with pytest.raises(ValueError, match="depth"):
            list(depth_items)


def test_reader_name_tree_recovery_keeps_order_and_unresolved_values() -> None:
    first = PdfReference(1)
    second = PdfReference(2)

    def resolve(value: object) -> object:
        if isinstance(value, PdfReference):
            raise AssertionError("tree values must remain indirect")
        return value

    assert list(
        iter_name_tree_items(
            {"Kids": [{"Names": ["a", first, 17, "skip", "odd"]}, {"Names": ["b", second]}]},
            resolve,
            lambda value: value if isinstance(value, str) else None,
            recover_entries=True,
            resolve_values=False,
        )
    ) == [("a", first), ("b", second)]


def test_reader_tree_recovery_does_not_suppress_resolver_failure() -> None:
    def fail(value: object) -> object:
        raise ValueError("resolver failure")

    with pytest.raises(ValueError, match="resolver failure"):
        list(iter_number_tree_items({"Nums": [0, 1]}, fail, recover=True))


@pytest.mark.parametrize("recover", [False, True])
def test_reader_preserves_missing_metadata_for_malformed_catalog(recover: bool) -> None:
    resolver = ObjectResolver(b"", {}, {})
    try:
        assert resolve_metadata_stream(resolver, {"Root": 17}, recover=recover) is None
        assert resolve_metadata_stream(resolver, {}, recover=recover) is None
        stream = PdfStream({}, b"<x/>")
        assert resolve_metadata_stream(
            resolver, {"Root": {"Metadata": stream}}, recover=recover
        ) == {"tag": "x", "attributes": {}}
    finally:
        resolver.close()


def test_reader_preserves_metadata_error_policy() -> None:
    resolver = ObjectResolver(b"", {}, {})
    try:
        with pytest.raises(ValueError, match="Metadata"):
            resolve_metadata_stream(resolver, {"Root": {"Metadata": 17}})
        assert resolve_metadata_stream(resolver, {"Root": {"Metadata": 17}}, recover=True) is None
    finally:
        resolver.close()
