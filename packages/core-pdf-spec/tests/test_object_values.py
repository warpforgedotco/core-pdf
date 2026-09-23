# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import zlib
from collections.abc import Iterator
from typing import Any

import pytest

from core_pdf_spec.s_07_document.page import iter_page_nodes
from core_pdf_spec.s_07_filters.decode_spec import FilterStep, StreamDecodeSpec
from core_pdf_spec.s_07_syntax.inherited_values import (
    collect_inherited_values,
    inherited_dictionary_value,
)
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.resources import resolve_resource_dict
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfObject
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    decoded_name,
    parse_box,
    parse_float,
    parse_int,
    require_pdf_integer,
    require_pdf_number,
    require_pdf_number_array,
)
from core_pdf_spec.types import PdfName, PdfReference


@pytest.fixture
def resolver() -> Iterator[ObjectResolver]:
    instance = ObjectResolver(b"", {})
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("value", [True, False, "12", b"12", float("inf"), float("nan"), 10**400])
def test_number_objects_reject_coercion_and_unrepresentable_values(value: object) -> None:
    with pytest.raises(ValueError, match="expected PDF number"):
        require_pdf_number(value)
    with pytest.raises(ValueError, match="expected PDF number array"):
        require_pdf_number_array([value])
    assert parse_box([0, 0, value, 100]) is None


@pytest.mark.parametrize("value", [True, False, 12.0, "12", b"12", None])
def test_integer_objects_reject_nonintegers(value: object) -> None:
    with pytest.raises(ValueError, match="expected PDF integer"):
        require_pdf_integer(value)


def test_object_validation_and_lexical_conversion_have_distinct_inputs() -> None:
    assert require_pdf_number_array([1, -2.5, 0]) == (1.0, -2.5, 0.0)
    assert require_pdf_number_array(()) == ()
    assert require_pdf_integer(10**400) == 10**400
    assert parse_box([0, -2.5, 100, 200]) == (0.0, -2.5, 100.0, 200.0)
    assert parse_int(b"+12") == 12
    assert parse_float(b"-.5") == -0.5


@pytest.mark.parametrize("value", ["12", b"12", True, float("inf")])
def test_resolver_validates_numeric_objects(resolver: ObjectResolver, value: object) -> None:
    with pytest.raises(ValueError):
        resolver.resolve_float(value)
    assert resolver.resolve_float(None, 7.0) == 7.0
    assert resolver.resolve_int(None, 7) == 7


@pytest.mark.parametrize(
    ("source", "name"), [(b"/#2FFoo", "/Foo"), (b"/Foo", "Foo"), (b"/", ""), (b"/#23#E9", "#é")]
)
def test_decoded_names_preserve_escaped_characters(
    resolver: ObjectResolver, source: bytes, name: str
) -> None:
    lexer = PdfLexer(source)
    try:
        value = lexer.parse_object()
    finally:
        lexer.close()
    assert isinstance(value, PdfName)
    assert decoded_name(value) == name
    assert decoded_name(name) == name
    assert decoded_name(name.encode("latin-1")) == name
    resolver.objects[key_for(1)] = value
    assert resolver.resolve_name(value) == name
    assert resolver.resolve_name(PdfReference(1)) == name


def test_escaped_slash_resource_names_do_not_collide(resolver: ObjectResolver) -> None:
    lexer = PdfLexer(b"<< /Foo 1 /#2FFoo 2 >>")
    try:
        resources = lexer.parse_object()
    finally:
        lexer.close()
    resolver.objects[key_for(1)] = resources
    for value in (resources, PdfReference(1)):
        dictionary = resolve_resource_dict(value, resolver)
        assert dictionary is not None
        assert dictionary["Foo"] == 1
        assert dictionary["/Foo"] == 2


@pytest.mark.parametrize("value", [None, PdfReference(1), PdfReference(999)])
def test_null_values_inherit_from_parent(resolver: ObjectResolver, value: PdfObject) -> None:
    resolver.objects[key_for(1)] = None
    parent = [0, 0, 100, 100]
    assert inherited_dictionary_value({"V": value}, "V", parent, resolver.resolve) is parent
    node: PdfDict = {"MediaBox": value, "Parent": {"MediaBox": parent}}
    assert collect_inherited_values(node, ("MediaBox",), resolver.resolve)["MediaBox"] is parent
    page = {"Type": PdfName.of("Page"), "MediaBox": value}
    root = {"Type": PdfName.of("Pages"), "MediaBox": parent, "Kids": [page]}
    assert next(iter_page_nodes(root, resolver.resolve)).inherited_values["MediaBox"] is parent


@pytest.mark.parametrize("value", [0, False, [], {}, PdfReference(2)])
def test_nonnull_inherited_values_preserve_identity(
    resolver: ObjectResolver, value: PdfObject
) -> None:
    resolver.objects[key_for(2)] = {"Unused": PdfReference(999)}
    assert inherited_dictionary_value({"V": value}, "V", "parent", resolver.resolve) is value


def test_inheritance_does_not_resolve_shadowed_or_nested_values(resolver: ObjectResolver) -> None:
    selected = PdfReference(1)
    resources: PdfDict = {"Unused": PdfReference(999)}
    resolver.objects[key_for(1)] = resources
    visited: list[object] = []

    def resolve(value: object) -> object:
        visited.append(value)
        if value == PdfReference(999):
            raise AssertionError("unused resource was resolved")
        return resolver.resolve(value)

    parent = {"Resources": PdfReference(999)}
    node: PdfDict = {"Resources": selected, "Parent": parent}
    values = collect_inherited_values(node, ("Resources",), resolve)
    assert values["Resources"] is selected
    assert visited == [selected, parent]


def test_page_traversal_resolves_inheritance_only_after_leaf_selection() -> None:
    def resolve(value: object) -> object:
        if isinstance(value, PdfReference):
            raise ValueError("damaged resource reference")
        return value

    leaf = {"Type": PdfName.of("Page"), "Resources": {"Unused": PdfReference(100)}}
    root = {"Type": PdfName.of("Pages"), "Resources": PdfReference(99), "Kids": [leaf]}
    page = next(iter_page_nodes(root, resolve))
    assert page.inherited_values["Resources"] is leaf["Resources"]
    del leaf["Resources"]
    with pytest.raises(ValueError, match="damaged resource reference"):
        next(iter_page_nodes(root, resolve))


def test_resolving_a_decoded_stream_dictionary_does_not_decode_again(
    resolver: ObjectResolver,
) -> None:
    stream = PdfStream(
        {"Filter": PdfName.of("FlateDecode"), "Unused": PdfReference(999)},
        raw_data=b"already decoded",
        spec=None,
    )
    resolved = resolver.deep_resolve(stream)
    assert isinstance(resolved, PdfStream)
    assert resolved.dictionary is not stream.dictionary
    assert resolved.spec is None
    assert resolved.data == b"already decoded"


def test_stream_copy_preserves_independent_and_dictionary_bound_plans() -> None:
    original = {"Filter": PdfName.of("FlateDecode")}
    replacement: PdfDict = {"Filter": PdfName.of("ASCIIHexDecode")}
    bound = PdfStream(original, zlib.compress(b"x"), original)
    assert bound.data == b"x"
    replaced = bound.replace(dictionary=replacement, raw_data=b"78>")
    assert replaced.spec is replacement
    assert replaced.data == b"x"
    plan = StreamDecodeSpec((FilterStep("FlateDecode"),))
    independent = bound.replace(spec=plan)
    assert independent.replace(dictionary=replacement).spec is plan
    assert independent.replace(dictionary=replacement).data == b"x"
    assert bound.replace(dictionary=replacement, raw_data=b"plain", spec=None).data == b"plain"


def test_stream_copy_is_lazy_and_preserves_custom_decoder() -> None:
    def decoder(
        data: bytes | memoryview, dictionary: object, *, parent_dictionary: object = None
    ) -> bytes:
        raise RuntimeError("decoder invoked")

    stream = PdfStream(raw_data=b"source", decoder=decoder)
    copied = stream.replace(dictionary={"Length": 6})
    assert copied.decoder is decoder
    assert copied.raw_data is stream.raw_data
    with pytest.raises(RuntimeError, match="decoder invoked"):
        assert copied.data == b"source"
    assert copied.replace(decoder=None).data == b"source"
    unknown: dict[str, Any] = {"typo": b"replacement"}
    with pytest.raises(TypeError, match="unexpected keyword"):
        stream.replace(**unknown)
