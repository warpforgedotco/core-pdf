"""Scalar resolution, reference sharing, and page inheritance contracts."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from core_pdf_spec.s_07_document.page import iter_page_nodes
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.types import CachedPdfObject
from core_pdf_spec.s_07_syntax.xref import PdfXRefEntry, key_for
from core_pdf_spec.types import PdfReference, PdfString


@pytest.mark.parametrize("default", [None, 7.0])
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (42, 42.0),
        (-2.5, -2.5),
        (b"1.5", None),
        (b"1e2", None),
        (float("inf"), None),
        (float("-inf"), None),
        (float("nan"), None),
        (10**400, None),
        (True, None),
        (None, None),
    ],
)
def test_direct_and_indirect_real_conversion_agree(
    value: CachedPdfObject, expected: float | None, default: float | None
) -> None:
    resolver = ObjectResolver(b"", {})
    resolver.objects[key_for(1)] = value
    try:
        for candidate in (value, PdfReference(1)):
            if value is not None and expected is None:
                with pytest.raises(ValueError, match="expected PDF number"):
                    resolver.resolve_float(candidate, default)
            else:
                result = default if expected is None else expected
                assert resolver.resolve_float(candidate, default) == result
    finally:
        resolver.close()


def test_reference_chains_preserve_cycles_and_shared_containers() -> None:
    resolver = ObjectResolver(b"", {})
    text = PdfString(b"text")
    first, second = PdfReference(1), PdfReference(2)
    cycle_first, cycle_second = PdfReference(3), PdfReference(4)
    resolver.objects.update(
        {key_for(1): second, key_for(2): text, key_for(3): cycle_second, key_for(4): cycle_first}
    )
    try:
        assert resolver.resolve_str(first) == "text"
        assert resolver.deep_resolve(first) is text
        assert resolver.deep_resolve(cycle_first) is cycle_first
        assert resolver.resolve_str(cycle_first) is None
        assert resolver.deep_resolve(PdfReference(99)) is None

        untouched = {"key": text}
        assert resolver.deep_resolve(untouched) is untouched
        shared = {"key": first}
        resolved = resolver.deep_resolve([shared, shared])
        assert isinstance(resolved, list)
        assert resolved == [{"key": text}, {"key": text}]
        assert resolved[0] is resolved[1]
        assert resolved[0] is not shared
        assert shared["key"] is first

        # A destination array is not a string; its page graph must remain untouched.
        destination = [PdfReference(100), "XYZ", 0, 0, 1]
        assert resolver.resolve_str(destination) is None
        assert key_for(100) not in resolver.objects
    finally:
        resolver.close()


def test_resolver_caches_undefined_reference_as_null() -> None:
    class Resolver(ObjectResolver):
        calls = 0

        def missing_object(self, ref: PdfReference) -> object:
            self.calls += 1
            return None

    resolver = Resolver(b"", {})
    try:
        assert resolver.resolve(PdfReference(1)) is None
        assert resolver.resolve(PdfReference(1)) is None
        assert resolver.calls == 1
    finally:
        resolver.close()


def test_concurrent_resolution_returns_one_cached_object_identity() -> None:
    barrier = Barrier(2, timeout=5)

    class Resolver(ObjectResolver):
        def load_indirect_object(self, lexer: PdfLexer, offset: int) -> object:
            value = {"text": "shared"}
            barrier.wait()
            return value

    resolver = Resolver(b"", {key_for(1): PdfXRefEntry(0)})
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            values = list(executor.map(resolver.resolve, [PdfReference(1), PdfReference(1)]))
        assert values[0] is values[1]
        assert resolver.resolve(PdfReference(1)) is values[0]
    finally:
        resolver.close()


def test_page_inheritance_preserves_null_defaults_identity_and_sibling_isolation() -> None:
    parent_resources, child_resources = PdfReference(1), PdfReference(2)
    first = {"Type": "Page", "Resources": None, "Custom": "child"}
    second = {"Type": "Page", "Resources": child_resources}
    root = {
        "Type": "Pages",
        "Resources": parent_resources,
        "Custom": "parent",
        "Kids": [first, second],
    }
    pages = list(iter_page_nodes(root, lambda value: value, inherited_keys=("Resources", "Custom")))
    assert [page.dictionary for page in pages] == [first, second]
    assert pages[0].inherited_values == {"Resources": parent_resources, "Custom": "child"}
    assert pages[1].inherited_values == {"Resources": child_resources, "Custom": "parent"}
    assert pages[0].inherited_values["Resources"] is parent_resources
    assert pages[1].inherited_values["Resources"] is child_resources
    assert root["Custom"] == "parent"
