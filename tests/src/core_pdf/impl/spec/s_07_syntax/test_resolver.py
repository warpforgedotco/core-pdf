from contextlib import closing

import pytest

from core_pdf.impl.spec.s_07_syntax.resolver import ObjectResolver
from core_pdf.impl.types import PdfReference, PdfString


def test_deep_resolve_terminates_on_a_cyclic_reference_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A damaged file can encode 1 0 R -> 2 0 R -> 1 0 R.

    deep_resolve used to recurse into itself for a PdfReference while only ever
    recording container ids in ``seen``, so such a chain ran to RecursionError.
    """
    with closing(ObjectResolver(b"", {}, {})) as resolver:
        monkeypatch.setattr(
            type(resolver),
            "resolve",
            lambda self, ref: (
                PdfReference(2, 0)
                if type(ref) is PdfReference and ref.object_number == 1
                else PdfReference(1, 0)
                if type(ref) is PdfReference
                else ref
            ),
        )
        assert resolver.deep_resolve(PdfReference(1, 0)) == PdfReference(1, 0)


def test_deep_resolve_preserves_shared_subgraphs_within_one_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = {"Value": PdfReference(1, 0)}
    root = [shared, shared]
    with closing(ObjectResolver(b"", {}, {})) as resolver:
        monkeypatch.setattr(
            type(resolver),
            "resolve",
            lambda self, ref: "resolved" if type(ref) is PdfReference else ref,
        )
        resolved = resolver.deep_resolve(root)

    assert isinstance(resolved, list)
    assert resolved[0] is resolved[1]
    assert resolved[0] == {"Value": "resolved"}


def test_resolve_str_does_not_expand_composite_object_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def internal_reject_deep_resolution(*args: object) -> object:
        del args
        raise AssertionError("composite value was deep-resolved")

    monkeypatch.setattr(ObjectResolver, "deep_resolve", internal_reject_deep_resolution)
    with closing(ObjectResolver(b"", {}, {})) as resolver:
        assert resolver.resolve_str([PdfReference(1), "XYZ"]) is None
        assert resolver.resolve_str(PdfString(b"https://example.invalid")) == (
            "https://example.invalid"
        )
