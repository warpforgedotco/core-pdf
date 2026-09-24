# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Iterator

import pytest

from core_pdf_spec.s_07_filters.decode_spec import StreamDecodeSpec
from core_pdf_spec.s_07_syntax.resolution import resolve_reference_chain
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfObject
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.types import PdfReference


@pytest.fixture
def resolver() -> Iterator[ObjectResolver]:
    resolver = ObjectResolver(b"", {})
    resolver.objects[key_for(1)] = 42
    try:
        yield resolver
    finally:
        resolver.close()


@pytest.mark.parametrize("array", [False, True])
def test_unchanged_self_cycle_retains_original_identity(
    resolver: ObjectResolver, array: bool
) -> None:
    value: dict[str, object] | list[object] = [] if array else {}
    if isinstance(value, list):
        value.append(value)
    else:
        value["Self"] = value
    assert resolver.deep_resolve(value) is value


def test_changed_mutual_cycle_reconnects_backedges_and_retains_unaffected_siblings(
    resolver: ObjectResolver,
) -> None:
    unchanged = {"Direct": [1, 2, 3]}
    first: dict[str, object] = {"Value": PdfReference(1), "Unchanged": unchanged}
    second: list[object] = [first]
    first["Second"] = second
    root = [first, first, second, unchanged]
    result = resolver.deep_resolve(root)
    assert isinstance(result, list)
    assert result is not root
    resolved_first = result[0]
    resolved_second = result[2]
    assert isinstance(resolved_first, dict)
    resolved_first = resolved_first
    assert isinstance(resolved_second, list)
    assert result[1] is resolved_first
    assert resolved_first is not first
    assert resolved_second is not second
    assert resolved_first["Second"] is resolved_second
    assert resolved_second[0] is resolved_first
    assert resolved_first["Value"] == 42
    assert resolved_first["Unchanged"] is unchanged
    assert result[3] is unchanged
    assert isinstance(first["Value"], PdfReference)
    assert first["Second"] is second
    assert second[0] is first


def test_unchanged_mutual_cycle_is_not_copied(resolver: ObjectResolver) -> None:
    first: dict[str, object] = {}
    second: list[object] = [first]
    first["Second"] = second
    assert resolver.deep_resolve(first) is first


def test_indirect_backedge_becomes_a_shared_resolved_container(resolver: ObjectResolver) -> None:
    reference = PdfReference(2)
    source = {"Self": reference, "Missing": PdfReference(99)}
    resolver.objects[key_for(2)] = source  # ty: ignore[invalid-assignment]
    result = resolver.deep_resolve(reference)
    assert isinstance(result, dict)
    result = result
    assert result is not source
    assert result["Self"] is result
    assert result["Missing"] is None
    assert source["Self"] is reference


def test_non_caching_resolver_still_forms_one_indirect_cycle() -> None:
    class FreshResolver(ObjectResolver):
        calls = 0

        def resolve(self, ref: object) -> PdfObject:
            if not isinstance(ref, PdfReference):
                return ref  # ty: ignore[invalid-return-type]
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("indirect container was resolved more than once")
            return {"Self": PdfReference(1)}

    resolver = FreshResolver(b"", {})
    try:
        result = resolver.deep_resolve([PdfReference(1), PdfReference(1)])
        assert isinstance(result, list)
        assert result[0] is result[1]
        assert isinstance(result[0], dict)
        resolved = result[0]
        assert resolved["Self"] is resolved
        assert resolver.calls == 1
    finally:
        resolver.close()


def test_shared_unchanged_container_reached_by_reference_keeps_identity(
    resolver: ObjectResolver,
) -> None:
    child = {"Direct": [1, 2]}
    resolver.objects[key_for(2)] = child  # ty: ignore[invalid-assignment]
    source = [PdfReference(2), child]
    result = resolver.deep_resolve(source)
    assert isinstance(result, list)
    assert result is not source
    assert result[0] is child
    assert result[1] is child


def test_tuple_normalization_reconnects_a_mutual_cycle(resolver: ObjectResolver) -> None:
    array: list[object] = []
    source = (array,)
    array.append(source)
    result = resolver.deep_resolve(source)
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert result[0] is not array
    assert result[0][0] is result
    assert array[0] is source
    assert resolver.deep_resolve((1, 2)) == [1, 2]


@pytest.mark.parametrize("plan_kind", ["bound", "independent", "none"])
def test_changed_cyclic_stream_preserves_source_decoder_and_decode_plan(
    resolver: ObjectResolver, plan_kind: str
) -> None:
    def decoder(
        data: bytes | memoryview, dictionary: object, *, parent_dictionary: object = None
    ) -> bytes:
        raise AssertionError("graph resolution must not decode streams")

    dictionary: dict[object, object] = {"Value": PdfReference(1)}
    independent = StreamDecodeSpec(())
    plan = (
        dictionary if plan_kind == "bound" else independent if plan_kind == "independent" else None
    )
    raw = b"encoded source"
    stream = PdfStream(dictionary, raw, plan, decoder=decoder)
    dictionary["Self"] = stream
    source = [stream, stream, dictionary]
    result = resolver.deep_resolve(source)
    assert isinstance(result, list)
    resolved_stream = result[0]
    assert isinstance(resolved_stream, PdfStream)
    assert resolved_stream is not stream
    assert result[1] is resolved_stream
    assert result[2] is resolved_stream.dictionary
    assert resolved_stream.dictionary["Self"] is resolved_stream
    assert resolved_stream.dictionary["Value"] == 42
    assert resolved_stream.raw_data is raw
    assert resolved_stream.decoder is decoder
    assert resolved_stream.spec is (resolved_stream.dictionary if plan_kind == "bound" else plan)
    assert dictionary["Self"] is stream
    assert isinstance(dictionary["Value"], PdfReference)


def test_unchanged_cyclic_stream_keeps_identity(resolver: ObjectResolver) -> None:
    dictionary: dict[object, object] = {}
    stream = PdfStream(dictionary, b"source", dictionary)
    dictionary["Self"] = stream
    assert resolver.deep_resolve(stream) is stream


def test_reference_only_cycles_keep_their_own_repeated_reference(resolver: ObjectResolver) -> None:
    first, second = PdfReference(2), PdfReference(3)
    resolver.objects.update({key_for(2): second, key_for(3): first})
    source = [first, second]
    assert resolver.deep_resolve(first) is first
    assert resolver.deep_resolve(second) is second
    assert resolver.deep_resolve(source) is source


def test_deep_graph_uses_iterative_resolution(resolver: ObjectResolver) -> None:
    source: object = PdfReference(1)
    for _ in range(2000):
        source = [source]
    result = resolver.deep_resolve(source)
    for _ in range(2000):
        assert isinstance(result, list)
        result = result[0]
    assert result == 42


def test_failed_resolution_does_not_modify_partially_discovered_graph() -> None:
    class FailingResolver(ObjectResolver):
        def resolve(self, ref: object) -> PdfObject:
            if ref == PdfReference(3):
                raise ValueError("resolution failed")
            return super().resolve(ref)

    resolver = FailingResolver(b"", {})
    source: dict[str, object] = {"Value": PdfReference(1), "Failure": PdfReference(3)}
    source["Self"] = source
    resolver.objects[key_for(1)] = [42]
    try:
        with pytest.raises(ValueError, match="resolution failed"):
            resolver.deep_resolve(source)
        assert source["Value"] == PdfReference(1)
        assert source["Failure"] == PdfReference(3)
        assert source["Self"] is source
    finally:
        resolver.close()


def test_scalar_chain_does_not_traverse_terminal_containers(resolver: ObjectResolver) -> None:
    terminal = {"Unused": PdfReference(99)}
    resolver.objects.update({key_for(2): PdfReference(3), key_for(3): terminal})  # ty: ignore[no-matching-overload]
    assert resolve_reference_chain(PdfReference(2), resolver.resolve) is terminal
    assert key_for(99) not in resolver.objects
    assert resolve_reference_chain(terminal, resolver.resolve) is terminal


def test_stream_decode_keys_share_one_resolution_graph(resolver: ObjectResolver) -> None:
    shared = {"Value": PdfReference(1)}
    stream = PdfStream({"DecodeParms": shared, "FDecodeParms": shared, "Unused": PdfReference(99)})
    result = resolver.resolve_stream(stream)
    assert result.dictionary["DecodeParms"] is result.dictionary["FDecodeParms"]
    assert result.dictionary["DecodeParms"] == {"Value": 42}
    assert result.dictionary["Unused"] is stream.dictionary["Unused"]
    assert key_for(99) not in resolver.objects
