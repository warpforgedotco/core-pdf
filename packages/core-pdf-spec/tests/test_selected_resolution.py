import pytest

from core_pdf_spec.s_07_document.metadata import metadata_stream
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_14_structure.dictionaries import attribute_entries
from core_pdf_spec.types import PdfReference


@pytest.mark.parametrize("indirect_root", [False, True])
@pytest.mark.parametrize("indirect_metadata", [False, True])
def test_metadata_resolves_only_selected_objects_and_preserves_stream(
    indirect_root: bool, indirect_metadata: bool
) -> None:
    class Resolver(ObjectResolver):
        def __init__(self) -> None:
            super().__init__(b"", {})
            self.references: list[PdfReference] = []

        def resolve(self, ref: object) -> object:
            if isinstance(ref, PdfReference):
                self.references.append(ref)
            return super().resolve(ref)

        def missing_object(self, ref: PdfReference) -> object:
            pytest.fail(f"unrelated catalog reference resolved: {ref}")

    def decode(data: object, dictionary: object, *, parent_dictionary: object = None) -> bytes:
        assert data == b"encoded"
        return b"metadata"

    resolver = Resolver()
    stream = PdfStream(raw_data=b"encoded", decoder=decode)
    catalog: PdfDict = {
        "Pages": PdfReference(99),
        "Metadata": PdfReference(2) if indirect_metadata else stream,
    }
    resolver.objects.update({key_for(1): catalog, key_for(2): stream})
    try:
        selected = metadata_stream(
            resolver, {"Root": PdfReference(1) if indirect_root else catalog}
        )
        assert selected is stream
        assert selected.decoder is decode
        assert selected.data == b"metadata"
        assert resolver.references == (
            ([PdfReference(1)] if indirect_root else [])
            + ([PdfReference(2)] if indirect_metadata else [])
        )
    finally:
        resolver.close()


def revision_number(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def test_attribute_lookahead_resolves_each_entry_once_and_preserves_identity() -> None:
    first, second, third = {}, PdfStream(raw_data=b"attributes"), {}
    entries: list[object] = [first, second, 2, third]
    resolved: list[object] = []

    def resolve(value: object) -> object:
        resolved.append(value)
        return value

    attributes = list(attribute_entries(entries, resolve=resolve, resolve_revision=revision_number))
    assert len(resolved) == len(entries)
    assert all(actual is expected for actual, expected in zip(resolved, entries, strict=True))
    assert all(
        actual.value is expected
        for actual, expected in zip(attributes, [first, second, third], strict=True)
    )
    assert [(item.revision, item.explicit_revision) for item in attributes] == [
        (0, False),
        (2, True),
        (0, False),
    ]


def test_attribute_iteration_defers_next_attribute_after_explicit_revision() -> None:
    first = {}
    resolved: list[object] = []

    def resolve(value: object) -> object:
        resolved.append(value)
        if value == "broken":
            raise ValueError("cannot resolve attribute")
        return value

    attributes = attribute_entries(
        [first, 0, "broken"], resolve=resolve, resolve_revision=revision_number
    )
    selected = next(attributes)
    assert selected.value is first
    assert selected.revision == 0
    assert selected.explicit_revision
    assert resolved == [first, 0]
    with pytest.raises(ValueError, match="cannot resolve attribute"):
        next(attributes)
    assert resolved == [first, 0, "broken"]


@pytest.mark.parametrize("revision", [None, True, -1, "bad"])
def test_invalid_attribute_revision_fails_before_yield(revision: object) -> None:
    attributes = attribute_entries(
        [{}, revision], resolve=lambda value: value, resolve_revision=revision_number
    )
    with pytest.raises(ValueError, match="invalid structure attribute revision"):
        next(attributes)


@pytest.mark.parametrize("entry", [None, 0, "bad"])
def test_invalid_attribute_is_rejected(entry: object) -> None:
    with pytest.raises(ValueError, match="invalid structure attribute entry"):
        next(
            attribute_entries(
                [entry], resolve=lambda value: value, resolve_revision=revision_number
            )
        )


def test_empty_attribute_sequence_resolves_nothing() -> None:
    def unexpected(value: object) -> object:
        pytest.fail("empty sequence must not resolve an entry")

    assert list(attribute_entries([], resolve=unexpected, resolve_revision=revision_number)) == []
