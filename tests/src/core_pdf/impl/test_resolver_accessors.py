from collections.abc import Iterator

import pytest

from core_pdf.impl.recovery_resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.types import PdfName, PdfReference


class Subdict(dict):
    pass


@pytest.fixture
def resolver() -> Iterator[ObjectResolver]:
    resolver = ObjectResolver(b"", {})
    resolver.objects[key_for(1)] = Subdict(Type=PdfName(b"Page"))
    resolver.objects[key_for(2)] = [1, 2]
    resolver.objects[key_for(3)] = "12"
    yield resolver
    resolver.close()


def test_dictionary_accessors_resolve_one_level_and_keep_subclasses(resolver) -> None:
    container = {"A": PdfReference(1), "B": PdfReference(2), "C": {}}
    assert type(resolver.dict_at(container, "A")) is Subdict
    assert resolver.dict_at(container, "B") is None
    assert resolver.dict_at(container, "missing") is None
    assert resolver.as_dict(PdfReference(1)) is resolver.dict_at(container, "A")
    assert resolver.as_dict(container["C"]) is container["C"]


def test_array_accessor_accepts_only_arrays(resolver) -> None:
    assert resolver.array_at({"K": PdfReference(2)}, "K") == [1, 2]
    assert resolver.array_at({"K": (1, 2)}, "K") is None
    assert resolver.array_at({"K": PdfReference(1)}, "K") is None


def test_number_and_name_accessors_are_lenient(resolver) -> None:
    container = {"N": PdfReference(3), "F": "1.5", "S": PdfName(b"Link"), "X": object()}
    assert resolver.int_at(container, "N") == 12
    assert resolver.int_at(container, "X") is None
    assert resolver.int_at(container, "X", 4) == 4
    assert resolver.float_at(container, "F") == 1.5
    assert resolver.float_at(container, "X") == 0.0
    assert resolver.float_at(container, "X", None) is None
    assert resolver.name_at(container, "S") == "Link"
    assert resolver.name_at(container, "missing") is None
