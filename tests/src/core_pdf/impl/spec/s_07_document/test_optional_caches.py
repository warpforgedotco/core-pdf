from __future__ import annotations

from typing import Any, cast

from core_pdf.impl.spec.s_07_document.document import PdfDocument


class MissingResolver:
    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, value: object) -> None:
        self.calls += 1
        return None


class OptionalDocument:
    # The properties under test read a catalog entry through
    # PdfDocument.internal_catalog_dict, so the stand-in borrows it the same
    # way the tests below borrow the properties themselves.
    internal_catalog_dict = PdfDocument.internal_catalog_dict
    recovery_enabled = False

    def __init__(self) -> None:
        self.resolver = MissingResolver()
        self.xref_was_recovered = False
        self.page_tree_was_recovered = False

    def catalog(self) -> dict[object, object]:
        return {}


def test_absent_structure_and_mark_info_are_derived_directly() -> None:
    document = OptionalDocument()
    structure = cast(Any, PdfDocument.structure)
    mark_info = cast(Any, PdfDocument.mark_info)

    assert structure.__get__(document, OptionalDocument) is None
    assert structure.__get__(document, OptionalDocument) is None
    assert document.resolver.calls == 2

    assert mark_info.__get__(document, OptionalDocument) is None
    assert mark_info.__get__(document, OptionalDocument) is None
    assert document.resolver.calls == 4


def test_absent_acroform_is_derived_directly() -> None:
    document = OptionalDocument()
    acroform = cast(Any, PdfDocument.acroform)

    assert acroform.__get__(document, OptionalDocument) is None
    assert acroform.__get__(document, OptionalDocument) is None
    assert document.resolver.calls == 2


class PageLabelsDocument:
    def __init__(self) -> None:
        self.calls = 0

    def build_page_labels(self) -> None:
        self.calls += 1
        return None


def test_absent_page_labels_are_derived_directly() -> None:
    document = PageLabelsDocument()
    page_labels = cast(Any, PdfDocument.page_labels)

    assert page_labels.__get__(document, PageLabelsDocument) is None
    assert page_labels.__get__(document, PageLabelsDocument) is None
    assert document.calls == 2
