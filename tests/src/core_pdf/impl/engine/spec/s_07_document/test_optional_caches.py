from __future__ import annotations

import threading
from typing import Any, cast

from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument
from core_pdf.impl.primitives import MISSING


class MissingResolver:
    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, value: object) -> None:
        self.calls += 1
        return None


class OptionalDocument:
    def __init__(self) -> None:
        self.internal_cache_lock = threading.RLock()
        self.resolver = MissingResolver()
        self.structure_cache = MISSING
        self.mark_info_cache = MISSING
        self.acroform_cache = MISSING
        self.xref_was_recovered = False
        self.page_tree_was_recovered = False

    def catalog(self) -> dict[object, object]:
        return {}


def test_absent_structure_and_mark_info_are_cached() -> None:
    document = OptionalDocument()
    structure = cast(Any, PdfDocument.structure)
    mark_info = cast(Any, PdfDocument.mark_info)

    assert structure.__get__(document, OptionalDocument) is None
    assert structure.__get__(document, OptionalDocument) is None
    assert document.resolver.calls == 1

    assert mark_info.__get__(document, OptionalDocument) is None
    assert mark_info.__get__(document, OptionalDocument) is None
    assert document.resolver.calls == 2


def test_absent_acroform_is_cached() -> None:
    document = OptionalDocument()
    acroform = cast(Any, PdfDocument.acroform)

    assert acroform.__get__(document, OptionalDocument) is None
    assert acroform.__get__(document, OptionalDocument) is None
    assert document.resolver.calls == 1


class PageLabelsDocument:
    def __init__(self) -> None:
        self.internal_cache_lock = threading.RLock()
        self.page_labels_cache = MISSING
        self.calls = 0

    def build_page_labels(self) -> None:
        self.calls += 1
        return None


def test_absent_page_labels_are_cached() -> None:
    document = PageLabelsDocument()
    page_labels = cast(Any, PdfDocument.page_labels)

    assert page_labels.__get__(document, PageLabelsDocument) is None
    assert page_labels.__get__(document, PageLabelsDocument) is None
    assert document.calls == 1
