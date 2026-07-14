# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Sized
from typing import Protocol

from core_pdf.impl.engine.spec.s_07_document.document_labels import format_page_label
from core_pdf.impl.engine.spec.s_07_document.name_trees import iter_number_tree_items
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.types import PdfDict


class DocumentPageLabelsHost(Protocol):
    page_labels_cache: list[str] | None
    xref_was_recovered: bool
    page_tree_was_recovered: bool

    @property
    def pages(self) -> Sized: ...

    @property
    def page_labels(self) -> list[str] | None: ...

    def catalog(self) -> PdfDict: ...
    def resolve(self, ref: object) -> object: ...
    def build_page_labels(self) -> list[str] | None: ...


class DocumentPageLabelsMixin:
    @property
    def page_labels(self: DocumentPageLabelsHost) -> list[str] | None:
        if self.page_labels_cache is None:
            self.page_labels_cache = self.build_page_labels()
        return self.page_labels_cache

    def page_label(self: DocumentPageLabelsHost, page_index: int) -> str | None:
        labels = self.page_labels
        if labels is None or page_index < 0 or page_index >= len(labels):
            return None
        return labels[page_index]

    def build_page_labels(self: DocumentPageLabelsHost) -> list[str] | None:
        try:
            labels_root = self.resolve(lookup_dict_key(self.catalog(), "PageLabels"))
        except ValueError:
            if self.xref_was_recovered or self.page_tree_was_recovered:
                return None
            raise
        if labels_root is None:
            return None
        if not isinstance(labels_root, dict):
            raise ValueError("invalid PageLabels number tree")

        specs = [
            (page_index, spec)
            for page_index, spec in iter_number_tree_items(
                labels_root,
                self.resolve,
                recover=self.xref_was_recovered or self.page_tree_was_recovered,
            )
            if isinstance(spec, dict)
        ]
        if not specs:
            return None
        specs.sort(key=lambda item: item[0])

        page_count = len(self.pages)
        labels: list[str] = []
        spec_pos = 0
        current_index, current_spec = specs[0]
        for page_index in range(page_count):
            while spec_pos + 1 < len(specs) and page_index >= specs[spec_pos + 1][0]:
                spec_pos += 1
                current_index, current_spec = specs[spec_pos]
            labels.append(format_page_label(current_spec, page_index - current_index, self.resolve))
        return labels


__all__ = ("DocumentPageLabelsMixin",)
