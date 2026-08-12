"""Concrete public editor with verified commit workflows."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core_pdf import PdfDocument as EngineDocument
from core_pdf.impl.engine.document import PdfDocumentEditor as EngineEditor

from .models import (
    AccessibilityRepairVerification,
    PreservationManifest,
    RedactionVerification,
    SanitizationVerification,
    compare_fingerprints,
    compare_object_graphs,
    verify_preservation,
)


class PdfEditor(EngineEditor):
    """Engine editor extended only with public verification workflows."""

    @contextmanager
    def internal_commit_and_reopen(self, target: str | Path | Any) -> Iterator[tuple[bytes, Any]]:
        from .document import PdfDocument

        data = self.commit(target)
        with EngineDocument.open(data) as reopened:
            yield data, PdfDocument.internal_from_engine(reopened)

    def internal_document_view(self) -> Any:
        from .document import PdfDocument

        return PdfDocument.internal_from_engine(self.document)

    def commit_verified(
        self, target: str | Path | Any, *, expected_unchanged_pages: tuple[int, ...] = ()
    ) -> PreservationManifest:
        before = self.internal_document_view().fingerprint()
        with self.internal_commit_and_reopen(target) as (_data, document):
            after = document.fingerprint()
        return verify_preservation(
            before,
            after,
            expected_unchanged_pages=expected_unchanged_pages,
        )

    def commit_redactions_verified(
        self,
        target: str | Path | Any,
        redactions: Mapping[int, Iterable[tuple[float, float, float, float]]],
        *,
        queries: Iterable[str] = (),
    ) -> RedactionVerification:
        requested = tuple(dict.fromkeys(queries))
        self.apply_redactions(redactions)
        current = self.internal_document_view()
        before = current.fingerprint()
        before_graph = current.object_graph()
        with self.internal_commit_and_reopen(target) as (data, document):
            remaining_raw = tuple(query for query in requested if query.encode("utf-8") in data)
            remaining = tuple(query for query in requested if any(document.search(query)))
            after = document.fingerprint()
            graph_diff = compare_object_graphs(before_graph, document.object_graph())
        return RedactionVerification(
            requested_queries=requested,
            remaining_queries=remaining,
            remaining_raw_queries=remaining_raw,
            changed_pages=tuple(sorted(set(compare_fingerprints(before, after).changed_pages))),
            became_unreachable_objects=graph_diff.became_unreachable,
            removed_objects=graph_diff.removed_objects,
            passed=not remaining and not remaining_raw,
        )

    def commit_sanitized_verified(
        self,
        target: str | Path | Any,
        *,
        metadata: bool = True,
        annotations: bool = True,
        links: bool = True,
        forms: bool = True,
        attachments: bool = True,
        outlines: bool = True,
        actions: bool = True,
    ) -> SanitizationVerification:
        if metadata:
            self.set_metadata({})
        current = self.internal_document_view()
        before_graph = current.object_graph()
        if annotations or links:
            for page in tuple(current.pages()):
                if annotations and tuple(page.annotations()):
                    self.remove_annotations(page.info.number)
                if links and tuple(page.links()):
                    self.remove_links(page.info.number)
        if forms:
            names = tuple(field.name for page in current.pages() for field in page.form_fields())
            if names:
                self.remove_form_fields(names)
        if attachments:
            self.set_attachments({})
        if outlines:
            self.set_outlines(())
        with self.internal_commit_and_reopen(target) as (_data, document):
            annotation_inventory = document.annotation_inventory()
            remaining_annotations = annotation_inventory.annotation_count
            remaining_links = annotation_inventory.link_count
            remaining_forms = document.form_inventory().field_count
            remaining_attachments = len(tuple(document.attachments))
            remaining_outlines = bool(tuple(document.outlines))
            remaining_actions = document.action_inventory().action_count
            graph_diff = compare_object_graphs(before_graph, document.object_graph())
        passed = not (
            (annotations and remaining_annotations)
            or (links and remaining_links)
            or (forms and remaining_forms)
            or (attachments and remaining_attachments)
            or (outlines and remaining_outlines)
            or (actions and remaining_actions)
        )
        return SanitizationVerification(
            removed_annotations=annotations,
            removed_links=links,
            removed_forms=forms,
            removed_attachments=attachments,
            removed_outlines=outlines,
            removed_actions=actions,
            remaining_annotations=remaining_annotations,
            remaining_links=remaining_links,
            remaining_forms=remaining_forms,
            remaining_attachments=remaining_attachments,
            remaining_outlines=remaining_outlines,
            remaining_actions=remaining_actions,
            became_unreachable_objects=graph_diff.became_unreachable,
            removed_objects=graph_diff.removed_objects,
            passed=passed,
        )

    def commit_accessibility_repair_verified(
        self,
        target: str | Path | Any,
        *,
        title: str | None = None,
        language: str | None = None,
    ) -> AccessibilityRepairVerification:
        values: dict[str, object] = {}
        if title is not None:
            values["Title"] = title
        if language is not None:
            values["Lang"] = language
        self.set_metadata(values)
        with self.internal_commit_and_reopen(target) as (_data, document):
            inventory = document.accessibility_inventory()
        passed = (title is None or inventory.has_title) and (
            language is None or inventory.document_language == language
        )
        return AccessibilityRepairVerification(
            requested_title=title,
            requested_language=language,
            has_title=inventory.has_title,
            language=inventory.document_language,
            passed=passed,
        )


__all__ = ("PdfEditor",)
