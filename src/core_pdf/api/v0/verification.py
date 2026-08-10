"""Commit-and-verify workflows behind the v0 editor surface.

Each function takes the ENGINE editor (``editor.document`` is the engine
document being edited), commits through it, reopens the produced bytes, and
verifies the requested invariants against the fresh document.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from core_pdf import PdfDocument

from .models import (
    AccessibilityRepairVerification,
    PreservationManifest,
    RedactionVerification,
    SanitizationVerification,
    compare_fingerprints,
    compare_object_graphs,
    verify_preservation,
)


def commit_verified(
    editor: Any,
    target: str | Path | Any,
    *,
    expected_unchanged_pages: tuple[int, ...] = (),
) -> PreservationManifest:
    """Commit and verify that untouched pages were preserved byte-for-byte."""
    from .adapters import adapt_document

    before = adapt_document(editor.document).fingerprint()
    data = editor.commit(target)
    with PdfDocument.open(data) as reopened:
        after = adapt_document(reopened).fingerprint()
    return verify_preservation(
        before,
        after,
        expected_unchanged_pages=expected_unchanged_pages,
    )


def commit_redactions_verified(
    editor: Any,
    target: str | Path | Any,
    redactions: Mapping[int, Iterable[tuple[float, float, float, float]]],
    *,
    queries: Iterable[str] = (),
) -> RedactionVerification:
    """Apply redactions, commit, and verify the queries are unrecoverable."""
    from .adapters import adapt_document

    requested = tuple(dict.fromkeys(queries))
    editor.apply_redactions(redactions)
    before = adapt_document(editor.document).fingerprint()
    before_graph = adapt_document(editor.document).object_graph()
    data = editor.commit(target)
    remaining_raw = tuple(query for query in requested if query.encode("utf-8") in data)
    with PdfDocument.open(data) as reopened:
        document = adapt_document(reopened)
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
    editor: Any,
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
    """Strip the selected features, commit, and verify nothing remains."""
    from .adapters import adapt_document

    if metadata:
        editor.set_metadata({})
    current = adapt_document(editor.document)
    before_graph = current.object_graph()
    if annotations or links:
        for page in tuple(current.pages()):
            if annotations and tuple(page.annotations()):
                editor.remove_annotations(page.info.number)
            if links and tuple(page.links()):
                editor.remove_links(page.info.number)
    if forms:
        names = tuple(field.name for page in current.pages() for field in page.form_fields())
        if names:
            editor.remove_form_fields(names)
    if attachments:
        editor.set_attachments({})
    if outlines:
        editor.set_outlines(())
    data = editor.commit(target)
    with PdfDocument.open(data) as reopened:
        document = adapt_document(reopened)
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
    editor: Any,
    target: str | Path | Any,
    *,
    title: str | None = None,
    language: str | None = None,
) -> AccessibilityRepairVerification:
    """Write title/language metadata, commit, and verify the repair took."""
    from .adapters import adapt_document

    values: dict[str, object] = {}
    if title is not None:
        values["Title"] = title
    if language is not None:
        values["Lang"] = language
    editor.set_metadata(values)
    data = editor.commit(target)
    with PdfDocument.open(data) as reopened:
        inventory = adapt_document(reopened).accessibility_inventory()
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


__all__ = (
    "commit_accessibility_repair_verified",
    "commit_redactions_verified",
    "commit_sanitized_verified",
    "commit_verified",
)
