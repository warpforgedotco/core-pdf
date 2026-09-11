"""Unstructured-style APIs requiring spaCy and the English en_core_web_sm model."""

from __future__ import annotations

from typing import Any, TypeAlias

# Enforce the NLP requirement before any element submodule can be cached.
from . import _classification as _classification
from ._elements import (
    Address,
    Element,
    ElementMetadata,
    EmailAddress,
    Footer,
    Header,
    Image,
    ListItem,
    NarrativeText,
    PageBreak,
    Table,
    Title,
    UncategorizedText,
)
from ._partition import (
    partition_pdf,
)

PdfInput: TypeAlias = Any

# Preserve the historical public identity for imports, reprs, and pickling.
for internal_export in (
    Element,
    ElementMetadata,
    EmailAddress,
    Footer,
    Header,
    Image,
    ListItem,
    NarrativeText,
    PageBreak,
    Table,
    Title,
    UncategorizedText,
    partition_pdf,
    Address,
):
    internal_export.__module__ = __name__


__all__ = (
    "Element",
    "ElementMetadata",
    "EmailAddress",
    "Footer",
    "Header",
    "Image",
    "ListItem",
    "NarrativeText",
    "PageBreak",
    "Table",
    "Title",
    "UncategorizedText",
    "partition_pdf",
)
