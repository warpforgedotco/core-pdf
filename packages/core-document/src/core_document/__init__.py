# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable structured document records and derived views."""

from core_document.adapters import DocumentAdapter
from core_document.editor import DocumentEditor
from core_document.model import (
    Annotation,
    BBox,
    Block,
    BlockKind,
    Diagnostic,
    Document,
    Figure,
    FormField,
    Link,
    Page,
    Table,
    TableCell,
    TextLine,
)

__all__ = (
    "BBox",
    "Annotation",
    "Block",
    "BlockKind",
    "Diagnostic",
    "DocumentAdapter",
    "DocumentEditor",
    "Document",
    "Figure",
    "FormField",
    "Link",
    "Page",
    "Table",
    "TableCell",
    "TextLine",
)
