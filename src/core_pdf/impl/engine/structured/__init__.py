# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable structured document records and derived views."""

from core_pdf.impl.engine.structured.adapters import DocumentAdapter
from core_pdf.impl.engine.structured.editor import DocumentEditor
from core_pdf.impl.engine.structured.model import (
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
    TextSpan,
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
    "TextSpan",
    "TextLine",
)
