"""Public promotion of the engine's structured document IR.

The structured records returned by ``PdfDocumentAdapter.structured_document`` and
``PdfPageAdapter.structured_view`` ARE the engine IR types.  This module re-exports
them at an api-sanctioned path so callers can name them without importing
``core_pdf.impl`` directly.
"""

from core_pdf.impl.engine.structured import (
    SCHEMA_VERSION,
    Annotation,
    BBox,
    Block,
    BlockKind,
    ContentNode,
    Diagnostic,
    Document,
    DocumentTableView,
    DocumentTextView,
    Figure,
    FormField,
    Link,
    Page,
    Table,
    TableAssociatedText,
    TableCell,
    TableColumnBand,
    TableReference,
    TableRowBand,
    TableView,
    TextDiagnostics,
    TextLine,
    TextLineReference,
    TextRun,
    TextSpan,
    TextView,
    TextWord,
)

__all__ = (
    "SCHEMA_VERSION",
    "Annotation",
    "BBox",
    "Block",
    "BlockKind",
    "ContentNode",
    "Diagnostic",
    "Document",
    "DocumentTableView",
    "DocumentTextView",
    "Figure",
    "FormField",
    "Link",
    "Page",
    "Table",
    "TableAssociatedText",
    "TableCell",
    "TableColumnBand",
    "TableReference",
    "TableRowBand",
    "TableView",
    "TextDiagnostics",
    "TextLine",
    "TextLineReference",
    "TextRun",
    "TextSpan",
    "TextView",
    "TextWord",
)
