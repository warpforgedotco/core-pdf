from __future__ import annotations

# unstructured.partition.pdf reads files with pypdf's PdfReader, and callers patch it there.
from core_pdf_compat.pypdf import PdfReader

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
    Text,
    Title,
    UncategorizedText,
)
from ._metadata import get_last_modified_date
from ._partition import (
    DEFAULT_MAX_CONTENT_STREAM_ARRAY_ENTRIES,
    DEFAULT_MAX_RAW_STREAM_BYTES,
    DEFAULT_MAX_TOTAL_ARRAY_ENTRIES,
    DEFAULT_MAX_TOTAL_STREAM_BYTES,
    DEFAULT_MIN_FILE_SIZE_BYTES,
    DEFAULT_MIN_RAW_STREAM_BYTES,
    PdfComplexity,
    partition_pdf,
    pdf_complexity,
)
from .errors import PageCountExceededError, UnprocessableEntityError

for export in (
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
    Text,
    Title,
    partition_pdf,
    pdf_complexity,
    PdfComplexity,
    get_last_modified_date,
    Address,
):
    export.__module__ = __name__


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
    "Text",
    "Title",
    "UncategorizedText",
    "DEFAULT_MAX_CONTENT_STREAM_ARRAY_ENTRIES",
    "DEFAULT_MAX_RAW_STREAM_BYTES",
    "DEFAULT_MAX_TOTAL_ARRAY_ENTRIES",
    "DEFAULT_MAX_TOTAL_STREAM_BYTES",
    "DEFAULT_MIN_FILE_SIZE_BYTES",
    "DEFAULT_MIN_RAW_STREAM_BYTES",
    "PageCountExceededError",
    "PdfComplexity",
    "PdfReader",
    "UnprocessableEntityError",
    "get_last_modified_date",
    "partition_pdf",
    "pdf_complexity",
)
