from __future__ import annotations

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
