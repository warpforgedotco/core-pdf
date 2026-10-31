from __future__ import annotations

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
from ._metadata import get_last_modified_date
from ._partition import (
    partition_pdf,
)

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
    Title,
    UncategorizedText,
    partition_pdf,
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
    "Title",
    "UncategorizedText",
    "get_last_modified_date",
    "partition_pdf",
)
