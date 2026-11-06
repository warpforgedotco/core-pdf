from __future__ import annotations

import datetime
import hashlib
import os
import uuid
from collections.abc import Iterable
from pathlib import Path

from ._elements import Element


def get_last_modified_date(filename: str | os.PathLike[str]) -> str | None:
    path = Path(filename)
    if not path.is_file():
        return None
    # unstructured reports local time without an offset.
    modified = datetime.datetime.fromtimestamp(path.stat().st_mtime)  # noqa: DTZ006
    return modified.strftime("%Y-%m-%dT%H:%M:%S%z")


def assign_element_ids(elements: Iterable[Element], *, unique: bool) -> None:
    # A hash of the filename, text, page number and position on the page, so a document
    # split into fragments gets the same ids. unique_element_ids asks for random ones.
    sequence_by_page: dict[object, int] = {}
    for element in elements:
        metadata = element.metadata
        if unique:
            metadata["element_id"] = str(uuid.uuid4())
            continue
        page_number = metadata.page_number
        sequence = sequence_by_page.get(page_number, 0)
        sequence_by_page[page_number] = sequence + 1
        data = f"{metadata.filename}{element.text}{page_number}{sequence}"
        metadata["element_id"] = hashlib.sha256(data.encode()).hexdigest()[:32]


# Ported from unstructured.partition.common.metadata (Apache-2.0): the categories an
# element of each category may parent.
_TITLE_CHILDREN = (
    "Text",
    "UncategorizedText",
    "NarrativeText",
    "ListItem",
    "BulletedText",
    "Table",
    "FigureCaption",
    "CheckBox",
)
HIERARCHY_RULE_SET: dict[str, frozenset[str]] = {
    "Title": frozenset(_TITLE_CHILDREN),
    "Header": frozenset(("Title", *_TITLE_CHILDREN)),
}


def assign_parent_ids(elements: Iterable[Element]) -> None:
    # Ported from unstructured's set_element_hierarchy (Apache-2.0), run after the ids are
    # assigned: an element's parent is the nearest open element whose category ranks above
    # it, or one of its own category at a shallower depth.
    stack: list[Element] = []
    for element in elements:
        metadata = element.metadata
        if metadata.parent_id is not None:
            continue
        category = element.category
        depth = metadata.category_depth or 0
        parent_id = None
        while stack:
            top = stack[-1]
            top_category = top.category
            if (top_category == category and (top.metadata.category_depth or 0) < depth) or (
                top_category != category and category in HIERARCHY_RULE_SET.get(top_category, ())
            ):
                parent_id = top.element_id
                break
            stack.pop()
        if parent_id is not None:
            metadata["parent_id"] = parent_id
        stack.append(element)
