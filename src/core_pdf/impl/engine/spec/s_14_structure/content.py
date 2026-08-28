# SPDX-License-Identifier: AGPL-3.0-only
"""Native structure-content reference value objects."""

from __future__ import annotations

from typing import Any

from core_pdf.impl.engine.spec.s_07_syntax.types import PdfDict, PdfObject


class StructureContentItem:
    """Marked-content reference from a structure element K entry."""

    __slots__ = ("page_index", "mcid", "stream")

    page_index: int | None
    mcid: int
    stream: PdfObject

    def __init__(self, page_index: int | None, mcid: int, stream: Any = None) -> None:
        if page_index is not None and type(page_index) is not int:
            raise ValueError("invalid structure content page index")
        if type(page_index) is int and page_index < 0:
            raise ValueError("invalid structure content page index")
        if type(mcid) is not int:
            raise ValueError("invalid structure content mcid")
        if mcid < 0:
            raise ValueError("invalid structure content mcid")
        self.page_index = page_index
        self.mcid = mcid
        self.stream = stream


class StructureContentObject:
    """Object reference content item from a structure element K entry."""

    __slots__ = ("page_index", "props")

    page_index: int | None
    props: PdfDict

    def __init__(self, page_index: int | None, props: PdfDict) -> None:
        if page_index is not None and type(page_index) is not int:
            raise ValueError("invalid structure content page index")
        if type(page_index) is int and page_index < 0:
            raise ValueError("invalid structure content page index")
        if not isinstance(props, dict):
            raise ValueError("invalid structure content props")
        self.page_index = page_index
        self.props = props
