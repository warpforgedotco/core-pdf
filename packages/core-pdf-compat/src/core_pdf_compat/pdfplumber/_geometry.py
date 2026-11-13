# Parts ported from pdfplumber, Copyright (c) 2015 Jeremy Singer-Vine, under the MIT License.
"""What pdfplumber 0.11.10's PDF.pages raises on a file whose page boxes it cannot read.

PDF.pages wraps an error of pdfminer's PDFPage.create_pages in PdfminerException, then
builds each Page from the entries pdfminer read: Page.__init__ resolves /Rotate and the page
boxes with resolve_all, which follows references until Python's stack overflows on a cycle
(raised as MalformedPDFException), and _normalize_box raises MalformedPDFException, naming
the box as pdfminer parsed it, when a coordinate is no number. The facade reads its boxes
from core-pdf, which parses damaged files differently, so where its own reading fails the
pages are read again here, through pdfminer's parsing, for pdfplumber's exception.
"""

from __future__ import annotations

import numbers
from collections.abc import Collection
from typing import Any

from core_pdf.impl.base.types import PdfByteBuffer

from ..pdfminer._parsing import (
    MESSAGE,
    Document,
    Loop,
    PDFObjRef,
    PSLiteral,
    Unfollowed,
    create_pages,
    open_document,
)
from .exceptions import MalformedPDFException, PdfminerException

BOXES = ("CropBox", "TrimBox", "BleedBox", "ArtBox")


def dict_type(value: object) -> object:
    """pdfplumber's get_dict_type, for the names it compares with "Page" and "Annot"."""
    if not isinstance(value, dict):
        return None
    kind = value.get("Type")
    if isinstance(kind, PSLiteral):
        return kind.name if isinstance(kind.name, str) else None
    return kind


def resolve_all(value: object, active: frozenset[Any] = frozenset()) -> Any:
    """pdfplumber's resolve_all, which recurses without end on a cycle of references."""
    if isinstance(value, PDFObjRef):
        resolved = value.resolve()
        if dict_type(resolved) == "Page":
            return value
        if value.objid in active:
            raise MalformedPDFException(RecursionError(MESSAGE))
        return resolve_all(resolved, active | {value.objid})
    if isinstance(value, (list, tuple)):
        return type(value)(resolve_all(item, active) for item in value)
    if isinstance(value, dict):
        exceptions = ["Parent"] if dict_type(value) == "Annot" else []
        return {k: v if k in exceptions else resolve_all(v, active) for k, v in value.items()}
    return value


def normalize_box(box_raw: Any, rotation: Any) -> tuple[Any, ...]:
    """pdfplumber's _normalize_box."""
    if not all(isinstance(x, numbers.Number) for x in box_raw):
        raise MalformedPDFException(f"Bounding box contains non-number coordinate(s): {box_raw}")
    x0, x1 = sorted((box_raw[0], box_raw[2]))
    y0, y1 = sorted((box_raw[1], box_raw[3]))
    return (y0, x0, y1, x1) if rotation in [90, 270] else (x0, y0, x1, y1)


def build_page(entries: dict[str, Any]) -> None:
    """Page.__init__, as far as it reads the page's rotation and boxes."""

    def get_attr(key: str, default: Any = None) -> Any:
        value = resolve_all(entries.get(key))
        return default if value is None else value

    rotation = get_attr("Rotate", 0) % 360
    normalize_box(get_attr("MediaBox"), rotation)
    for name in BOXES:
        if name in entries:
            normalize_box(get_attr(name), rotation)


def pdfplumber_pages_error(
    data: PdfByteBuffer, selected: Collection[int] | None
) -> BaseException | None:
    """The exception pdfplumber's PDF.pages raises on this file, or None if it raises none.

    None also when pdfminer's parsing goes where the port does not follow it.
    """
    opened = open_document(data)
    if not isinstance(opened, Document):
        return None
    pages = create_pages(opened)
    number = 0
    try:
        while True:
            try:
                entries = next(pages)
            except StopIteration:
                return None
            except Loop:
                return PdfminerException(RecursionError(MESSAGE))
            except Unfollowed, RecursionError:
                return None
            except Exception as error:  # noqa: BLE001  # what pdfminer raises
                return PdfminerException(error)
            number += 1
            if selected is not None and number not in selected:
                continue
            build_page(entries)
    except Unfollowed, Loop, RecursionError:
        return None
    except Exception as error:  # noqa: BLE001  # what pdfplumber raises
        return error


__all__ = ("pdfplumber_pages_error",)
