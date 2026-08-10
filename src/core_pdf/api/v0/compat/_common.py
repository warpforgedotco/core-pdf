"""Shared compat kernel: opening, writing, geometry, and the structured-IR bridge.

This module is the single sanctioned ``core_pdf.impl`` import site for the
compatibility facades.  Facade packages import engine IR types, serializers,
and cross-cutting plumbing from here (or from ``core_pdf.api.v0.structured``)
instead of drilling into ``core_pdf.impl`` directly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from io import BytesIO
from operator import itemgetter
from os import PathLike
from pathlib import Path
from typing import IO, Any, Self, TypeVar, cast

from core_pdf import PdfDocument
from core_pdf.api.v0.protocols import PdfInput
from core_pdf.api.v0.structured import (
    Annotation,
    Block,
    BlockKind,
    Document,
    Figure,
    FormField,
    Link,
    Page,
    TextLine,
)
from core_pdf.impl.engine.layout.geometry import (
    RectTuple,
    bbox_union,
    flip_rect_vertical,
    rect_tuple,
)
from core_pdf.impl.engine.writing.encryption import StandardPdfEncryption
from core_pdf.impl.engine.writing.semantic import serialize_document_to_pdf
from core_pdf.impl.exceptions import PdfUnsupportedError

WriteTarget = str | PathLike[str] | IO[bytes]

_T = TypeVar("_T")


def open_source(source: PdfInput, *, password: str = "") -> PdfDocument:
    """Open any supported source shape through one normalization path.

    bytes/bytearray/memoryview become an owned ``BytesIO``; objects exposing
    ``read()`` are drained into one; paths and readers pass through.  The
    password is always forwarded to the engine.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        return PdfDocument.open(BytesIO(bytes(source)), password=password)
    if not isinstance(source, (str, PathLike)) and hasattr(source, "read"):
        return PdfDocument.open(BytesIO(cast(Any, source).read()), password=password)
    return PdfDocument.open(source, password=password)


def write_bytes(target: WriteTarget, data: bytes) -> None:
    """Write ``data`` to a filesystem path or a binary stream."""
    if isinstance(target, (str, PathLike)):
        Path(cast("str | PathLike[str]", target)).write_bytes(data)
    else:
        target.write(data)


class ClosingMixin:
    """Context-manager lifecycle for facades whose exit simply closes."""

    def close(self) -> None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def flip_box(box: object, page_height: float) -> RectTuple:
    """Convert a rectangle between bottom-left- and top-left-origin coordinates."""
    return flip_rect_vertical(cast(Sequence[float], box), page_height)


def coerce_bbox(value: object) -> RectTuple:
    """Coerce a 4-item box (tuple/list or x0..y1 object) to a float tuple."""
    rect = rect_tuple(value)
    if rect is None:
        raise ValueError(f"value does not describe a rectangle: {value!r}")
    return rect


def synthesize_characters(text: str, box: Sequence[float]) -> Iterator[tuple[str, RectTuple]]:
    """Yield per-character sub-rectangles by dividing ``box`` evenly over ``text``."""
    x0, y0, x1, y1 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    width = (x1 - x0) / max(1, len(text))
    for index, character in enumerate(text):
        yield character, (x0 + index * width, y0, x0 + (index + 1) * width, y1)


def cluster_by(
    values: Iterable[_T], key: Callable[[_T], Any] | str, tolerance: float = 0
) -> list[list[_T]]:
    """Group sorted items whose consecutive keys stay within ``tolerance``.

    Non-numeric keys group on equality.  This is the low-level clustering
    mechanic behind pdfplumber's public ``utils.cluster_list``/``cluster_objects``.
    """
    if isinstance(key, str):
        getter: Callable[[_T], Any] = cast("Callable[[_T], Any]", itemgetter(key))
    else:
        getter = key
    groups: list[list[_T]] = []
    for item in sorted(values, key=getter):
        current = getter(item)
        previous = getter(groups[-1][-1]) if groups else None
        separated = bool(groups) and (
            current - previous > tolerance
            if isinstance(current, (int, float)) and isinstance(previous, (int, float))
            else current != previous
        )
        if not groups or separated:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


__all__ = (
    "Annotation",
    "Block",
    "BlockKind",
    "ClosingMixin",
    "Document",
    "Figure",
    "FormField",
    "Link",
    "Page",
    "PdfUnsupportedError",
    "RectTuple",
    "StandardPdfEncryption",
    "TextLine",
    "WriteTarget",
    "bbox_union",
    "cluster_by",
    "coerce_bbox",
    "flip_box",
    "open_source",
    "serialize_document_to_pdf",
    "synthesize_characters",
    "write_bytes",
)
