# SPDX-License-Identifier: AGPL-3.0-only
"""PDF document writing entry points."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast

from core_pdf.impl.engine.spec.s_07_syntax.xref import XRefScanner
from core_pdf.impl.engine.writing.encryption import StandardPdfEncryptionContext
from core_pdf.impl.engine.writing.incremental import append_incremental_update
from core_pdf.impl.exceptions import PdfDocumentClosedError, PdfUnsupportedError

if TYPE_CHECKING:
    from core_pdf.impl.types import PdfObject


class PdfDocumentWritingMixin:
    """Format-specific save operations for parsed PDF documents."""

    def save_incremental(
        self: Any,
        target: str | PathLike[str] | BinaryIO,
        objects: Mapping[int, PdfObject],
        *,
        trailer: Mapping[object, object] | None = None,
    ) -> bytes:
        """Append an incremental update and write it to ``target``.

        ``objects`` contains replacement or newly allocated indirect objects.
        The returned bytes are also written to the target, which may be a path
        or an already-open binary stream.
        """
        if self.closed:
            raise PdfDocumentClosedError("PDF document is closed")
        original = bytes(self.raw_data)
        previous_xref_offset = find_startxref(original)
        previous_size = previous_object_count(self.xref, self.trailer_dict)
        objects_to_write = objects
        if self.decipher is not None:
            handler = getattr(self.decipher, "__self__", None)
            if handler is None:
                raise PdfUnsupportedError("encrypted PDF security handler is unavailable")
            try:
                context = StandardPdfEncryptionContext.from_security_handler(handler)
            except ValueError as exc:
                raise PdfUnsupportedError(str(exc)) from exc
            objects_to_write = {
                number: cast(Any, context.encrypt_object(value, number))
                for number, value in objects.items()
            }
        updated = append_incremental_update(
            original,
            objects_to_write,
            trailer={**self.trailer_dict, **dict(trailer or {})},
            previous_xref_offset=previous_xref_offset,
            previous_size=previous_size,
        )
        if isinstance(target, (str, PathLike)):
            Path(cast(str | PathLike[str], target)).write_bytes(updated)
        else:
            cast(BinaryIO, target).write(updated)
        return updated


def find_startxref(data: bytes) -> int:
    offset = XRefScanner.find_startxref(data)
    if offset is None:
        raise ValueError("PDF does not contain a startxref marker")
    return offset


def previous_object_count(
    xref: Mapping[int, object],
    trailer: Mapping[object, object],
) -> int:
    raw_size = trailer.get("Size")
    if raw_size is None:
        raw_size = trailer.get(b"Size")
    if type(raw_size) is int and raw_size > 0:
        return raw_size
    object_numbers = [key >> 16 for key in xref if key > 0]
    return max(object_numbers, default=0) + 1


__all__ = ("PdfDocumentWritingMixin", "find_startxref", "previous_object_count")
