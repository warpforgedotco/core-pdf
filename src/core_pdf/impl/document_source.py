# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import contextlib
import mmap
from contextlib import AbstractContextManager
from os import PathLike
from types import TracebackType
from typing import TYPE_CHECKING, BinaryIO, Self

from core_pdf.impl.document_contracts import DocumentState
from core_pdf.impl.exceptions import PdfDocumentClosedError, PdfEmptySourceError, PdfSourceError
from core_pdf.impl.fonts_fallback import RasterFontRepository
from core_pdf.impl.types import PdfByteBuffer, PdfSource

if TYPE_CHECKING:
    pass


class DocumentOperation(AbstractContextManager["DocumentOperation"]):
    __slots__ = ("document", "released")

    def __init__(self, document: DocumentLifecycle) -> None:
        self.document = document
        self.released = False

    @property
    def cancelled(self) -> bool:
        return self.document.operation_cancelled.is_set()

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.document.release_operation()

    def __exit__(self, *args: object) -> None:
        self.release()


class DocumentLifecycle(DocumentState):
    __slots__ = ()

    def __enter__(self) -> Self:
        if self.closed:
            raise PdfDocumentClosedError("PDF document is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self.closing or self._closed

    def acquire_operation(self) -> DocumentOperation:
        with self.operation_lock:
            if self.closed:
                raise PdfDocumentClosedError("PDF document is closed")
            self.active_operations += 1
        return DocumentOperation(self)

    def release_operation(self) -> None:
        with self.operation_lock:
            self.active_operations = max(0, self.active_operations - 1)
            should_close = self.closing and self.active_operations == 0
        if should_close:
            self.close_resources()

    def close(self) -> None:
        with self.operation_lock:
            if self.closing or self._closed:
                return
            self.closing = True
            self.operation_cancelled.set()
            if self.active_operations:
                return
        self.close_resources()

    def close_resources(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.font_decoders.clear()
        self.caches.clear()

        resolver = getattr(self, "resolver", None)
        if resolver is not None:
            resolver.close()

        raster_fonts = self.raster_font_provider
        if isinstance(raster_fonts, RasterFontRepository):
            raster_fonts.close()

        raw_data = self.raw_data
        self.raw_data = b""
        if isinstance(raw_data, mmap.mmap):
            with contextlib.suppress(BufferError, OSError, ValueError):
                raw_data.close()

        if self.file_handle is not None:
            with contextlib.suppress(OSError):
                self.file_handle.close()
            self.file_handle = None


def load_source(source: PdfSource) -> tuple[PdfByteBuffer, BinaryIO | None]:
    if isinstance(source, (str, PathLike)):
        if isinstance(source, str) and source.startswith("%PDF"):
            return source.encode("latin-1"), None
        file_handle = open(source, "rb")  # noqa: SIM115
        try:
            return mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ), file_handle
        except (OSError, ValueError) as exc:
            try:
                is_empty = file_handle.seek(0, 2) == 0
            except OSError:
                is_empty = False
            file_handle.close()
            if is_empty:
                raise PdfEmptySourceError("PDF source is empty") from exc
            raise PdfSourceError(str(exc)) from exc
    if isinstance(source, bytes):
        return source, None
    if isinstance(source, (memoryview, bytearray)):
        return bytes(source), None

    mapped = try_mmap_reader(source)
    if mapped is not None:
        return mapped, None

    read = getattr(source, "read", None)
    if not callable(read):
        raise PdfSourceError(f"PDF source type {type(source).__name__} is not supported")
    reader = source
    tell = getattr(source, "tell", None)
    seek = getattr(source, "seek", None)
    position: int | None = None
    if callable(tell) and callable(seek):
        try:
            position = tell()
            seek(0)
        except OSError, TypeError, ValueError:
            position = None
    try:
        raw = reader.read()
    except OSError as exc:
        raise PdfSourceError(str(exc)) from exc
    finally:
        if position is not None and callable(seek):
            seek(position)
    return (raw if isinstance(raw, bytes) else bytes(raw)), None


def try_mmap_reader(source: object) -> mmap.mmap | None:
    fileno = getattr(source, "fileno", None)
    if not callable(fileno):
        return None
    try:
        fd = fileno()
    except OSError, TypeError, ValueError:
        return None
    try:
        return mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    except ValueError as error:
        raise PdfEmptySourceError("PDF source is empty") from error
    except OSError:
        return None


__all__ = ("DocumentLifecycle", "DocumentOperation", "load_source", "try_mmap_reader")
