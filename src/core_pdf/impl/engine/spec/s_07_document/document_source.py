# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import mmap
from os import PathLike
from typing import BinaryIO, Protocol, cast

from core_pdf.impl.exceptions import PdfSourceError
from core_pdf.impl.types import (
    BinaryReader,
    PathSource,
    PdfByteBuffer,
    PdfDict,
    PdfSource,
    SeekableBinaryReader,
)


class DocumentSourceHost(Protocol):
    raw_data: PdfByteBuffer
    trailer_dict: PdfDict
    file_handle: BinaryIO | None

    def try_mmap_reader(self, source: object) -> mmap.mmap | None: ...


class DocumentSourceMixin:
    raw_data: PdfByteBuffer
    trailer_dict: PdfDict
    file_handle: BinaryIO | None

    @property
    def data(self: DocumentSourceHost) -> PdfByteBuffer:
        return self.raw_data

    @property
    def trailer(self: DocumentSourceHost) -> PdfDict:
        return self.trailer_dict

    def load_data(self: DocumentSourceHost, source: PdfSource) -> PdfByteBuffer:
        if isinstance(source, (str, PathLike)):
            if isinstance(source, str) and source.startswith("%PDF"):
                return source.encode("latin-1")
            file_handle = open(cast(PathSource, source), "rb")  # noqa: SIM115
            self.file_handle = file_handle
            try:
                return mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            except (OSError, ValueError) as exc:
                try:
                    is_empty = file_handle.seek(0, 2) == 0
                except OSError:
                    is_empty = False
                file_handle.close()
                self.file_handle = None
                if is_empty:
                    raise PdfSourceError("PDF source is empty") from exc
                raise PdfSourceError(str(exc)) from exc
        if isinstance(source, bytes):
            return source
        if isinstance(source, (memoryview, bytearray)):
            return bytes(source)

        mapped = self.try_mmap_reader(source)
        if mapped is not None:
            return mapped

        read = getattr(source, "read", None)
        if not callable(read):
            raise PdfSourceError(f"PDF source type {type(source).__name__} is not supported")
        reader = cast(BinaryReader, source)
        tell = getattr(source, "tell", None)
        seek = getattr(source, "seek", None)
        position: int | None = None
        seekable: SeekableBinaryReader | None = None
        if callable(tell) and callable(seek):
            seekable = cast(SeekableBinaryReader, source)
            try:
                position = seekable.tell()
                seekable.seek(0)
            except (OSError, TypeError, ValueError):
                position = None
                seekable = None
        try:
            raw = reader.read()
        except OSError as exc:
            raise PdfSourceError(str(exc)) from exc
        finally:
            if position is not None and seekable is not None:
                seekable.seek(position)
        return raw if isinstance(raw, bytes) else bytes(raw)

    def try_mmap_reader(self: DocumentSourceHost, source: object) -> mmap.mmap | None:
        fileno = getattr(source, "fileno", None)
        if not callable(fileno):
            return None
        try:
            fd = fileno()
        except (OSError, TypeError, ValueError):
            return None
        try:
            return mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        except ValueError:
            raise PdfSourceError("PDF source is empty")
        except OSError:
            return None


__all__ = ("DocumentSourceMixin",)
