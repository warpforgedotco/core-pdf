# SPDX-License-Identifier: AGPL-3.0-only
"""Document source loading, cross-reference recovery, and security."""

from __future__ import annotations

import mmap
from collections.abc import Sequence
from os import PathLike
from typing import BinaryIO, cast

from core_pdf.impl.engine.spec.s_07_document.document_xref import DocumentXRefMixin
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_objects.resolver_values import PdfValueResolver
from core_pdf.impl.engine.spec.s_07_security.crypto_handlers import SECURITY_HANDLER_REGISTRY
from core_pdf.impl.engine.spec.s_07_security.errors import (
    PDFEncryptionError,
    PDFPasswordIncorrect,
)
from core_pdf.impl.exceptions import PdfSourceError, PdfUnsupportedError
from core_pdf.impl.objects import PdfReference
from core_pdf.impl.types import (
    Decipher,
    PathSource,
    PdfByteBuffer,
    PdfDict,
    PdfSource,
    SeekableBinaryReader,
)


class DocumentCoreMixin(DocumentXRefMixin):
    """Behavior needed to turn a source into a resolvable PDF object graph."""

    raw_data: PdfByteBuffer
    trailer_dict: PdfDict
    file_handle: BinaryIO | None
    decipher: Decipher | None
    resolver: PdfValueResolver

    @property
    def data(self) -> PdfByteBuffer:
        return self.raw_data

    @property
    def trailer(self) -> PdfDict:
        return self.trailer_dict

    def load_data(self, source: PdfSource) -> PdfByteBuffer:
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
        reader = source
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

    def try_mmap_reader(self, source: object) -> mmap.mmap | None:
        fileno = getattr(source, "fileno", None)
        if not callable(fileno):
            return None
        try:
            fd = fileno()
        except (OSError, TypeError, ValueError):
            return None
        try:
            return mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        except ValueError as error:
            raise PdfSourceError("PDF source is empty") from error
        except OSError:
            return None

    def init_security(self, password: str) -> None:
        encrypt_ref = lookup_dict_key(self.trailer_dict, "Encrypt")
        if encrypt_ref is None:
            return

        encrypt_dict = self.resolver.resolve_dict(encrypt_ref)
        if not isinstance(encrypt_dict, dict):
            raise PdfUnsupportedError("Invalid Encrypt dictionary")

        filter_name = self.resolver.resolve_name(lookup_dict_key(encrypt_dict, "Filter"))
        if filter_name is None:
            raise PdfUnsupportedError("Invalid encryption dictionary")
        if filter_name in {"Adobe.PubSec", "PubSec"}:
            raise PdfUnsupportedError("Public-key encryption is not supported")
        if filter_name != "Standard":
            raise PdfUnsupportedError(f"Unsupported encryption filter: {filter_name}")

        raw_v = lookup_dict_key(encrypt_dict, "V")
        if raw_v is None:
            raise PdfUnsupportedError("Invalid encryption dictionary")
        v = self.resolver.resolve_int(raw_v)
        if type(v) is not int:
            raise PdfUnsupportedError("Invalid encryption dictionary")
        handler_cls = SECURITY_HANDLER_REGISTRY.get(v)
        if handler_cls is None:
            raise PdfUnsupportedError(f"Unsupported standard encryption algorithm V={v}")

        docid = lookup_dict_key(self.trailer_dict, "ID")
        if docid is None:
            docid = [b""]
        if isinstance(docid, PdfReference):
            docid = self.resolver.resolve(docid)
        if not isinstance(docid, (list, tuple)) or len(docid) == 0:
            raise PdfUnsupportedError("Invalid trailer ID array")
        docid_list: Sequence[object] = docid

        try:
            handler = handler_cls(docid_list, encrypt_dict, password)
        except PDFPasswordIncorrect as exc:
            raise PdfUnsupportedError("Incorrect password") from exc
        except PDFEncryptionError as exc:
            raise PdfUnsupportedError("Invalid encryption dictionary") from exc
        self.decipher = handler.decrypt


__all__ = ("DocumentCoreMixin",)
