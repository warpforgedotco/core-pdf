"""Input aliases and small callback contracts shared by the concrete v0 API."""

from __future__ import annotations

from collections.abc import Iterable
from os import PathLike
from typing import Protocol, TypeAlias

from .errors import OperationCancelled

PageSelection: TypeAlias = int | str | range | Iterable[int]


class ReadableSource(Protocol):
    def read(self, size: int = -1, /) -> bytes | bytearray | memoryview: ...


PdfInput: TypeAlias = str | PathLike[str] | bytes | bytearray | memoryview | ReadableSource


class CancellationToken(Protocol):
    @property
    def cancelled(self) -> bool: ...

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise OperationCancelled


class ExecutionContext(Protocol):
    cancellation: CancellationToken


class SignatureProvider(Protocol):
    def sign(self, data: bytes) -> bytes: ...


__all__ = (
    "CancellationToken",
    "ExecutionContext",
    "PageSelection",
    "PdfInput",
    "ReadableSource",
    "SignatureProvider",
)
