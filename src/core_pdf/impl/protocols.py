# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Protocol


class BinaryReader(Protocol):
    """Readable binary source that can be materialized before parsing."""

    def read(self, size: int = -1, /) -> bytes | bytearray | memoryview: ...


class SeekableBinaryReader(BinaryReader, Protocol):
    """Random-access binary source aligned with PDF byte-offset structure."""

    def seek(self, offset: int, whence: int = 0, /) -> int: ...

    def tell(self) -> int: ...

    def fileno(self) -> int: ...
