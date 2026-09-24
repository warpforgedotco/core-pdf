# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable


class ExtractionCancelled(RuntimeError):
    def __init__(self) -> None:
        super().__init__("PDF extraction was cancelled")


class ExtractionScope:
    __slots__ = ("_cancelled",)

    def __init__(
        self,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self._cancelled = cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled is not None and self._cancelled():
            raise ExtractionCancelled()


__all__ = ("ExtractionScope",)
