# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable


class internal_ExtractionCancelled(RuntimeError):
    def __init__(self) -> None:
        super().__init__("PDF extraction was cancelled")


class ExtractionScope:
    __slots__ = ("internal_cancelled",)

    def __init__(
        self,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self.internal_cancelled = cancelled

    def raise_if_cancelled(self) -> None:
        if self.internal_cancelled is not None and self.internal_cancelled():
            raise internal_ExtractionCancelled()


__all__ = ("ExtractionScope",)
