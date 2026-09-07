# SPDX-License-Identifier: AGPL-3.0-only
"""Operation-local extraction cancellation.

Extraction deliberately runs on the calling thread. Keeping cancellation in a
small scope object preserves cooperative document shutdown without retaining a
process-wide executor, worker queues, or reusable resource pools.
"""

from __future__ import annotations

from collections.abc import Callable


class internal_ExtractionCancelled(RuntimeError):
    def __init__(self) -> None:
        super().__init__("PDF extraction was cancelled")


class ExtractionScope:
    """Cancellation check shared by one synchronous extraction."""

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
