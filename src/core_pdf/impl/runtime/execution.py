# SPDX-License-Identifier: AGPL-3.0-only
"""Operation-local extraction cancellation.

Extraction deliberately runs on the calling thread. Keeping cancellation in a
small scope object preserves cooperative document shutdown without retaining a
process-wide executor, worker queues, or reusable resource pools.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager


class internal_ExtractionCancelled(RuntimeError):
    def __init__(self) -> None:
        super().__init__("PDF extraction was cancelled")


class ExtractionScope(AbstractContextManager["ExtractionScope"]):
    """Cancellation callbacks shared by one synchronous extraction."""

    __slots__ = ("internal_cancellations",)

    def __init__(
        self,
        cancelled: Callable[[], bool] | None = None,
        *,
        internal_cancellations: tuple[Callable[[], bool], ...] | None = None,
    ) -> None:
        self.internal_cancellations = (
            internal_cancellations
            if internal_cancellations is not None
            else ((cancelled,) if cancelled is not None else ())
        )

    def __enter__(self) -> ExtractionScope:
        return self

    def __exit__(self, *internal_args: object) -> None:
        return None

    def with_cancellation(self, cancelled: Callable[[], bool]) -> ExtractionScope:
        return ExtractionScope(
            internal_cancellations=(*self.internal_cancellations, cancelled),
        )

    def cancelled(self) -> bool:
        return any(cancelled() for cancelled in self.internal_cancellations)

    def raise_if_cancelled(self) -> None:
        if self.cancelled():
            raise internal_ExtractionCancelled()


__all__ = ("ExtractionScope",)
