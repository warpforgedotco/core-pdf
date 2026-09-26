# SPDX-License-Identifier: AGPL-3.0-only
"""What a reader does with something malformed: raise, or skip it.

A check calls its policy with the message it would raise, then skips what
it checked; the strict policy raises ValueError(message) instead, so the
skip only runs while recovering.
"""

from __future__ import annotations

from core_pdf_spec.s_07_syntax.trees import MalformedFn, raise_malformed


def skip_malformed(message: str) -> None:
    """The recovering policy: nothing, so the caller skips what is malformed."""


def malformed_policy(recover: bool) -> MalformedFn:
    return skip_malformed if recover else raise_malformed


__all__ = ("MalformedFn", "malformed_policy", "raise_malformed", "skip_malformed")
