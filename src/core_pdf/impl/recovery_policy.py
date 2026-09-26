# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core_pdf_spec.s_07_syntax.trees import MalformedFn, raise_malformed

type RecoveredErrors = type[BaseException] | tuple[type[BaseException], ...]


class Recovery:
    __slots__ = ("enabled",)

    enabled: bool

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def malformed(self, message: str) -> None:
        if not self.enabled:
            raise_malformed(message)

    def attempt[T](
        self,
        strict: Callable[..., T],
        repair: Callable[..., T],
        /,
        *args: Any,
        errors: RecoveredErrors = ValueError,
    ) -> T:
        if not self.enabled:
            return strict(*args)
        try:
            return strict(*args)
        except errors:
            return repair(*args)

    def reject[T](self, error: BaseException, context: str, fallback: T) -> T:  # noqa: ARG002
        if not self.enabled:
            raise error
        return fallback


STRICT = Recovery(False)
LENIENT = Recovery(True)


def recovery_policy(recover: bool | Recovery) -> Recovery:
    if isinstance(recover, Recovery):
        return recover
    return LENIENT if recover else STRICT


def malformed_policy(recover: bool | Recovery) -> MalformedFn:
    return recovery_policy(recover).malformed


__all__ = (
    "LENIENT",
    "STRICT",
    "MalformedFn",
    "RecoveredErrors",
    "Recovery",
    "malformed_policy",
    "recovery_policy",
)
