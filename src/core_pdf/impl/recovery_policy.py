# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import ClassVar

from core_pdf_spec.s_07_syntax.trees import MalformedFn, raise_malformed
from core_records import GeneratedRecord


class Recovery:
    __slots__ = ("enabled",)

    enabled: bool

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def malformed(self, message: str) -> None:
        if not self.enabled:
            raise_malformed(message)

    def reject[T](self, error: BaseException, context: str, fallback: T) -> T:  # noqa: ARG002
        if not self.enabled:
            raise error
        return fallback


STRICT = Recovery(False)
LENIENT = Recovery(True)


class RecoveryMode(GeneratedRecord):
    objects: bool
    dictionary_structure: bool

    STRICT: ClassVar[RecoveryMode]
    TOLERANT: ClassVar[RecoveryMode]

    @property
    def dictionaries(self) -> bool:
        return self.objects and self.dictionary_structure


RecoveryMode.STRICT = RecoveryMode(False, False)
RecoveryMode.TOLERANT = RecoveryMode(True, True)


def recovery_policy(recover: bool | Recovery) -> Recovery:
    if isinstance(recover, Recovery):
        return recover
    return LENIENT if recover else STRICT


__all__ = (
    "LENIENT",
    "STRICT",
    "MalformedFn",
    "Recovery",
    "RecoveryMode",
    "recovery_policy",
)
