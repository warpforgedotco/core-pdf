# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import ClassVar

from core_pdf_spec.s_07_syntax.trees import MalformedFn, raise_malformed
from core_records import Record, frozen_setattr


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


class RecoveryMode(Record):
    __slots__ = ("objects", "dictionary_structure")

    objects: bool
    dictionary_structure: bool

    __fields__: ClassVar[tuple[str, ...]] = ("objects", "dictionary_structure")
    __match_args__ = ("objects", "dictionary_structure")

    STRICT: ClassVar[RecoveryMode]
    TOLERANT: ClassVar[RecoveryMode]

    def __init__(self, objects: bool, dictionary_structure: bool) -> None:
        frozen_setattr(self, "objects", objects)
        frozen_setattr(self, "dictionary_structure", dictionary_structure)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.objects == other.objects
            and self.dictionary_structure == other.dictionary_structure
        )

    def __hash__(self) -> int:
        return hash((self.objects, self.dictionary_structure))

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
