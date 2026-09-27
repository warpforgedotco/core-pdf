# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from core_pdf_spec.s_07_syntax.trees import MalformedFn, raise_malformed
from core_records import Record, frozen_setattr

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


class RecoveryMode(Record):
    __slots__ = ("objects", "dictionary_structure", "malformed")

    objects: bool
    dictionary_structure: bool
    malformed: Recovery

    __fields__: ClassVar[tuple[str, ...]] = ("objects", "dictionary_structure", "malformed")
    __match_args__ = ("objects", "dictionary_structure", "malformed")

    STRICT: ClassVar[RecoveryMode]
    TOLERANT: ClassVar[RecoveryMode]

    def __init__(self, objects: bool, dictionary_structure: bool, malformed: Recovery) -> None:
        frozen_setattr(self, "objects", objects)
        frozen_setattr(self, "dictionary_structure", dictionary_structure)
        frozen_setattr(self, "malformed", malformed)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.objects == other.objects
            and self.dictionary_structure == other.dictionary_structure
            and self.malformed is other.malformed
        )

    def __hash__(self) -> int:
        return hash((self.objects, self.dictionary_structure, id(self.malformed)))

    @classmethod
    def for_document(cls, recovery_enabled: bool) -> RecoveryMode:
        return cls.TOLERANT if recovery_enabled else cls.STRICT

    @classmethod
    def for_objects(cls, objects: bool, dictionary_structure: bool) -> RecoveryMode:
        if objects and dictionary_structure:
            return cls.TOLERANT
        return cls(objects, dictionary_structure, LENIENT if objects else STRICT)


RecoveryMode.STRICT = RecoveryMode(False, False, STRICT)
RecoveryMode.TOLERANT = RecoveryMode(True, True, LENIENT)


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
    "RecoveryMode",
    "malformed_policy",
    "recovery_policy",
)
