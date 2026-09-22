# SPDX-License-Identifier: AGPL-3.0-only
"""Shared behaviour for the hand-written value classes.

Every value class in the engine declares a ``__fields__`` tuple and hand-writes
the dunders ``@dataclass`` would generate. Everything that can be driven from
``__fields__`` alone lives here instead, so a class body carries its fields and
its own behaviour rather than a hundred lines of restated protocol.

Only ``__init__``, ``__eq__`` and ``__hash__`` stay unrolled per class: the
first because each class has its own parameter list and defaults, the other two
because a field-driven version would have to build a tuple on every comparison.

A locally defined dunder always wins, so a class whose ``__repr__`` hides part
of its state, or whose ``__replace__`` cannot rebuild from its fields
positionally, simply keeps its own.
"""

from __future__ import annotations

from typing import Any, ClassVar, NoReturn, Self

frozen_setattr = object.__setattr__


class FrozenFields:
    """Rejects assignment and deletion, as ``@dataclass(frozen=True)`` does.

    Raises ``AttributeError`` rather than ``FrozenInstanceError`` so the guard
    needs no import; ``FrozenInstanceError`` derives from it.
    """

    __slots__ = ()

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")


class PickleFields:
    """Pickle support for slotted classes whose ``__setattr__`` refuses writes."""

    __slots__ = ()

    __fields__: ClassVar[tuple[str, ...]]

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)


class ReprFields:
    """``Qualname(field=value, ...)`` over every declared field, in order."""

    __slots__ = ()

    __fields__: ClassVar[tuple[str, ...]]

    def __repr__(self) -> str:
        fields = ", ".join([f"{name}={getattr(self, name)!r}" for name in self.__fields__])
        return f"{self.__class__.__qualname__}({fields})"


class ReplaceFields:
    """``copy.replace`` for classes whose ``__init__`` takes the fields
    positionally, in ``__fields__`` order."""

    __slots__ = ()

    __fields__: ClassVar[tuple[str, ...]]

    def __replace__(self, /, **changes: Any) -> Self:
        values = [changes.pop(name, getattr(self, name)) for name in self.__fields__]
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(*values)


class Record(FrozenFields, PickleFields, ReprFields, ReplaceFields):
    """The whole set: frozen, picklable, field-printing, field-replacing."""

    __slots__ = ()
