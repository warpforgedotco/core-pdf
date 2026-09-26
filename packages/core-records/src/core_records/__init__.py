# SPDX-License-Identifier: AGPL-3.0-only

"""Value-class mixins shared by every workspace package.

A value class declares ``__fields__`` and writes its own ``__init__``, ``__eq__``
and ``__hash__``; these supply the rest from ``__fields__``. Each is written to
behave exactly as the hand-written method it replaces did, and a subclass
that adds fields gets them in its repr, pickle state and replace because the
mixins read the instance's ``__fields__``.
"""

from __future__ import annotations

from typing import Any, ClassVar, NoReturn, Self

__all__ = (
    "FrozenFields",
    "PickleFields",
    "Record",
    "ReplaceFields",
    "ReprFields",
    "frozen_setattr",
)

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
    """``Qualname(field=value, ...)`` over the declared fields, in order.

    A class that should not print all of them -- a cached closure, a private
    backing field -- narrows the list with ``__repr_fields__`` rather than
    hand-writing the whole method.
    """

    __slots__ = ()

    __fields__: ClassVar[tuple[str, ...]]
    __repr_fields__: ClassVar[tuple[str, ...] | None] = None

    def __repr__(self) -> str:
        names = self.__repr_fields__ if self.__repr_fields__ is not None else self.__fields__
        fields = ", ".join([f"{name}={getattr(self, name)!r}" for name in names])
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
