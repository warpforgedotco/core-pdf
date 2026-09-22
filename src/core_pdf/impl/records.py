# SPDX-License-Identifier: AGPL-3.0-only
"""Shared behaviour for the hand-written value classes.

Every value class in the engine declares a ``__fields__`` tuple and hand-writes
the dunders ``@dataclass`` would generate. The three hot ones -- ``__init__``,
``__eq__`` and ``__hash__`` -- stay unrolled in each class. The rest carry no
per-instance advantage from being restated, so they live here.

``__setattr__``, ``__delattr__``, ``__getstate__`` and ``__setstate__`` were
byte-identical in every class that had them, and inheriting them costs nothing
measurable.

``__repr__`` and ``__replace__`` are deliberately *not* here, and the reason is
not the arithmetic they do. A method defined once on a shared base is one code
object reached from ~90 receiver types, so the interpreter's inline caches
cannot specialise it the way they specialise a method that belongs to a single
class. Measured in one process against the unrolled originals, a shared
``__repr__`` cost between 11% and 61% even when the formatting itself was a
single ``%``-interpolation over an ``operator.attrgetter``, and every shared
``__replace__`` lost by at least an eighth on the common one-change call. Both
run inside the render and extract loops, so each class keeps its own.

The four below are exempt from that argument: the guards do no field work at
all, and the pickle pair already read ``self.__fields__`` at call time, so
nothing was being specialised for them to lose.
"""

from __future__ import annotations

from typing import Any, ClassVar, NoReturn

internal_frozen_setattr = object.__setattr__


class internal_FrozenFields:
    """Rejects assignment and deletion, as ``@dataclass(frozen=True)`` does.

    Raises ``AttributeError`` rather than ``FrozenInstanceError`` so the guard
    needs no import; ``FrozenInstanceError`` derives from it.
    """

    __slots__ = ()

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")


class internal_PickleFields:
    """Pickle support for slotted classes whose ``__setattr__`` refuses writes."""

    __slots__ = ()

    __fields__: ClassVar[tuple[str, ...]]

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)


class internal_Record(internal_FrozenFields, internal_PickleFields):
    """Frozen and picklable: the pair nearly every value class here wants."""

    __slots__ = ()
