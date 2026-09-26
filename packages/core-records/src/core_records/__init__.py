# SPDX-License-Identifier: AGPL-3.0-only


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
    __slots__ = ()

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")


class PickleFields:
    __slots__ = ()

    __fields__: ClassVar[tuple[str, ...]]

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)


class ReprFields:
    __slots__ = ()

    __fields__: ClassVar[tuple[str, ...]]
    __repr_fields__: ClassVar[tuple[str, ...] | None] = None

    def __repr__(self) -> str:
        names = self.__repr_fields__ if self.__repr_fields__ is not None else self.__fields__
        fields = ", ".join([f"{name}={getattr(self, name)!r}" for name in names])
        return f"{self.__class__.__qualname__}({fields})"


class ReplaceFields:
    __slots__ = ()

    __fields__: ClassVar[tuple[str, ...]]

    def __replace__(self, /, **changes: Any) -> Self:
        values = [changes.pop(name, getattr(self, name)) for name in self.__fields__]
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(*values)


class Record(FrozenFields, PickleFields, ReprFields, ReplaceFields):
    __slots__ = ()
