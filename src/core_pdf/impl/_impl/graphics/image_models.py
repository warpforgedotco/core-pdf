# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, ClassVar, NoReturn, Self

import numpy

internal_frozen_setattr = object.__setattr__


class DecodedImage:
    __slots__ = ("array", "source")

    array: numpy.ndarray[Any, Any]
    source: str

    __fields__: ClassVar[tuple[str, ...]] = ("array", "source")
    __match_args__ = ("array", "source")

    def __init__(self, array: numpy.ndarray[Any, Any], source: str) -> None:
        internal_frozen_setattr(self, "array", array)
        internal_frozen_setattr(self, "source", source)
        self._post_init()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(array={self.array!r}, source={self.source!r})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.array == other.array and self.source == other.source

    def __hash__(self) -> int:
        return hash((self.array, self.source))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        array = changes.pop("array", self.array)
        source = changes.pop("source", self.source)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(array, source)

    def _post_init(self) -> None:
        if self.array.ndim not in {2, 3}:
            raise ValueError("decoded image must have two or three dimensions")
        if self.array.dtype not in (numpy.uint8, numpy.uint16):
            raise ValueError("decoded image samples must be uint8 or uint16")
        if not self.array.flags.c_contiguous:
            raise ValueError("decoded image must be C-contiguous")

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def channels(self) -> int:
        return 1 if self.array.ndim == 2 else int(self.array.shape[2])
