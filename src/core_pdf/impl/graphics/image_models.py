# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy

from core_pdf.impl.records import internal_Record

internal_frozen_setattr = object.__setattr__


class DecodedImage(internal_Record):
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
