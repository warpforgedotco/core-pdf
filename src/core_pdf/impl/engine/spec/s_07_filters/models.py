# SPDX-License-Identifier: AGPL-3.0-only
"""Models shared by PDF stream filters and image consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy


@dataclass(frozen=True, slots=True)
class DecodedImage:
    """Native decoded image samples produced by an image filter."""

    array: numpy.ndarray[Any, Any]
    source: str

    def __post_init__(self) -> None:
        if self.array.ndim not in {2, 3}:
            raise ValueError("decoded image must have two or three dimensions")
        if self.array.dtype != numpy.uint8:
            raise ValueError("decoded image samples must be uint8")
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
