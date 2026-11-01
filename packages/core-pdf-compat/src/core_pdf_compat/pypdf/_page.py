"""The import path pypdf gives its page class and Transformation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ._images import ImageFile, VirtualListImages
from ._page_merge import Transformation
from ._writer import PageObject


class _VirtualList(Sequence[PageObject]):
    def __init__(
        self, length_function: Callable[[], int], get_function: Callable[[int], PageObject]
    ) -> None:
        self.length_function = length_function
        self.get_function = get_function

    def __len__(self) -> int:
        return self.length_function()

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return _VirtualList(indices.__len__, lambda position: self[indices[position]])
        if not isinstance(index, int):
            raise TypeError("Sequence indices must be integers")
        length = len(self)
        position = index + length if index < 0 else index
        if not 0 <= position < length:
            raise IndexError("Sequence index out of range")
        return self.get_function(position)

    def __str__(self) -> str:
        return f"[{', '.join(f'PageObject({index})' for index in range(len(self)))}]"


__all__ = ["ImageFile", "PageObject", "Transformation", "VirtualListImages"]
