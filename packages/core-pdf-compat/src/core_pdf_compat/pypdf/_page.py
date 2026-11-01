"""The import path pypdf gives its page class and Transformation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ._images import ImageFile, VirtualListImages
from ._page_merge import Transformation
from ._writer import PageObject
from .errors import PdfReadError
from .generic import NameObject, NumberObject


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

    def __delitem__(self, index: Any) -> None:
        if isinstance(index, slice):
            for position in sorted(range(*index.indices(len(self))), reverse=True):
                del self[position]
            return
        if not isinstance(index, int):
            raise TypeError("Index must be integers")
        length = len(self)
        position = index + length if index < 0 else index
        if not 0 <= position < length:
            raise IndexError("Index out of range")
        self._remove_from_tree(position)

    def _remove_from_tree(self, position: int) -> None:
        child = self[position].indirect_reference
        assert child is not None
        pdf = child.pdf
        parent = child.get_object().get("/Parent")
        first = True
        while parent is not None:
            node = parent.get_object()
            try:
                del node["/Kids"][node["/Kids"].index(child)]
            except ValueError as error:
                if first:
                    raise PdfReadError(f"Page not found in page tree: {child}") from error
                return
            if first:
                del pdf.flattened_pages[position]
            first = False
            if "/Count" in node:
                node[NameObject("/Count")] = NumberObject(node["/Count"] - 1)
            if not node["/Kids"]:
                child = node.indirect_reference
            parent = node.get("/Parent")

    def __str__(self) -> str:
        return f"[{', '.join(f'PageObject({index})' for index in range(len(self)))}]"


__all__ = ["ImageFile", "PageObject", "Transformation", "VirtualListImages"]
