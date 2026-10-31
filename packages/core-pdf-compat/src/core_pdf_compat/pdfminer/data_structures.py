from __future__ import annotations

from typing import Any

from .pdftypes import resolve1


class NumberTree:
    def __init__(self, obj: Any) -> None:
        node = resolve1(obj)
        self.nums = resolve1(node.get("Nums"))
        self.kids = resolve1(node.get("Kids"))

    @property
    def values(self) -> list[tuple[int, Any]]:
        items: list[tuple[int, Any]] = []
        nums = self.nums or []
        items.extend((resolve1(nums[i]), nums[i + 1]) for i in range(0, len(nums) - 1, 2))
        for kid in self.kids or []:
            items += NumberTree(kid).values
        return sorted(items, key=lambda item: item[0])


__all__ = ("NumberTree",)
