# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Callable, Iterator

from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key

ResolveFn = Callable[[object], object]
NameDecodeFn = Callable[[object], str | None]


def iter_number_tree_items(
    node: object,
    resolve: ResolveFn,
    *,
    recover: bool = False,
    max_depth: int = 100,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Iterator[tuple[int, object]]:
    if seen is None:
        seen = set()
    stack: list[tuple[object, int]] = [(node, depth)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            if recover:
                continue
            raise ValueError("invalid number tree depth")
        current = resolve(current)
        if current is None:
            continue
        if not isinstance(current, dict):
            if recover:
                continue
            raise ValueError("invalid number tree node")
        marker = id(current)
        if marker in seen:
            if recover:
                continue
            raise ValueError("number tree cycle detected")
        seen.add(marker)

        nums = resolve(lookup_dict_key(current, "Nums"))
        if nums is not None:
            if not isinstance(nums, list):
                if recover:
                    continue
                raise ValueError("invalid number tree Nums array")
            if len(nums) % 2 != 0 and not recover:
                raise ValueError("invalid number tree Nums array")
            nums_len = len(nums) - (len(nums) % 2)
            for index in range(0, nums_len, 2):
                key = resolve(nums[index])
                if type(key) is not int:
                    if recover:
                        continue
                    raise ValueError("invalid number tree key")
                yield key, resolve(nums[index + 1])

        kids = resolve(lookup_dict_key(current, "Kids"))
        if kids is None:
            continue
        if not isinstance(kids, list):
            if recover:
                continue
            raise ValueError("invalid number tree Kids array")
        for kid in reversed(kids):
            stack.append((kid, depth + 1))


def iter_name_tree_items(
    node: object,
    resolve: ResolveFn,
    decode_name: NameDecodeFn,
    *,
    recover: bool = False,
    max_depth: int = 100,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Iterator[tuple[str, object]]:
    if seen is None:
        seen = set()
    stack: list[tuple[object, int]] = [(node, depth)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            if recover:
                continue
            raise ValueError("invalid name tree depth")
        current = resolve(current)
        if current is None:
            continue
        if not isinstance(current, dict):
            if recover:
                continue
            raise ValueError("invalid name tree node")
        marker = id(current)
        if marker in seen:
            if recover:
                continue
            raise ValueError("name tree cycle detected")
        seen.add(marker)

        names = resolve(lookup_dict_key(current, "Names"))
        if names is not None:
            if not isinstance(names, list):
                if recover:
                    continue
                raise ValueError("invalid name tree Names array")
            if len(names) % 2 != 0 and not recover:
                raise ValueError("invalid name tree Names array")
            names_len = len(names) - (len(names) % 2)
            for index in range(0, names_len, 2):
                name = decode_name(names[index])
                if name is None:
                    if recover:
                        continue
                    raise ValueError("invalid name tree key")
                yield name, resolve(names[index + 1])

        kids = resolve(lookup_dict_key(current, "Kids"))
        if kids is None:
            continue
        if not isinstance(kids, list):
            if recover:
                continue
            raise ValueError("invalid name tree Kids array")
        for kid in reversed(kids):
            stack.append((kid, depth + 1))
