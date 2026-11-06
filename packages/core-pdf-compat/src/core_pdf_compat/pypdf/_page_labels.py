# The logic of this module is pypdf._page_labels, Copyright (c) 2006, Mathieu Fenniak,
# under its BSD-style license. pypdf reads the /PageLabels number tree in its own way, not
# quite as ISO 32000-2 12.4.2 describes it, so that reading is followed here rather than
# core-pdf's, and page_labels are pypdf's for every file.
"""pypdf's page labels: the label of a page index, from the document's /PageLabels."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

from .generic import NullObject, is_null_or_none

logger = logging.getLogger(__name__)

_ROMAN = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)


def number2uppercase_roman_numeral(num: int) -> str:
    def roman_num(num: int) -> Iterator[str]:
        for decimal, roman_repr in _ROMAN:
            x, _ = divmod(num, decimal)
            yield roman_repr * x
            num -= decimal * x
            if num <= 0:
                break

    return "".join(roman_num(num))


def number2lowercase_roman_numeral(number: int) -> str:
    return number2uppercase_roman_numeral(number).lower()


def number2uppercase_letter(number: int) -> str:
    if number <= 0:
        raise ValueError("Expecting a positive number")
    rep = ""
    while number > 0:
        remainder = number % 26 or 26
        rep = chr(ord("A") + remainder - 1) + rep
        number = (number - remainder) // 26
    return rep


def number2lowercase_letter(number: int) -> str:
    return number2uppercase_letter(number).lower()


_STYLES: dict[Any, Callable[[int], str]] = {
    None: lambda _: "",
    "/D": str,
    "/R": number2uppercase_roman_numeral,
    "/r": number2lowercase_roman_numeral,
    "/A": number2uppercase_letter,
    "/a": number2lowercase_letter,
}


def get_label_from_nums(dictionary_object: Any, index: int) -> str:
    # /Nums is [key_1 value_1 ... key_n value_n], keys sorted (ISO 32000-2 7.9.7).
    nums = dictionary_object["/Nums"]
    nums_length = len(nums)
    i = 0
    value = None
    start_index = 0
    while i < nums_length:
        if i + 1 >= nums_length:
            logger.warning("Ignoring last /Nums key without a value.")
            break
        start_index = nums[i]
        value = nums[i + 1].get_object()
        if i + 2 == nums_length:
            break
        if nums[i + 2] > index:
            break
        i += 2
    # /Nums is empty, or does not follow the specification.
    if not isinstance(value, dict):
        return str(index + 1)
    start = value.get("/St", 1)
    prefix = value.get("/P", "")
    mapping_function = _STYLES.get(value.get("/S"))
    if mapping_function is None:
        logger.warning("Ignoring unknown page label numbering style %r in /Nums.", value.get("/S"))
        return str(index + 1)
    try:
        return prefix + mapping_function(index - start_index + start)
    except TypeError, ValueError:
        logger.warning(
            "Ignoring malformed page label entry in /Nums (/St=%r, /P=%r).", start, prefix
        )
        return str(index + 1)


def index2label(reader: Any, index: int) -> str:
    """The label of the page at index, e.g. "iv" or "4", read from /PageLabels as pypdf does."""
    root = reader.root_object
    if "/PageLabels" not in root:
        return str(index + 1)
    number_tree = root["/PageLabels"].get_object()
    if "/Nums" in number_tree:
        return get_label_from_nums(number_tree, index)
    if "/Kids" in number_tree and not isinstance(number_tree["/Kids"], NullObject):
        level = 0
        while level < 100:
            for kid in number_tree["/Kids"]:
                limits = kid.get("/Limits", NullObject()).get_object()
                if not isinstance(limits, list) or len(limits) < 2:
                    logger.warning("Ignoring kid with missing or malformed /Limits in /PageLabels.")
                    continue
                if limits[0] <= index <= limits[1]:
                    if not is_null_or_none(kid.get("/Kids", None)):
                        level += 1
                        if level == 100:  # pragma: no cover
                            raise NotImplementedError("Too deep nesting is not supported.")
                        number_tree = kid
                        break
                    return get_label_from_nums(kid, index)
            else:
                break
    logger.warning("Could not reliably determine page label for %d.", index)
    return str(index + 1)
