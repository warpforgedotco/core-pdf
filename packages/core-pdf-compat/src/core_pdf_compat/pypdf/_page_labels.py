# The logic of this module is pypdf._page_labels of pypdf 6.21.4, Copyright (c) 2006,
# Mathieu Fenniak, under its BSD-style license. pypdf reads the /PageLabels number tree in its
# own way, not quite as ISO 32000-2 12.4.2 describes it, so that reading is followed here
# rather than core-pdf's, and page_labels are pypdf's for every file.
"""pypdf's page labels: the label of a page index, from the document's /PageLabels."""

from __future__ import annotations

import logging
import string
from collections.abc import Callable
from typing import Any

from .generic import (
    ArrayObject,
    DictionaryObject,
    IndirectObject,
    NameObject,
    NullObject,
    NumberObject,
    is_null_or_none,
)

logger = logging.getLogger(__name__)

# The largest number the usual Roman symbols and subtractive notation represent.
MAXIMUM_ROMAN_NUMERAL = 3_999

# pypdf allows /A and /a labels of up to 512 repeated letters (512 * 26 = 13_312 pages).
MAXIMUM_PAGE_LABEL_LENGTH = 512

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
    if num <= 0:
        raise ValueError("Expecting a positive number.")
    if num > MAXIMUM_ROMAN_NUMERAL:
        raise ValueError("Number is out of range.")
    result = []
    for value, symbol in _ROMAN:
        count, num = divmod(num, value)
        result.append(symbol * count)
    return "".join(result)


def number2lowercase_roman_numeral(number: int) -> str:
    return number2uppercase_roman_numeral(number).lower()


def number2uppercase_letter(number: int) -> str:
    if number <= 0:
        raise ValueError("Expecting a positive number.")
    # A to Z, then the same letter repeated: AA to ZZ, AAA to ZZZ, and so on.
    repetitions, position = divmod(number - 1, 26)
    if repetitions + 1 > MAXIMUM_PAGE_LABEL_LENGTH:
        raise ValueError("Number is too large.")
    return string.ascii_uppercase[position] * (repetitions + 1)


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
        next_key = nums[i + 2]
        if not isinstance(next_key, (int, float)):
            logger.warning("Ignoring malformed page label key %r in /Nums.", next_key)
            return str(index + 1)
        if next_key > index:
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
    except (TypeError, ValueError) as exception:
        logger.warning(
            "Ignoring malformed page label entry in /Nums (/St=%r, /P=%r): %s",
            start,
            prefix,
            exception,
        )
        return str(index + 1)


def index2label(reader: Any, index: int) -> str:
    """The label of the page at index, e.g. "iv" or "4", read from /PageLabels as pypdf does."""
    root = reader.root_object
    if "/PageLabels" not in root:
        return str(index + 1)
    number_tree = root["/PageLabels"].get_object()
    if not isinstance(number_tree, DictionaryObject):
        logger.warning("Page labels are not a dictionary: %s", number_tree)
        return str(index + 1)
    if "/Nums" in number_tree:
        return get_label_from_nums(number_tree, index)
    if "/Kids" in number_tree and not isinstance(number_tree["/Kids"], NullObject):
        level = 0
        while level < 100:
            kids = number_tree["/Kids"].get_object()
            if not isinstance(kids, ArrayObject):
                logger.warning("Page label kids are not an array: %s", kids)
                break
            for reference in kids:
                kid = reference.get_object()
                if not isinstance(kid, DictionaryObject):
                    logger.warning("Ignoring kid which is not a dictionary in /PageLabels.")
                    continue
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


def _get_single_pass_nums_length(nums: ArrayObject, page_count: int) -> int | None:
    """How much of a flat /Nums array labels the first page_count pages, if it can be read once.

    get_label_from_nums reads the first pair, then stops at the first key above the page
    index, so only the array up to the first key at or above page_count after the first pair
    matters. None unless that part holds pairs of increasing integer keys and direct values.
    """
    nums_length = len(nums)
    previous = None
    for i in range(0, nums_length, 2):
        key = nums[i]
        if (
            i + 1 >= nums_length
            or not isinstance(key, NumberObject)
            or isinstance(nums[i + 1], IndirectObject)
            or (previous is not None and key <= previous)
        ):
            return None
        if key >= page_count and i > 0:
            return i + 2
        previous = key
    return nums_length or None


def index2labels(pdf: Any, page_count: int) -> list[str]:
    """The labels of the first page_count pages, as index2label returns them, read in one pass
    over a flat /Nums array with increasing integer keys and direct values (pypdf #4251)."""
    if page_count <= 0:
        return []
    root = pdf.root_object
    number_tree = root["/PageLabels"].get_object() if "/PageLabels" in root else None
    nums = (
        number_tree["/Nums"]
        if isinstance(number_tree, DictionaryObject) and "/Nums" in number_tree
        else None
    )
    nums_length = (
        _get_single_pass_nums_length(nums, page_count) if isinstance(nums, ArrayObject) else None
    )
    if nums is None or nums_length is None:
        return [index2label(pdf, index) for index in range(page_count)]
    labels = []
    position = 0
    for index in range(page_count):
        while position + 2 < nums_length and nums[position + 2] <= index:
            position += 2
        entry = DictionaryObject({NameObject("/Nums"): ArrayObject(nums[position : position + 2])})
        labels.append(get_label_from_nums(entry, index))
    return labels
