# The logic of this module is pypdf._page_labels of pypdf 7.0.0, Copyright (c) 2006,
# Mathieu Fenniak, under its BSD-style license. pypdf reads the /PageLabels number tree in its
# own way, not quite as ISO 32000-2 12.4.2 describes it, so that reading is followed here
# rather than core-pdf's, and page_labels are pypdf's for every file.
"""pypdf's page labels: the label of a page index, from the document's /PageLabels."""

from __future__ import annotations

import logging
import math
import string
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

from .generic import ArrayObject, DictionaryObject, NullObject, is_null_or_none

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


def _get_label(value: Any, start_index: Any, index: int) -> str:
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


def get_label_from_nums(dictionary_object: Any, index: int) -> str:
    # /Nums is [key_1 value_1 ... key_n value_n], keys sorted (ISO 32000-2 7.9.7).
    nums = dictionary_object["/Nums"]
    if not isinstance(nums, ArrayObject):
        logger.warning("Page label numbers are not an array: %s", nums)
        return str(index + 1)
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
    return _get_label(value, start_index, index)


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


def _get_labels_for_page_indexes(nums: ArrayObject, indexes: Iterable[int]) -> list[str]:
    """Labels for increasing page indexes, as get_label_from_nums returns them.

    get_label_from_nums stops at the first pair whose next key is above the page index. That
    pair only moves forward as the index grows, so the array is read once for all pages, and
    each value is resolved when its pair is first reached (pypdf #4321).
    """
    nums_length = len(nums)
    labels = []
    i = 0
    resolved = -1
    start_index: Any = 0
    value: Any = None
    for index in indexes:
        is_key_malformed = False
        while i < nums_length:
            if i + 1 >= nums_length:
                logger.warning("Ignoring last /Nums key without a value.")
                break
            if resolved != i:
                start_index = nums[i]
                value = nums[i + 1].get_object()
                resolved = i
            if i + 2 == nums_length:
                break
            next_key = nums[i + 2]
            if not isinstance(next_key, (int, float)):
                logger.warning("Ignoring malformed page label key %r in /Nums.", next_key)
                is_key_malformed = True
                break
            if next_key > index:
                break
            i += 2
        labels.append(str(index + 1) if is_key_malformed else _get_label(value, start_index, index))
    return labels


def _get_page_range(limits: list[Any], page_count: int) -> tuple[int, int] | None:
    """The first and last page index between the two /Limits values, if any."""
    low, high = limits[0], limits[1]
    if low >= page_count or high < 0:
        return None
    last_page = page_count - 1
    first = 0 if low <= 0 else math.ceil(low)
    last = last_page if high >= last_page else math.floor(high)
    return (first, last) if first <= last else None


class _FirstKidFinder:
    """The kid index2label picks for a page index: the first kid in order whose /Limits
    contain it. Kids are read in order and only as far as needed, each once; the pages the
    kids read so far cover are kept as sorted ranges. A kid that cannot be read is returned
    as the exception, which index2label raises for every page reaching it.
    """

    def __init__(self, kids: ArrayObject, page_count: int) -> None:
        self._kids = kids
        self._kid_count = len(kids)
        self._page_count = page_count
        self._next_kid = 0
        # Pages covered by any kid read so far, as merged ranges.
        self._covered_starts: list[int] = []
        self._covered_ends: list[int] = []
        # Ranges of pages with the same first kid.
        self._starts: list[int] = []
        self._ends: list[int] = []
        self._first_kids: list[DictionaryObject | Exception] = []

    def find(self, index: int) -> DictionaryObject | Exception | None:
        while True:
            position = bisect_right(self._starts, index) - 1
            if position >= 0 and index <= self._ends[position]:
                return self._first_kids[position]
            if self._next_kid >= self._kid_count:
                return None
            kid_range = self._read_kid(self._kids[self._next_kid])
            self._next_kid += 1
            if kid_range is not None:
                self._cover(*kid_range)

    def _read_kid(self, kid: Any) -> tuple[int, int, DictionaryObject | Exception] | None:
        try:
            kid = kid.get_object()
            limits = (
                kid.get("/Limits", NullObject()).get_object()
                if isinstance(kid, DictionaryObject)
                else None
            )
        except MemoryError:  # pragma: no cover
            raise
        except Exception as exception:  # noqa: BLE001 - raised later, for each page reaching it
            return 0, self._page_count - 1, exception
        if not isinstance(kid, DictionaryObject):
            logger.warning("Ignoring kid which is not a dictionary in /PageLabels.")
            return None
        if (
            not isinstance(limits, list)
            or len(limits) < 2
            or not all(isinstance(limit, (int, float)) for limit in limits[:2])
        ):
            logger.warning("Ignoring kid with missing or malformed /Limits in /PageLabels.")
            return None
        page_range = _get_page_range(limits, self._page_count)
        return None if page_range is None else (*page_range, kid)

    def _cover(self, first: int, last: int, kid: DictionaryObject | Exception) -> None:
        """Make kid the first kid of the pages from first to last that no earlier kid covers."""
        start = bisect_left(self._covered_ends, first)
        end = bisect_right(self._covered_starts, last)
        gaps = []
        position = first
        for covered_start, covered_end in zip(
            self._covered_starts[start:end], self._covered_ends[start:end], strict=True
        ):
            if covered_start > position:
                gaps.append((position, covered_start - 1))
            position = max(position, covered_end + 1)
        if position <= last:
            gaps.append((position, last))
        if start < end:
            first = min(first, self._covered_starts[start])
            last = max(last, self._covered_ends[end - 1])
        self._covered_starts[start:end] = [first]
        self._covered_ends[start:end] = [last]
        for gap_start, gap_end in gaps:
            position = bisect_left(self._starts, gap_start)
            self._starts.insert(position, gap_start)
            self._ends.insert(position, gap_end)
            self._first_kids.insert(position, kid)


class _TreeNode(NamedTuple):
    """A node of a /Kids number tree with the page indexes reaching it."""

    node: DictionaryObject
    indexes: list[int]
    level: int


class _KidPages(NamedTuple):
    """A kid with the page indexes it is picked for."""

    kid: DictionaryObject
    indexes: list[int]


class _LeafPages(NamedTuple):
    """A leaf with its /Nums array, if it is one, and the page indexes it labels."""

    nums: ArrayObject | None
    leaf: DictionaryObject
    indexes: list[int]


def _get_labels_from_kids(number_tree: DictionaryObject, page_count: int) -> list[str]:
    """The labels of the first page_count pages of a /Kids number tree, as index2label returns
    them: each /Kids array is indexed once and each /Nums array read once for all its pages,
    and warnings about malformed kids are logged once per array, not once per page.
    """
    # The finders keep their /Kids arrays, so the ids stay unique.
    finders: dict[int, _FirstKidFinder] = {}
    leaves: dict[int, _LeafPages] = {}
    unmatched: list[int] = []
    nodes = [_TreeNode(number_tree, list(range(page_count)), 0)]
    while nodes:
        node = nodes.pop()
        kids = node.node["/Kids"].get_object()
        if not isinstance(kids, ArrayObject):
            logger.warning("Page label kids are not an array: %s", kids)
            unmatched.extend(node.indexes)
            continue
        if id(kids) not in finders:
            finders[id(kids)] = _FirstKidFinder(kids, page_count)
        finder = finders[id(kids)]
        kid_pages: dict[int, _KidPages] = {}
        for index in node.indexes:
            kid = finder.find(index)
            if isinstance(kid, Exception):
                raise kid
            if kid is None:
                unmatched.append(index)
            else:
                kid_pages.setdefault(id(kid), _KidPages(kid, [])).indexes.append(index)
        for kid, indexes in kid_pages.values():
            if not is_null_or_none(kid.get("/Kids", None)):
                # Same nesting limit as in index2label().
                if node.level + 1 == 100:  # pragma: no cover
                    raise NotImplementedError("Too deep nesting is not supported.")
                nodes.append(_TreeNode(kid, indexes, node.level + 1))
                continue
            # Leaves sharing a /Nums array are read together.
            nums = kid["/Nums"]
            if isinstance(nums, ArrayObject):
                leaves.setdefault(id(nums), _LeafPages(nums, kid, [])).indexes.extend(indexes)
            else:
                leaves.setdefault(id(kid), _LeafPages(None, kid, [])).indexes.extend(indexes)

    labels = [""] * page_count
    for leaf_nums, leaf, leaf_indexes in leaves.values():
        leaf_indexes.sort()
        leaf_labels = (
            _get_labels_for_page_indexes(leaf_nums, leaf_indexes)
            if leaf_nums is not None
            else [get_label_from_nums(leaf, index) for index in leaf_indexes]
        )
        for index, label in zip(leaf_indexes, leaf_labels, strict=True):
            labels[index] = label
    for index in sorted(unmatched):
        logger.warning("Could not reliably determine page label for %d.", index)
        labels[index] = str(index + 1)
    return labels


def _index2labels_page_by_page(pdf: Any, page_count: int) -> list[str]:
    return [index2label(pdf, index) for index in range(page_count)]


def index2labels(pdf: Any, page_count: int) -> list[str]:
    """The labels of the first page_count pages, as index2label returns them; number trees are
    read once for all pages instead of from the start for every page (pypdf #4321)."""
    if page_count <= 0:
        return []
    root = pdf.root_object
    if "/PageLabels" not in root:
        return _index2labels_page_by_page(pdf, page_count)
    number_tree = root["/PageLabels"].get_object()
    if not isinstance(number_tree, DictionaryObject):
        return _index2labels_page_by_page(pdf, page_count)
    if "/Nums" in number_tree:
        nums = number_tree["/Nums"]
        if not isinstance(nums, ArrayObject):
            return [get_label_from_nums(number_tree, index) for index in range(page_count)]
        return _get_labels_for_page_indexes(nums, range(page_count))
    if "/Kids" in number_tree and not isinstance(number_tree["/Kids"], NullObject):
        return _get_labels_from_kids(number_tree, page_count)
    return _index2labels_page_by_page(pdf, page_count)
