from __future__ import annotations

import heapq
import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from html import escape
from io import BytesIO
from typing import Any, BinaryIO, TextIO, TypeAlias, cast

from core_pdf import PdfDocument, PdfPage
from core_pdf._vendor.fontTools.agl import toUnicode
from core_pdf.impl.engine.layout.geometry import bbox_union
from core_pdf.impl.engine.spec.s_09_fonts.data.base_encodings import (
    MAC_ROMAN_ENCODING,
    STANDARD_ENCODING,
    WIN_ANSI_ENCODING,
)
from core_pdf.impl.engine.spec.s_09_fonts.data.core14 import FONT_DATA
from core_pdf.impl.exceptions import PdfError
from core_pdf.impl.primitives import PdfReference

PdfInput: TypeAlias = Any


def _pdfminer_reference_is_resolvable(document: PdfDocument, value: object) -> bool:
    if not document.xref_was_recovered:
        return True
    if isinstance(value, (list, tuple)):
        return all(_pdfminer_reference_is_resolvable(document, item) for item in value)
    if not isinstance(value, PdfReference):
        return True
    key = (value.object_number << 16) | value.generation_number
    entry = document.xref.get(key)
    if entry is None or not entry.in_use:
        return False
    return entry.object_stream is not None or document.xref_entry_matches_header(key, entry)


def _pdfminer_resolvable_pages(document: PdfDocument) -> Iterator[tuple[int, PdfPage]]:
    """Omit recovered pages whose content reference does not resolve as PDFMiner requires."""
    for page_index, page in enumerate(document.pages):
        if _pdfminer_reference_is_resolvable(document, page.contents):
            yield page_index, page


@dataclass(frozen=True, slots=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True, slots=True)
class TextCharacter:
    text: str
    bbox: Rect
    font_name: str | None = None
    font_size: float | None = None
    sequence: int | None = None


@dataclass(frozen=True, slots=True)
class TextSpan:
    text: str
    bbox: Rect
    characters: tuple[TextCharacter, ...] = ()
    font_name: str | None = None
    font_size: float | None = None
    sequence: int | None = None


def synthesize_characters(
    text: str, box: tuple[float, float, float, float]
) -> Iterator[tuple[str, tuple[float, float, float, float]]]:
    x0, y0, x1, y1 = box
    width = (x1 - x0) / max(1, len(text))
    for index, character in enumerate(text):
        yield character, (x0 + index * width, y0, x0 + (index + 1) * width, y1)


@dataclass(slots=True)
class LAParams:
    """pdfminer.six layout parameters accepted by the compatibility facade."""

    line_overlap: float = 0.5
    char_margin: float = 2.0
    line_margin: float = 0.5
    word_margin: float = 0.1
    boxes_flow: float | None = 0.5
    detect_vertical: bool = False
    all_texts: bool = False


class LTItem:
    def analyze(self, laparams: LAParams) -> None:
        del laparams


@dataclass(slots=True)
class LTComponent(LTItem):
    bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        self.x0, self.y0, self.x1, self.y1 = self.bbox
        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0

    def is_hoverlap(self, other: LTComponent) -> bool:
        return other.x0 <= self.x1 and self.x0 <= other.x1

    def is_voverlap(self, other: LTComponent) -> bool:
        return other.y0 <= self.y1 and self.y0 <= other.y1

    def hdistance(self, other: LTComponent) -> float:
        return (
            0.0
            if self.is_hoverlap(other)
            else min(abs(self.x0 - other.x1), abs(self.x1 - other.x0))
        )

    def vdistance(self, other: LTComponent) -> float:
        return (
            0.0
            if self.is_voverlap(other)
            else min(abs(self.y0 - other.y1), abs(self.y1 - other.y0))
        )

    def hoverlap(self, other: LTComponent) -> float:
        return (
            min(abs(self.x0 - other.x1), abs(self.x1 - other.x0))
            if self.is_hoverlap(other)
            else 0.0
        )

    def voverlap(self, other: LTComponent) -> float:
        return (
            min(abs(self.y0 - other.y1), abs(self.y1 - other.y0))
            if self.is_voverlap(other)
            else 0.0
        )


class LTText(LTItem):
    def get_text(self) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class LTAnno(LTText):
    text: str

    def get_text(self) -> str:
        return self.text


@dataclass(slots=True)
class LTChar(LTComponent, LTText):
    text: str = ""
    fontname: str | None = None
    size: float = 0.0

    def get_text(self) -> str:
        return self.text

    @property
    def font_name(self) -> str | None:
        return self.fontname

    @property
    def adv(self) -> float:
        return self.width

    @property
    def upright(self) -> bool:
        return self.width >= self.height

    @property
    def matrix(self) -> tuple[float, float, float, float, float, float]:
        return (1.0, 0.0, 0.0, 1.0, self.x0, self.y0)


@dataclass(slots=True)
class LTTextLine(LTComponent, LTText):
    _objs: list[LTText | LTChar] = field(default_factory=list)
    fragment_group: object | None = None

    def __iter__(self) -> Iterator[LTText | LTChar]:
        return iter(self._objs)

    def get_text(self) -> str:
        return "".join(item.get_text() for item in self._objs)


class LTTextLineHorizontal(LTTextLine):
    pass


class LTTextLineVertical(LTTextLine):
    pass


@dataclass(slots=True)
class LTTextBox(LTComponent, LTText):
    _objs: list[LTTextLine] = field(default_factory=list)

    def __iter__(self) -> Iterator[LTTextLine]:
        return iter(self._objs)

    def get_text(self) -> str:
        return "".join(line.get_text() for line in self._objs)


class LTTextBoxHorizontal(LTTextBox):
    pass


class LTTextBoxVertical(LTTextBox):
    pass


LTTextContainer = LTTextBox


@dataclass(slots=True)
class LTImage(LTComponent):
    name: str = ""
    stream: object | None = None


@dataclass(slots=True)
class LTFigure(LTComponent):
    name: str = ""
    _objs: list[LTItem] = field(default_factory=list)
    text_snippets: tuple[str, ...] = ()

    def __iter__(self) -> Iterator[LTItem]:
        return iter(self._objs)


@dataclass(slots=True)
class LTPage(LTComponent):
    pageid: int
    rotate: float = 0
    _objs: list[LTItem] = field(default_factory=list)

    def __iter__(self) -> Iterator[LTItem]:
        return iter(self._objs)


def _bbox(rect: Rect) -> tuple[float, float, float, float]:
    return (rect.x0, rect.y0, rect.x1, rect.y1)


def _characters(span: TextSpan) -> Iterable[TextCharacter]:
    if span.characters:
        return span.characters
    box = (span.bbox.x0, span.bbox.y0, span.bbox.x1, span.bbox.y1)
    return tuple(
        TextCharacter(
            text=character,
            bbox=Rect(*sub_box),
            font_name=span.font_name,
            font_size=span.font_size,
            sequence=span.sequence,
        )
        for character, sub_box in synthesize_characters(span.text, box)
    )


def _span_fragments(span: TextSpan, char_margin: float = 2.0) -> Iterable[TextSpan]:
    characters = tuple(_characters(span))
    if not characters:
        return (span,)
    fragments: list[TextSpan] = []
    current: list[TextCharacter] = []
    current_sequence: int | None = None
    sequence_run_length = 0
    for character in characters:
        vertical_fragment = False
        if current:
            previous = current[-1].bbox
            current_box = character.bbox
            x_overlap = min(previous.x1, current_box.x1) - max(previous.x0, current_box.x0)
            vertical_fragment = x_overlap > 0 and abs(previous.y0 - current_box.y0) > (
                0.5 * min(previous.height, current_box.height)
            )
            horizontal_gap = current_box.x0 - previous.x1
            horizontal_fragment = horizontal_gap > char_margin * max(
                previous.height, current_box.height
            )
        else:
            horizontal_fragment = False
        sequence_fragment = (
            character.sequence != current_sequence and sequence_run_length >= 2
            if current
            else False
        )
        if current and (sequence_fragment or vertical_fragment or horizontal_fragment):
            fragments.append(_character_span(current, span))
            current = []
        if character.sequence == current_sequence:
            sequence_run_length += 1
        else:
            current_sequence = character.sequence
            sequence_run_length = 1
        current.append(character)
    if current:
        fragments.append(_character_span(current, span))
    return tuple(fragments)


def _character_span(characters: list[TextCharacter], source: TextSpan) -> TextSpan:
    box = bbox_union(
        (character.bbox.x0, character.bbox.y0, character.bbox.x1, character.bbox.y1)
        for character in characters
    )
    if box is None:
        raise ValueError("cannot build a span from zero characters")
    return TextSpan(
        text="".join(character.text for character in characters),
        bbox=Rect(*box),
        characters=tuple(characters),
        font_name=source.font_name,
        font_size=source.font_size,
        sequence=characters[0].sequence,
    )


def _is_vertical_span(span: TextSpan) -> bool:
    characters = tuple(_characters(span))
    if len(characters) < 2:
        return False
    x_values = [character.bbox.x0 for character in characters]
    y_values = [character.bbox.y0 for character in characters]
    return max(x_values) - min(x_values) <= 1.0 and max(y_values) - min(y_values) > 1.0


def _make_line(items: list[LTChar], params: LAParams) -> LTTextLine:
    vertical = params.detect_vertical and len(items) > 1 and items[0].is_hoverlap(items[-1])
    line_type = LTTextLineVertical if vertical else LTTextLineHorizontal
    output: list[LTText | LTChar] = []
    previous: LTChar | None = None
    for item in items:
        if previous is not None and params.word_margin:
            margin = params.word_margin * max(item.width, item.height)
            if vertical:
                separated = item.y1 + margin < previous.y0
            else:
                separated = previous.x1 < item.x0 - margin
            if separated:
                output.append(LTAnno(" "))
        output.append(item)
        previous = item
    output.append(LTAnno("\n"))
    box = bbox_union(item.bbox for item in items) or items[0].bbox
    return line_type(box, output)


def _group_objects(chars: list[LTChar], params: LAParams) -> list[LTTextLine]:
    lines: list[LTTextLine] = []
    current: list[LTChar] = []
    previous: LTChar | None = None
    current_vertical = False
    for item in chars:
        if previous is None:
            current = [item]
            previous = item
            continue
        horizontal = (
            previous.is_voverlap(item)
            and min(previous.height, item.height) * params.line_overlap < previous.voverlap(item)
            and previous.hdistance(item) < max(previous.width, item.width) * params.char_margin
        )
        vertical = (
            params.detect_vertical
            and previous.is_hoverlap(item)
            and min(previous.width, item.width) * params.line_overlap < previous.hoverlap(item)
            and previous.vdistance(item) < max(previous.height, item.height) * params.char_margin
        )
        continues = (horizontal and not current_vertical) or (vertical and current_vertical)
        if len(current) > 1 and continues:
            current.append(item)
        elif len(current) > 1:
            lines.append(_make_line(current, params))
            current = [item]
        elif vertical and not horizontal:
            current.extend([item])
            current_vertical = True
        elif horizontal and not vertical:
            current.extend([item])
            current_vertical = False
        else:
            lines.append(_make_line(current, params))
            current = [item]
            current_vertical = False
        previous = item
    if current:
        lines.append(_make_line(current, params))
    return lines


def _lines_are_neighbors(first: LTTextLine, second: LTTextLine, ratio: float) -> bool:
    if isinstance(first, LTTextLineHorizontal) and isinstance(second, LTTextLineHorizontal):
        tolerance = ratio * first.height
        return (
            # ``LTTextLineHorizontal.find_neighbors`` first queries a spatial
            # plane over the line's own horizontal extent. Alignment alone is
            # insufficient: lines separated along x are never candidates.
            not (second.x1 <= first.x0 or first.x1 <= second.x0)
            and not (second.y1 <= first.y0 - tolerance or first.y1 + tolerance <= second.y0)
            and abs(second.height - first.height) <= tolerance
            and (
                abs(second.x0 - first.x0) <= tolerance
                or abs(second.x1 - first.x1) <= tolerance
                or abs((second.x0 + second.x1 - first.x0 - first.x1) / 2) <= tolerance
            )
        )
    if isinstance(first, LTTextLineVertical) and isinstance(second, LTTextLineVertical):
        tolerance = ratio * first.width
        return (
            # The vertical counterpart's plane query is restricted to the
            # line's own vertical extent.
            not (second.y1 <= first.y0 or first.y1 <= second.y0)
            and not (second.x1 <= first.x0 - tolerance or first.x1 + tolerance <= second.x0)
            and abs(second.width - first.width) <= tolerance
            and (
                abs(second.y0 - first.y0) <= tolerance
                or abs(second.y1 - first.y1) <= tolerance
                or abs((second.y0 + second.y1 - first.y0 - first.y1) / 2) <= tolerance
            )
        )
    return False


class _LayoutPlane:
    """Small identity-based spatial index matching pdfminer's layout plane."""

    __slots__ = ("bbox", "grid", "gridsize", "objects", "sequence")

    def __init__(self, bbox: tuple[float, float, float, float], gridsize: int = 50) -> None:
        self.bbox = bbox
        self.gridsize = gridsize
        self.sequence: list[LTComponent | _TextGroup] = []
        self.objects: dict[int, LTComponent | _TextGroup] = {}
        self.grid: dict[tuple[int, int], list[LTComponent | _TextGroup]] = {}

    def _range(self, bbox: tuple[float, float, float, float]) -> Iterator[tuple[int, int]]:
        x0, y0, x1, y1 = bbox
        left, bottom, right, top = self.bbox
        if x1 <= left or right <= x0 or y1 <= bottom or top <= y0:
            return
        x0 = max(left, x0)
        y0 = max(bottom, y0)
        x1 = min(right, x1)
        y1 = min(top, y1)
        for grid_y in range(int(y0) // self.gridsize, int(y1 + self.gridsize) // self.gridsize):
            for grid_x in range(int(x0) // self.gridsize, int(x1 + self.gridsize) // self.gridsize):
                yield grid_x, grid_y

    def add(self, item: LTComponent | _TextGroup) -> None:
        for key in self._range(item.bbox):
            self.grid.setdefault(key, []).append(item)
        self.sequence.append(item)
        self.objects[id(item)] = item

    def remove(self, item: LTComponent | _TextGroup) -> None:
        for key in self._range(item.bbox):
            bucket = self.grid.get(key)
            if bucket is not None:
                self.grid[key] = [candidate for candidate in bucket if candidate is not item]
        self.objects.pop(id(item), None)

    def __iter__(self) -> Iterator[LTComponent | _TextGroup]:
        return (item for item in self.sequence if id(item) in self.objects)

    def find(self, bbox: tuple[float, float, float, float]) -> Iterator[LTComponent | _TextGroup]:
        x0, y0, x1, y1 = bbox
        seen: set[int] = set()
        for key in self._range(bbox):
            for item in self.grid.get(key, ()):
                item_id = id(item)
                if item_id in seen or item_id not in self.objects:
                    continue
                seen.add(item_id)
                item_x0, item_y0, item_x1, item_y1 = item.bbox
                if item_x1 <= x0 or x1 <= item_x0 or item_y1 <= y0 or y1 <= item_y0:
                    continue
                yield item


def _group_lines(
    lines: list[LTTextLine],
    margin: float,
    page_bbox: tuple[float, float, float, float] | None = None,
) -> list[LTTextBox]:
    if not lines:
        return []
    plane_bbox = page_bbox or bbox_union(line.bbox for line in lines) or lines[0].bbox
    plane = _LayoutPlane(plane_bbox)
    for line in lines:
        plane.add(line)
    groups_by_line: dict[int, list[LTTextLine]] = {}
    for line in lines:
        tolerance = margin * (line.width if isinstance(line, LTTextLineVertical) else line.height)
        query = (
            (line.x0 - tolerance, line.y0, line.x1 + tolerance, line.y1)
            if isinstance(line, LTTextLineVertical)
            else (line.x0, line.y0 - tolerance, line.x1, line.y1 + tolerance)
        )
        members: list[LTTextLine] = [line]
        for candidate in plane.find(query):
            if not isinstance(candidate, LTTextLine) or not _lines_are_neighbors(
                line, candidate, margin
            ):
                continue
            members.append(candidate)
            previous = groups_by_line.pop(id(candidate), None)
            if previous is not None:
                members.extend(previous)
        unique_members = list({id(member): member for member in members}.values())
        for member in unique_members:
            groups_by_line[id(member)] = unique_members
    groups: list[list[LTTextLine]] = []
    seen: set[int] = set()
    for line in lines:
        members = groups_by_line.get(id(line))
        if members is None:
            continue
        group_key = id(members)
        if group_key in seen:
            continue
        seen.add(group_key)
        groups.append(members)
    boxes: list[LTTextBox] = []
    for members in groups:
        vertical = isinstance(members[0], LTTextLineVertical)
        members.sort(key=(lambda item: -item.x1) if vertical else (lambda item: -item.y1))
        box = bbox_union(item.bbox for item in members) or members[0].bbox
        box_type = LTTextBoxVertical if vertical else LTTextBoxHorizontal
        text_box = box_type(box, members)
        if not text_box.get_text().isspace():
            boxes.append(text_box)
    return boxes


@dataclass(slots=True)
class _TextGroup:
    children: list[LTTextBox | _TextGroup]
    bbox: tuple[float, float, float, float]
    vertical: bool = False


def _reading_order(
    boxes: list[LTTextBox],
    boxes_flow: float | None,
    page_bbox: tuple[float, float, float, float] | None = None,
) -> list[LTTextBox]:
    if boxes_flow is None:
        return sorted(
            boxes,
            key=lambda box: (
                (0, -box.x1, -box.y0)
                if isinstance(box, LTTextBoxVertical)
                else (1, -box.y0, box.x0)
            ),
        )
    if len(boxes) < 2:
        return boxes

    active: dict[int, LTTextBox | _TextGroup] = {id(box): box for box in boxes}
    plane_order = list(active)

    def area_gap(first: LTTextBox | _TextGroup, second: LTTextBox | _TextGroup) -> float:
        x0 = min(first.bbox[0], second.bbox[0])
        y0 = min(first.bbox[1], second.bbox[1])
        x1 = max(first.bbox[2], second.bbox[2])
        y1 = max(first.bbox[3], second.bbox[3])
        first_area = (first.bbox[2] - first.bbox[0]) * (first.bbox[3] - first.bbox[1])
        second_area = (second.bbox[2] - second.bbox[0]) * (second.bbox[3] - second.bbox[1])
        return (x1 - x0) * (y1 - y0) - first_area - second_area

    queue: list[tuple[bool, float, int, int]] = []
    for first_index, first in enumerate(boxes):
        first_id = id(first)
        for second in boxes[first_index + 1 :]:
            second_id = id(second)
            heapq.heappush(queue, (False, area_gap(first, second), first_id, second_id))
    while queue:
        skip_between, _distance, first_id, second_id = heapq.heappop(queue)
        if first_id not in active or second_id not in active:
            continue
        first = active[first_id]
        second = active[second_id]
        union = bbox_union((first.bbox, second.bbox)) or first.bbox
        query = union
        if page_bbox is not None:
            query = (
                max(query[0], page_bbox[0]),
                max(query[1], page_bbox[1]),
                min(query[2], page_bbox[2]),
                min(query[3], page_bbox[3]),
            )
        between = []
        if query[0] < query[2] and query[1] < query[3]:
            for item_id in plane_order:
                if item_id in {first_id, second_id} or item_id not in active:
                    continue
                item = active[item_id]
                if page_bbox is not None and (
                    item.bbox[2] <= page_bbox[0]
                    or page_bbox[2] <= item.bbox[0]
                    or item.bbox[3] <= page_bbox[1]
                    or page_bbox[3] <= item.bbox[1]
                ):
                    continue
                if not (
                    item.bbox[2] <= query[0]
                    or query[2] <= item.bbox[0]
                    or item.bbox[3] <= query[1]
                    or query[3] <= item.bbox[1]
                ):
                    between.append(item)
        if between and not skip_between:
            heapq.heappush(queue, (True, _distance, first_id, second_id))
            continue
        vertical = (
            isinstance(first, LTTextBoxVertical)
            or isinstance(second, LTTextBoxVertical)
            or isinstance(first, _TextGroup)
            and first.vertical
            or isinstance(second, _TextGroup)
            and second.vertical
        )
        group = _TextGroup([first, second], union, vertical)
        del active[first_id], active[second_id]
        group_id = id(group)
        for other_id in plane_order:
            if other_id not in active:
                continue
            other = active[other_id]
            heapq.heappush(queue, (False, area_gap(group, other), group_id, other_id))
        active[group_id] = group
        plane_order.append(group_id)

    root = next(iter(active.values()))

    def flatten(item: LTTextBox | _TextGroup) -> list[LTTextBox]:
        if isinstance(item, LTTextBox):
            return [item]
        if item.vertical:
            ordered = sorted(
                item.children,
                key=lambda child: (
                    -(1 + boxes_flow) * (child.bbox[0] + child.bbox[2])
                    - (1 - boxes_flow) * child.bbox[3]
                ),
            )
        else:
            ordered = sorted(
                item.children,
                key=lambda child: (
                    (1 - boxes_flow) * child.bbox[0]
                    - (1 + boxes_flow) * (child.bbox[1] + child.bbox[3])
                ),
            )
        return [box for child in ordered for box in flatten(child)]

    return flatten(root)


def _mapping_value(mapping: object, name: str) -> object | None:
    if not isinstance(mapping, dict):
        return None
    return next((value for key, value in mapping.items() if str(key) == name), None)


def _pdfminer_base_encoding_text(
    decoder: Any,
    char_code: int,
    base_encoding: str | None = None,
) -> str:
    if base_encoding is None:
        base_encoding = getattr(decoder, "base_encoding", None)
    table = {
        "MacRomanEncoding": MAC_ROMAN_ENCODING,
        "WinAnsiEncoding": WIN_ANSI_ENCODING,
    }.get(base_encoding, STANDARD_ENCODING)
    # pdfminer follows its Latin glyph-name database here rather than the
    # Unicode-oriented tables used by core-pdf.  WinAnsi's soft hyphen maps to
    # a space, while the five reserved bullet placeholders are undefined.
    if base_encoding == "WinAnsiEncoding":
        if char_code == 173:
            return " "
        if char_code in {127, 129, 141, 143, 144, 157}:
            return ""
    if base_encoding == "MacRomanEncoding" and char_code in {
        173,
        176,
        178,
        179,
        182,
        183,
        184,
        185,
        186,
        189,
        195,
        197,
        198,
        215,
    }:
        return ""
    return table[char_code]


def _pdfminer_to_unicode_text(glyph: Any, to_unicode: Any) -> str | None:
    mappings = getattr(to_unicode, "mappings", {})
    decoder = glyph.font_decoder
    if not getattr(decoder, "is_cid_font", False):
        return mappings.get(glyph.code_bytes)
    cid = glyph.cid
    if cid is None:
        return None
    for length in getattr(to_unicode, "decode_lengths", (1,)):
        if cid >= 1 << (length * 8):
            continue
        mapped = mappings.get(cid.to_bytes(length, "big"))
        if mapped is not None:
            return mapped
    return None


def _pdfminer_glyph_text(glyph: Any) -> str:
    if glyph.unicode_source == "actual_text" and glyph.text == "\ufeff":
        return ""
    if glyph.unicode_source == "actual_text" and glyph.alternates:
        return glyph.alternates[0]
    to_unicode = getattr(glyph.font_decoder, "to_unicode", None)
    decoder = glyph.font_decoder
    glyph_name = getattr(decoder, "encoding_differences", {}).get(glyph.char_code)
    if glyph_name and glyph_name.isdecimal():
        glyph_name = None
    glyph_name_text = toUnicode(glyph_name) if glyph_name else ""
    if to_unicode is not None and glyph.code_bytes:
        mapped = _pdfminer_to_unicode_text(glyph, to_unicode)
        if mapped is not None and len(mapped) <= 1:
            return mapped or f"(cid:{glyph.cid})"
        if glyph_name_text and (
            glyph.unicode_source == "encoding" or not getattr(decoder, "is_cid_font", False)
        ):
            return glyph_name_text
        if (
            glyph.unicode_source != "identity"
            and not getattr(decoder, "is_cid_font", False)
            and glyph.char_code is not None
            and (not glyph_name or getattr(decoder, "base_encoding", None) == "WinAnsiEncoding")
        ):
            base_encoding = getattr(decoder, "base_encoding", None)
            if base_encoding is None and str(_mapping_value(decoder.font, "Subtype")) == "TrueType":
                base_encoding = "WinAnsiEncoding"
            encoded = _pdfminer_base_encoding_text(decoder, glyph.char_code, base_encoding)
            if encoded:
                return encoded
        if getattr(decoder, "is_cid_font", False):
            return f"(cid:{glyph.cid})"
        if glyph_name and not glyph_name_text and not getattr(decoder, "is_type3", False):
            return f"(cid:{glyph.cid})"
        if glyph.unicode_source == "identity":
            # An explicit ToUnicode CMap is authoritative to PDFMiner. When
            # it omits a simple-font code and the encoding supplies no glyph
            # name, PDFMiner exposes the unresolved code instead of applying
            # Core's useful identity fallback.
            return f"(cid:{glyph.cid})"
    elif glyph_name_text and (
        glyph.unicode_source == "encoding" or not getattr(decoder, "is_cid_font", False)
    ):
        return glyph_name_text
    # pdfminer consults an explicit ToUnicode map first, then resolves a
    # simple-font encoding name through its glyph list.  Core's font engine
    # can recover useful Unicode from an embedded program when that name is
    # private (for example TeX's ``suppress``), but pdfminer exposes the
    # unresolved character as a CID marker instead.
    if (
        glyph_name
        and getattr(decoder, "is_type3", False)
        and glyph.char_code is not None
        and (base_text := _pdfminer_base_encoding_text(decoder, glyph.char_code))
    ):
        return base_text
    if glyph_name:
        return f"(cid:{glyph.cid})"
    if glyph.cid is None:
        return glyph.text
    if glyph.unicode_source == "identity" and (
        to_unicode is None or getattr(decoder, "is_cid_font", False)
    ):
        return f"(cid:{glyph.cid})"
    if glyph.unicode_source in {"fallback_nul", "undefined"}:
        base_encoding = getattr(decoder, "base_encoding", None)
        if base_encoding in {"MacRomanEncoding", "StandardEncoding", "WinAnsiEncoding"} and (
            glyph.char_code is not None
        ):
            base_text = _pdfminer_base_encoding_text(decoder, glyph.char_code)
            if base_text:
                return base_text
        return f"(cid:{glyph.cid})"
    if glyph.unicode_source == "encoding" and not glyph_name and glyph.char_code is not None:
        base_text = _pdfminer_base_encoding_text(decoder, glyph.char_code)
        return base_text or f"(cid:{glyph.cid})"
    if glyph.unicode_source == "truetype_cmap" and getattr(decoder, "to_unicode", None) is None:
        descendants = _mapping_value(getattr(decoder, "font", None), "DescendantFonts")
        descendant = descendants[0] if isinstance(descendants, list) and descendants else None
        system_info = _mapping_value(descendant, "CIDSystemInfo")
        registry = _mapping_value(system_info, "Registry")
        registry_data = getattr(registry, "data", b"")
        if isinstance(registry_data, bytes) and registry_data.strip() == b"PDFAUTOCAD":
            return f"(cid:{glyph.cid})"
    return glyph.text


def _legacy_ligature_overrides(
    glyphs: tuple[Any, ...],
) -> tuple[
    dict[
        int,
        tuple[
            str,
            tuple[float, float, float, float],
            tuple[float, float, float, float] | None,
        ],
    ],
    set[int],
]:
    ligatures = {
        "ff": "ﬀ",
        "fi": "ﬁ",
        "fl": "ﬂ",
        "ffi": "ﬃ",
        "ffl": "ﬄ",
    }
    overrides: dict[
        int,
        tuple[
            str,
            tuple[float, float, float, float],
            tuple[float, float, float, float] | None,
        ],
    ] = {}
    skipped: set[int] = set()
    for index, glyph in enumerate(glyphs):
        decoder = glyph.font_decoder
        to_unicode = getattr(decoder, "to_unicode", None)
        if to_unicode is not None and glyph.code_bytes:
            mapped = _pdfminer_to_unicode_text(glyph, to_unicode)
            if mapped is not None:
                cluster_id = dict(glyph.provenance or ()).get("cluster_id")
                cluster = [glyph]
                if cluster_id is not None:
                    for item in glyphs[index + 1 :]:
                        if dict(item.provenance or ()).get("cluster_id") != cluster_id:
                            break
                        cluster.append(item)
                box = bbox_union(item.advance_bbox for item in cluster) or glyph.advance_bbox
                overrides[id(glyph)] = (mapped, box, glyph.baseline)
                skipped.update(id(item) for item in cluster[1:])
                continue
        if glyph.char_code is None:
            continue
        glyph_name = getattr(decoder, "encoding_differences", {}).get(glyph.char_code)
        glyph_name_text = toUnicode(glyph_name) if glyph_name else ""
        if len(glyph_name_text) > 1:
            cluster = glyphs[index : index + len(glyph_name_text)]
            if len(cluster) == len(glyph_name_text) and all(
                item.seqno == glyph.seqno
                and item.code_bytes == glyph.code_bytes
                and item.char_code == glyph.char_code
                for item in cluster
            ):
                box = bbox_union(item.advance_bbox for item in cluster) or glyph.advance_bbox
                overrides[id(glyph)] = (glyph_name_text, box, glyph.baseline)
                skipped.update(id(item) for item in cluster[1:])
                continue
        base_table = {
            "MacRomanEncoding": MAC_ROMAN_ENCODING,
            "WinAnsiEncoding": WIN_ANSI_ENCODING,
        }.get(getattr(decoder, "base_encoding", None), STANDARD_ENCODING)
        encoded_ligature = (
            base_table[glyph.char_code] if 0 <= glyph.char_code < len(base_table) else ""
        )
        expected_ligature = ligatures.get(glyph_name, encoded_ligature)
        if expected_ligature not in ligatures.values():
            # A decomposed Unicode sequence is not sufficient evidence that
            # pdfminer would emit a legacy presentation-form ligature.  Most
            # commonly it came from /ActualText or a ToUnicode mapping, both
            # of which pdfminer exposes as ordinary characters. Recombine
            # only when either the explicit glyph name or the applicable base
            # encoding identifies the original code as a ligature.
            continue
        cluster: tuple[Any, ...] | None = None
        legacy_text: str | None = None
        # Try the longest standard ligature first.  The engine emits one
        # observation per Unicode scalar, while pdfminer emits one LTChar for
        # the original encoded character.
        for cluster_size in (3, 2):
            candidate = glyphs[index : index + cluster_size]
            if len(candidate) != cluster_size:
                continue
            decomposition = "".join(item.text for item in candidate)
            candidate_text = ligatures.get(decomposition)
            if candidate_text is None or candidate_text != expected_ligature:
                continue
            if any(
                item.seqno != glyph.seqno
                or item.code_bytes != glyph.code_bytes
                or item.char_code != glyph.char_code
                for item in candidate[1:]
            ):
                continue
            cluster = candidate
            legacy_text = candidate_text
            break
        if cluster is None or legacy_text is None:
            continue
        box = bbox_union(item.advance_bbox for item in cluster) or glyph.advance_bbox
        first_baseline = cluster[0].baseline
        last_baseline = cluster[-1].baseline
        baseline = (
            (
                first_baseline[0],
                first_baseline[1],
                last_baseline[2],
                last_baseline[3],
            )
            if first_baseline is not None and last_baseline is not None
            else None
        )
        overrides[id(glyph)] = (legacy_text, box, baseline)
        skipped.update(id(item) for item in cluster[1:])
    return overrides, skipped


def _pdfminer_literal_glyphs(
    glyphs: Iterable[Any],
) -> tuple[tuple[Any, ...], dict[int, tuple[float, float]]]:
    """Project parser-level byte loss from malformed literal-string escapes."""
    source = tuple(glyphs)
    projected: list[Any] = []
    offsets: dict[int, tuple[float, float]] = {}
    index = 0
    while index < len(source):
        glyph = source[index]
        provenance = dict(glyph.provenance) if glyph.provenance else {}
        compatibility_data = provenance.get("compatibility_data")
        if not isinstance(compatibility_data, bytes):
            projected.append(glyph)
            offsets[id(glyph)] = (0.0, 0.0)
            index += 1
            continue
        end = index + 1
        while end < len(source) and source[end].seqno == glyph.seqno:
            end += 1
        group = source[index:end]
        wanted = [item.code_bytes for item in glyph.font_decoder.decode_glyphs(compatibility_data)]
        wanted_index = 0
        offset_x = 0.0
        offset_y = 0.0
        matched_cluster_id: object | None = None
        matched_code_bytes: bytes | None = None
        for candidate in group:
            if wanted_index < len(wanted) and candidate.code_bytes == wanted[wanted_index]:
                projected.append(candidate)
                offsets[id(candidate)] = (offset_x, offset_y)
                candidate_provenance = dict(candidate.provenance or ())
                matched_cluster_id = candidate_provenance.get("cluster_id")
                matched_code_bytes = candidate.code_bytes
                wanted_index += 1
            elif (
                matched_cluster_id is not None
                and candidate.code_bytes == matched_code_bytes
                and dict(candidate.provenance or ()).get("cluster_id") == matched_cluster_id
            ):
                # A single encoded character can expand into several engine
                # observations (most commonly a decomposed ligature). They
                # are one source token, not parser-level byte loss. Retain the
                # complete cluster at the same correction offset so the
                # legacy ligature projection below can collapse it again.
                projected.append(candidate)
                offsets[id(candidate)] = (offset_x, offset_y)
            elif candidate.baseline is not None:
                baseline = candidate.baseline
                offset_x -= baseline[2] - baseline[0]
                offset_y -= baseline[3] - baseline[1]
        index = end
    return tuple(projected), offsets


def _pdfminer_builtin_width(glyph: Any) -> float | None:
    """Return pdfminer's built-in width for a widthless Standard-14 font."""
    decoder = glyph.font_decoder
    projected_text = _pdfminer_glyph_text(glyph)
    font_widths = _mapping_value(decoder.font, "Widths")
    first_char = _mapping_value(decoder.font, "FirstChar")
    if isinstance(font_widths, (list, tuple)) and isinstance(first_char, int):
        legacy_widths: list[float] = []
        recovered_malformed_token = False
        for width_value in font_widths:
            if isinstance(width_value, str):
                match = re.fullmatch(r"[^0-9+.-]+([+-]?(?:\d+(?:\.\d*)?|\.\d+))", width_value)
                if match is not None:
                    # PDFMiner's lexer separates a leading malformed control
                    # token from trailing numeric bytes. Its numeric coercion
                    # gives the former zero width and retains the latter as
                    # the following array entry.
                    legacy_widths.extend((0.0, float(match.group(1))))
                    recovered_malformed_token = True
                    continue
            try:
                legacy_widths.append(float(width_value))
            except (TypeError, ValueError):
                legacy_widths.append(0.0)
        width_index = glyph.char_code - first_char if glyph.char_code is not None else -1
        if recovered_malformed_token and 0 <= width_index < len(legacy_widths):
            return legacy_widths[width_index]
    glyph_name = getattr(decoder, "encoding_differences", {}).get(glyph.char_code)
    if (
        not decoder.is_cid_font
        and len(projected_text) == 1
        and ord(projected_text) < 32
        and glyph_name is not None
        and toUnicode(glyph_name).isspace()
        and toUnicode(glyph_name) != projected_text
    ):
        # PDFMiner's simple-font decoder can project a Differences entry to a
        # control character while its width lookup remains keyed by the
        # encoded glyph name. Such controls are emitted as zero-width layout
        # marks even when the PDF supplies a /Widths array for the source code.
        return 0.0
    if decoder.is_cid_font or decoder.is_type3:
        return None
    font = decoder.font
    if _mapping_value(font, "Widths") is not None:
        return None
    base_font = str(_mapping_value(font, "BaseFont") or "").split("+")[-1]
    entry = FONT_DATA.get(base_font)
    if not isinstance(entry, dict):
        return None
    widths = entry.get("widths")
    code = glyph.char_code
    if not isinstance(widths, dict) or code is None or not 0 <= code < 256:
        return None
    encoded_text = _pdfminer_base_encoding_text(decoder, code)
    width = widths.get(encoded_text)
    # pdfminer treats an encoded character absent from the base-font metrics
    # as a zero-width glyph rather than applying PDF's MissingWidth fallback.
    return 0.0 if width is None else float(width)


def _pdfminer_form_glyph_is_clipped(glyph: Any) -> bool:
    provenance = dict(glyph.provenance) if glyph.provenance else {}
    if int(provenance.get("xobject_depth", 0) or 0) <= 0:
        return False
    clip = provenance.get("clip_bbox")
    if not isinstance(clip, (tuple, list)) or len(clip) != 4:
        return False
    left, bottom, right, top = (float(value) for value in clip)
    # PDFMiner ignores page-content clipping during layout, but form
    # traversal still rejects a form whose transformed bounds are empty.
    return right <= left and top <= bottom


def _pdfminer_layout_origin(
    baseline: tuple[float, float, float, float],
    *,
    normalize_noise: bool,
) -> tuple[float, float]:
    origin_x, origin_y = baseline[0], baseline[1]
    if normalize_noise:
        return round(origin_x, 12), round(origin_y, 12)
    return origin_x, origin_y


def _pdfminer_rotated_text_matrix(
    origin_x: float,
    origin_y: float,
    matrix: tuple[float, float, float, float],
    rotation: int,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float, float, float]:
    matrix_a, matrix_b, matrix_c, matrix_d = matrix
    if rotation == 90:
        return origin_y, page_width - origin_x, matrix_b, -matrix_a, matrix_d, -matrix_c
    if rotation == 180:
        return (
            page_width - origin_x,
            page_height - origin_y,
            -matrix_a,
            -matrix_b,
            -matrix_c,
            -matrix_d,
        )
    if rotation == 270:
        return page_height - origin_y, origin_x, -matrix_b, matrix_a, -matrix_d, matrix_c
    return origin_x, origin_y, matrix_a, matrix_b, matrix_c, matrix_d


def _pdfminer_layout_figure_box(
    figure_box: tuple[float, float, float, float],
    media_box: tuple[float, float, float, float],
    rotation: int,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = figure_box
    media_left, media_bottom, _media_right, _media_top = media_box
    x0 -= media_left
    x1 -= media_left
    y0 -= media_bottom
    y1 -= media_bottom
    if rotation == 90:
        return (y0, page_width - x1, y1, page_width - x0)
    if rotation == 180:
        return (
            page_width - x1,
            page_height - y1,
            page_width - x0,
            page_height - y0,
        )
    if rotation == 270:
        return (page_height - y1, x0, page_height - y0, x1)
    return (x0, y0, x1, y1)


def extract_pages(  # noqa: C901
    pdf_file: PdfInput,
    password: str = "",
    page_numbers: Iterable[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    laparams: LAParams | None = None,
) -> Iterator[LTPage]:
    """Yield pdfminer.six-shaped pages using core-pdf extraction evidence."""
    del caching
    params = laparams or LAParams()
    selected = set(page_numbers) if page_numbers is not None else None
    # pdfminer's fallback xref loader stops at the first trailer it encounters.
    # Keep the engine's default all-revision recovery for native callers, while
    # selecting the legacy recovery policy for this compatibility projection.
    document = PdfDocument.open(
        pdf_file,
        password=password,
        recovery_scan_all_revisions=False,
        legacy_pdfminer_text_operators=True,
    )
    try:
        yielded = 0
        for page_index, page in _pdfminer_resolvable_pages(document):
            if selected is not None and page_index not in selected:
                continue
            if maxpages and yielded >= maxpages:
                break
            page_width = abs(page.width)
            page_height = abs(page.height)
            chars: list[LTChar] = []
            products = page.get_page_program().products
            compatibility_glyphs: list[Any] = []
            for glyph in products.glyphs:
                provenance = dict(glyph.provenance) if glyph.provenance else {}
                underlying = provenance.get("compatibility_glyphs")
                if glyph.unicode_source == "actual_text" and isinstance(underlying, tuple):
                    compatibility_glyphs.extend(underlying)
                else:
                    compatibility_glyphs.append(glyph)
            projected_glyphs, literal_offsets = _pdfminer_literal_glyphs(compatibility_glyphs)
            ligatures, skipped_ligature_parts = _legacy_ligature_overrides(projected_glyphs)
            pdfminer_offsets: dict[int, tuple[float, float]] = {}
            correction_x = 0.0
            correction_y = 0.0
            for glyph_index, glyph in enumerate(projected_glyphs):
                baseline = glyph.baseline
                literal_x, literal_y = literal_offsets.get(id(glyph), (0.0, 0.0))
                pdfminer_offsets[id(glyph)] = (
                    correction_x + literal_x,
                    correction_y + literal_y,
                )
                if glyph_index + 1 >= len(projected_glyphs):
                    continue
                following = projected_glyphs[glyph_index + 1]
                following_baseline = following.baseline
                provenance = dict(glyph.provenance) if glyph.provenance else {}
                following_provenance = dict(following.provenance) if following.provenance else {}
                text_matrix = provenance.get("text_matrix")
                along_is_x = not (
                    isinstance(text_matrix, (tuple, list))
                    and len(text_matrix) == 4
                    and abs(float(text_matrix[1])) > abs(float(text_matrix[0]))
                )
                continuous = (
                    baseline is not None
                    and following_baseline is not None
                    and (
                        abs(baseline[2] - following_baseline[0]) <= 1e-6
                        if along_is_x
                        else abs(baseline[3] - following_baseline[1]) <= 1e-6
                    )
                    and provenance.get("line_matrix_origin")
                    == following_provenance.get("line_matrix_origin")
                )
                if not continuous:
                    correction_x = 0.0
                    correction_y = 0.0
                    continue
                builtin_width = _pdfminer_builtin_width(glyph)
                if builtin_width is not None:
                    width_code = (
                        glyph.cid
                        if getattr(glyph.font_decoder, "is_cid_font", False)
                        else glyph.char_code
                    )
                    source_width = (
                        float(glyph.font_decoder.glyph_width(width_code))
                        if width_code is not None
                        else 0.0
                    )
                    retained_ratio = builtin_width / source_width if source_width else 0.0
                    correction_x -= (baseline[2] - baseline[0]) * (1.0 - retained_ratio)
                    correction_y -= (baseline[3] - baseline[1]) * (1.0 - retained_ratio)
                if glyph.seqno == following.seqno:
                    continue
            runs = sorted(products.runs, key=lambda run: run.seqno)
            run_sequences = [run.seqno for run in runs]
            figure_chars: dict[tuple[object, ...], list[tuple[LTChar, int]]] = {}
            figure_boxes: dict[tuple[object, ...], tuple[float, float, float, float]] = {}
            figure_depths: dict[tuple[object, ...], int] = {}
            figure_identifiers: dict[tuple[object, ...], object] = {}
            form_ancestor_boxes: dict[tuple[object, ...], tuple[float, float, float, float]] = {}
            try:
                page_annotations = page.get_annotations()
            except (PdfError, ValueError):
                page_annotations = ()
            annotation_boxes = tuple(
                tuple(annotation.rect)
                for annotation in page_annotations
                if annotation.rect is not None
            )
            vertical_positions: dict[tuple[str | None, int], tuple[float, int]] = {}
            for glyph_index, glyph in enumerate(projected_glyphs):
                if _pdfminer_form_glyph_is_clipped(glyph):
                    continue
                run_index = bisect_right(run_sequences, glyph.seqno) - 1
                if id(glyph) in skipped_ligature_parts:
                    continue
                if not glyph.text:
                    continue
                ligature = ligatures.get(id(glyph))
                x0, y0, x1, y1 = ligature[1] if ligature is not None else glyph.advance_bbox
                baseline = ligature[2] if ligature is not None else glyph.baseline
                offset_x, offset_y = pdfminer_offsets.get(id(glyph), (0.0, 0.0))
                x0 += offset_x
                x1 += offset_x
                y0 += offset_y
                y1 += offset_y
                if baseline is not None and (offset_x or offset_y):
                    baseline = (
                        baseline[0] + offset_x,
                        baseline[1] + offset_y,
                        baseline[2] + offset_x,
                        baseline[3] + offset_y,
                    )
                text = ligature[0] if ligature is not None else _pdfminer_glyph_text(glyph)
                if not text:
                    continue
                effective_font_size = glyph.effective_font_size or glyph.font_size
                effective_font_height = glyph.effective_font_height or effective_font_size
                if (
                    glyph.baseline is not None
                    and glyph.font_size > 0
                    and not glyph.effective_font_size
                    and x1 - x0 >= glyph.font_size * 0.8
                    and y1 - y0 <= glyph.font_size * 0.1
                ):
                    baseline_x = glyph.baseline[0]
                    key = (glyph.font_name, round(baseline_x))
                    anchor, position = vertical_positions.get(key, (glyph.baseline[1], 0))
                    baseline_y = anchor - position * glyph.font_size
                    vertical_positions[key] = (anchor, position + 1)
                    x0 = baseline_x - glyph.font_size * 0.5
                    x1 = baseline_x + glyph.font_size * 0.5
                    y0 = baseline_y - glyph.font_size * 0.88
                    y1 = y0 + glyph.font_size
                # PDF text size precedes the text matrix, so it can be much larger than the
                # effective glyph size after horizontal scaling. Recover the transformed size
                # from core's baseline advance; the advance box already has pdfminer's x bounds.
                width_code = (
                    glyph.cid
                    if getattr(glyph.font_decoder, "is_cid_font", False)
                    else glyph.char_code
                    if glyph.char_code is not None
                    else glyph.cid
                )
                width_lookup = getattr(glyph.font_decoder, "glyph_width", None)
                if (
                    glyph.rotation_angle % 180 == 0
                    and baseline is not None
                    and not glyph.effective_font_size
                    and width_code is not None
                    and callable(width_lookup)
                ):
                    normalized_width = float(width_lookup(width_code)) * 0.001
                    if normalized_width > 0:
                        baseline_x0, baseline_y0, baseline_x1, baseline_y1 = baseline
                        baseline_length = (
                            (baseline_x1 - baseline_x0) ** 2 + (baseline_y1 - baseline_y0) ** 2
                        ) ** 0.5
                        effective_font_size = baseline_length / normalized_width
                normalized_width = 0.0
                if width_code is not None and callable(width_lookup):
                    normalized_width = float(width_lookup(width_code)) * 0.001
                    builtin_width = _pdfminer_builtin_width(glyph)
                    if builtin_width is not None:
                        normalized_width = builtin_width * 0.001
                    base_font = str(_mapping_value(glyph.font_decoder.font, "BaseFont"))
                    glyph_name = getattr(glyph.font_decoder, "encoding_differences", {}).get(
                        glyph.char_code
                    )
                    if (
                        glyph_name
                        and _mapping_value(glyph.font_decoder.font, "Widths") is None
                        and base_font.split("+")[-1] in {"Symbol", "ZapfDingbats"}
                    ):
                        # pdfminer indexes built-in Symbol/Zapf metrics by its
                        # legacy encoded character keys. A Differences entry
                        # resolves to Unicode and therefore has no built-in
                        # width unless /Widths explicitly supplies one.
                        normalized_width = 0.0
                orientation = glyph.rotation_angle % 360
                glyph_provenance = dict(glyph.provenance) if glyph.provenance else {}
                text_matrix = glyph_provenance.get("text_matrix")
                horizontal_scale = float(glyph_provenance.get("horizontal_scale", 100.0)) * 0.01
                coordinates_in_layout_space = False
                if (
                    getattr(glyph.font_decoder, "is_vertical", False)
                    and baseline is not None
                    and width_code is not None
                ):
                    metric = glyph.font_decoder.vertical_metrics.get(
                        width_code,
                        (
                            glyph.font_decoder.default_vertical_width,
                            glyph.font_decoder.glyph_width(width_code) / 2.0,
                            glyph.font_decoder.default_vertical_origin_y,
                        ),
                    )
                    # The engine advances an entire text-show array in bulk,
                    # while PDFMiner advances one token at a time. Normalize
                    # the resulting sub-picopoint accumulation noise before
                    # applying vertical displacement metrics; otherwise two
                    # mathematically touching boxes can miss by one ULP.
                    pdfminer_origin = glyph_provenance.get("pdfminer_origin")
                    if isinstance(pdfminer_origin, (tuple, list)) and len(pdfminer_origin) == 2:
                        origin_x, origin_y = (float(value) for value in pdfminer_origin)
                        origin_x += offset_x
                        origin_y += offset_y
                        if int(page.rotation) % 360:
                            origin_x = round(origin_x, 12)
                            origin_y = round(origin_y, 12)
                    else:
                        origin_x, origin_y = _pdfminer_layout_origin(
                            baseline,
                            # Core advances complete show arrays in bulk whereas
                            # PDFMiner updates the vertical cursor token by token.
                            normalize_noise=effective_font_size == glyph.font_size,
                        )
                    # PDFMiner applies the vertical displacement and advance
                    # as a local LTChar rectangle before transforming it.  In
                    # particular, the character origin is not the lower edge:
                    # a normal vertical advance extends down from ``v1y``.
                    matrix_a, matrix_b, matrix_c, matrix_d = (float(value) for value in text_matrix)
                    local_font_size = glyph.font_size
                    local_left = -float(metric[1]) * local_font_size * 0.001
                    local_top = (1000.0 - float(metric[2])) * local_font_size * 0.001 + float(
                        glyph_provenance.get("text_rise", 0.0)
                    )
                    local_advance = (
                        -float(metric[0]) * local_font_size * 0.001 * horizontal_scale
                    )
                    corners = tuple(
                        (
                            local_horizontal * matrix_a + local_vertical * matrix_c + origin_x,
                            local_horizontal * matrix_b + local_vertical * matrix_d + origin_y,
                        )
                        for local_horizontal in (local_left, local_left + local_font_size)
                        for local_vertical in (local_top + local_advance, local_top)
                    )
                    x0 = min(point[0] for point in corners)
                    y0 = min(point[1] for point in corners)
                    x1 = max(point[0] for point in corners)
                    y1 = max(point[1] for point in corners)
                    effective_font_height = x1 - x0
                    line_origin = glyph_provenance.get("line_matrix_origin")
                    if (
                        not (
                            isinstance(pdfminer_origin, (tuple, list)) and len(pdfminer_origin) == 2
                        )
                        and isinstance(line_origin, (tuple, list))
                        and len(line_origin) == 2
                    ):
                        matrix_a, matrix_b, matrix_c, matrix_d = (
                            float(value) for value in text_matrix
                        )
                        determinant = matrix_a * matrix_d - matrix_b * matrix_c
                        if determinant:
                            translate_x, translate_y = (float(value) for value in line_origin)
                            baseline_x0, baseline_y0, baseline_x1, baseline_y1 = baseline
                            advance_x = baseline_x1 - baseline_x0
                            advance_y = baseline_y1 - baseline_y0
                            local_advance = (
                                -matrix_b * advance_x + matrix_a * advance_y
                            ) / determinant
                            half_width = float(metric[1]) * glyph.font_size * 0.001
                            local_top = (
                                1000.0 - float(metric[2])
                            ) * glyph.font_size * 0.001 + float(
                                glyph_provenance.get("text_rise", 0.0)
                            )
                            corners = tuple(
                                (
                                    matrix_a * local_horizontal
                                    + matrix_c * local_vertical
                                    + translate_x,
                                    matrix_b * local_horizontal
                                    + matrix_d * local_vertical
                                    + translate_y,
                                )
                                for local_horizontal in (
                                    -half_width,
                                    -half_width + glyph.font_size,
                                )
                                for local_vertical in (local_top + local_advance, local_top)
                            )
                            x0 = min(point[0] for point in corners)
                            y0 = min(point[1] for point in corners)
                            x1 = max(point[0] for point in corners)
                            y1 = max(point[1] for point in corners)
                elif (
                    baseline is not None
                    and isinstance(text_matrix, (tuple, list))
                    and len(text_matrix) == 4
                ):
                    matrix_a, matrix_b, matrix_c, matrix_d = (float(value) for value in text_matrix)
                    # A horizontal show immediately following vertical writing
                    # resumes at the vertical cursor. Normalize only that
                    # hand-off; ordinary horizontal origins must retain
                    # PDFMiner's native floating-point arithmetic.
                    pdfminer_origin = glyph_provenance.get("pdfminer_origin")
                    if isinstance(pdfminer_origin, (tuple, list)) and len(pdfminer_origin) == 2:
                        origin_x, origin_y = (float(value) for value in pdfminer_origin)
                        origin_x += offset_x
                        origin_y += offset_y
                    else:
                        origin_x, origin_y = _pdfminer_layout_origin(
                            baseline,
                            normalize_noise=(
                                bool(glyph_index)
                                and projected_glyphs[glyph_index - 1].font_decoder.is_vertical
                            ),
                        )
                    descent_scale = 0.001
                    descent_value = float(getattr(glyph.font_decoder, "descent", -200.0))
                    base_font = str(
                        _mapping_value(glyph.font_decoder.font, "BaseFont") or ""
                    ).split("+")[-1]
                    builtin_metrics = FONT_DATA.get(base_font)
                    if (
                        not getattr(glyph.font_decoder, "is_cid_font", False)
                        and not getattr(glyph.font_decoder, "is_type3", False)
                        and isinstance(builtin_metrics, dict)
                        and isinstance(builtin_metrics.get("props"), dict)
                    ):
                        builtin_descent = builtin_metrics["props"].get("Descent")
                        if isinstance(builtin_descent, (int, float)):
                            descent_value = float(builtin_descent)
                    if getattr(glyph.font_decoder, "is_type3", False):
                        font_matrix = _mapping_value(glyph.font_decoder.font, "FontMatrix")
                        if isinstance(font_matrix, (tuple, list)) and len(font_matrix) == 6:
                            font_matrix_b = float(font_matrix[1])
                            font_matrix_d = float(font_matrix[3])
                            descent_scale = font_matrix_b + font_matrix_d
                        descriptor = _mapping_value(glyph.font_decoder.font, "FontDescriptor")
                        descriptor_bbox = _mapping_value(descriptor, "FontBBox")
                        if descriptor is not None:
                            # PDFType3Font derives ascent/descent from the
                            # descriptor's FontBBox and treats a missing or
                            # malformed box as all zeros. It does not fall
                            # back to the Type 3 font dictionary in this case.
                            descent_value = (
                                min(float(descriptor_bbox[1]), float(descriptor_bbox[3]))
                                if isinstance(descriptor_bbox, (tuple, list))
                                and len(descriptor_bbox) == 4
                                else 0.0
                            )
                        else:
                            font_bbox = _mapping_value(glyph.font_decoder.font, "FontBBox")
                            if isinstance(font_bbox, (tuple, list)) and len(font_bbox) == 4:
                                descent_value = min(float(font_bbox[1]), float(font_bbox[3]))
                    text_rise = float(glyph_provenance.get("text_rise", 0.0))
                    descent = descent_value * descent_scale * glyph.font_size + text_rise
                    # ``LTChar`` uses the font descent only to anchor horizontal
                    # glyphs; its box is always exactly one text-space unit tall.
                    # FontBBox/ascent describes ink, not pdfminer's layout box.
                    top = descent + glyph.font_size
                    advance = normalized_width * horizontal_scale * glyph.font_size
                    media_left, media_bottom, _media_right, _media_top = page.media_box
                    layout_origin_x = origin_x - media_left
                    layout_origin_y = origin_y - media_bottom
                    page_rotation = int(page.rotation) % 360
                    (
                        layout_origin_x,
                        layout_origin_y,
                        matrix_a,
                        matrix_b,
                        matrix_c,
                        matrix_d,
                    ) = _pdfminer_rotated_text_matrix(
                        layout_origin_x,
                        layout_origin_y,
                        (matrix_a, matrix_b, matrix_c, matrix_d),
                        page_rotation,
                        page_width,
                        page_height,
                    )
                    corners = tuple(
                        (
                            along * matrix_a + vertical * matrix_c + layout_origin_x,
                            along * matrix_b + vertical * matrix_d + layout_origin_y,
                        )
                        for along in (0.0, advance)
                        for vertical in (descent, top)
                    )
                    x0 = min(point[0] for point in corners)
                    y0 = min(point[1] for point in corners)
                    x1 = max(point[0] for point in corners)
                    y1 = max(point[1] for point in corners)
                    effective_font_height = x1 - x0 if orientation % 180 else y1 - y0
                    coordinates_in_layout_space = True
                elif orientation == 0:
                    if normalized_width > 0:
                        x1 = x0 + normalized_width * effective_font_size
                    y1 = y0 + effective_font_height
                elif orientation == 90:
                    x0 = x1 - effective_font_height
                    if normalized_width > 0:
                        y1 = y0 + normalized_width * effective_font_size
                elif orientation == 180:
                    if normalized_width > 0:
                        x0 = x1 - normalized_width * effective_font_size
                    y0 = y1 - effective_font_height
                elif orientation == 270:
                    x1 = x0 + effective_font_height
                    if normalized_width > 0:
                        y0 = y1 - normalized_width * effective_font_size
                # PDFMiner places the media-box lower-left at layout-space
                # (0, 0). Core's canonical geometry remains in PDF user space,
                # so normalize non-zero and negative media-box origins here.
                if not coordinates_in_layout_space:
                    media_left, media_bottom, _media_right, _media_top = page.media_box
                    x0 -= media_left
                    x1 -= media_left
                    y0 -= media_bottom
                    y1 -= media_bottom
                    rotation = int(page.rotation) % 360
                    if rotation == 90:
                        x0, y0, x1, y1 = y0, page_width - x1, y1, page_width - x0
                    elif rotation == 180:
                        x0, y0, x1, y1 = (
                            page_width - x1,
                            page_height - y1,
                            page_width - x0,
                            page_height - y0,
                        )
                    elif rotation == 270:
                        x0, y0, x1, y1 = page_height - y1, x0, page_height - y0, x1
                character = LTChar(
                    (x0, y0, x1, y1),
                    text,
                    glyph.font_name,
                    effective_font_height,
                )
                provenance = (
                    dict(glyph.provenance)
                    if glyph.provenance
                    else (dict(runs[run_index].provenance) if run_index >= 0 else {})
                )
                xobject_depth = int(provenance.get("xobject_depth", 0) or 0)
                if xobject_depth <= 0:
                    chars.append(character)
                    continue
                clip_bbox = provenance.get("clip_bbox")
                if isinstance(clip_bbox, (tuple, list)) and len(clip_bbox) == 4:
                    clip_left, clip_bottom, clip_right, clip_top = (
                        float(value) for value in clip_bbox
                    )
                    clip_area = max(0.0, clip_right - clip_left) * max(0.0, clip_top - clip_bottom)
                    if clip_area > 0 and any(
                        max(0.0, min(clip_right, right) - max(clip_left, left))
                        * max(0.0, min(clip_top, top) - max(clip_bottom, bottom))
                        / clip_area
                        > 0.5
                        for left, bottom, right, top in annotation_boxes
                    ):
                        continue
                stream_order = int(provenance.get("stream_order", 0) or 0)
                layout_form_id = provenance.get("layout_form_id")
                layout_bbox = provenance.get("layout_form_bbox")
                if isinstance(layout_form_id, tuple):
                    for ancestor_index, ancestor_entry in enumerate(layout_form_id, start=1):
                        if (
                            isinstance(ancestor_entry, tuple)
                            and len(ancestor_entry) == 2
                            and isinstance(ancestor_entry[1], (tuple, list))
                            and len(ancestor_entry[1]) == 4
                        ):
                            form_ancestor_boxes[layout_form_id[:ancestor_index]] = tuple(
                                float(value) for value in ancestor_entry[1]
                            )
                if isinstance(layout_bbox, (tuple, list)) and len(layout_bbox) == 4:
                    resolved_layout_bbox = tuple(float(value) for value in layout_bbox)
                    figure_key: tuple[object, ...] = (
                        "form",
                        layout_form_id if layout_form_id is not None else stream_order,
                        *resolved_layout_bbox,
                    )
                    figure_identifiers[figure_key] = layout_form_id
                    figure_boxes[figure_key] = resolved_layout_bbox
                else:
                    figure_key = ("stream", stream_order)
                    layout_bbox = provenance.get("clip_bbox")
                figure_chars.setdefault(figure_key, []).append((character, glyph.seqno))
                figure_depths[figure_key] = xobject_depth
                if isinstance(layout_bbox, (tuple, list)) and len(layout_bbox) == 4:
                    figure_boxes[figure_key] = tuple(float(value) for value in layout_bbox)
            lines = _group_objects(chars, params)
            empty_lines = [line for line in lines if line.get_text().isspace()]
            layout_width, layout_height = (
                (page_height, page_width) if int(page.rotation) % 180 else (page_width, page_height)
            )
            boxes: list[LTItem] = list(
                _reading_order(
                    _group_lines(
                        [line for line in lines if not line.get_text().isspace()],
                        params.line_margin,
                        (0.0, 0.0, layout_width, layout_height),
                    ),
                    params.boxes_flow,
                    (0.0, 0.0, layout_width, layout_height),
                )
            )
            boxes.extend(empty_lines)
            # PDFMiner inserts layout children only for marks that reach its
            # device. Graphics-state and clipping records are engine
            # provenance, not LTItems, and therefore cannot delimit figure
            # text during recursive extraction.
            drawing_sequences_by_depth: dict[int, list[int]] = {}
            for drawing in products.drawings:
                if drawing.kind in {
                    "clip",
                    "group-begin",
                    "group-end",
                    "state-push",
                    "state-pop",
                }:
                    continue
                drawing_sequences_by_depth.setdefault(drawing.xobject_depth, []).append(
                    drawing.seqno
                )
            for sequences in drawing_sequences_by_depth.values():
                sequences.sort()
            figures: list[tuple[LTFigure, int, object]] = []

            for figure_index, (figure_key, entries) in enumerate(figure_chars.items()):
                drawing_sequences = drawing_sequences_by_depth.get(figure_depths[figure_key], [])
                snippets: list[str] = []
                current: list[str] = []
                previous_sequence: int | None = None
                for character, sequence in entries:
                    if previous_sequence is not None and bisect_left(
                        drawing_sequences, sequence
                    ) > bisect_right(drawing_sequences, previous_sequence):
                        snippets.append("".join(current))
                        current = []
                    current.append(character.get_text())
                    previous_sequence = sequence
                if current:
                    snippets.append("".join(current))
                merged_snippets: list[str] = []
                for snippet in snippets:
                    if (
                        merged_snippets
                        and snippet
                        and not any(character.isalnum() for character in snippet)
                        and not any(character.isalnum() for character in merged_snippets[-1])
                        and merged_snippets[-1][-1] == snippet[0]
                    ):
                        merged_snippets[-1] += snippet
                    else:
                        merged_snippets.append(snippet)
                snippets = merged_snippets
                figure_box = figure_boxes.get(figure_key) or bbox_union(
                    character.bbox for character, _ in entries
                )
                if figure_box is not None:
                    x0, y0, x1, y1 = _pdfminer_layout_figure_box(
                        figure_box,
                        tuple(page.media_box),
                        int(page.rotation) % 360,
                        page_width,
                        page_height,
                    )
                    figures.append(
                        (
                            LTFigure(
                                (x0, y0, x1, y1),
                                f"Form{figure_index}",
                                [character for character, _ in entries],
                                tuple(snippets),
                            ),
                            figure_depths[figure_key],
                            figure_identifiers.get(figure_key),
                        )
                    )
            represented_identifiers = {identifier for _figure, _depth, identifier in figures}
            for identifier, ancestor_box in form_ancestor_boxes.items():
                if identifier in represented_identifiers:
                    continue
                figures.append(
                    (
                        LTFigure(
                            _pdfminer_layout_figure_box(
                                ancestor_box,
                                tuple(page.media_box),
                                int(page.rotation) % 360,
                                page_width,
                                page_height,
                            ),
                            f"Form{len(figures)}",
                            [],
                            (),
                        ),
                        len(identifier),
                        identifier,
                    )
                )
            for figure, depth, identifier in figures:
                parent_identifier = (
                    identifier[:-1]
                    if isinstance(identifier, tuple) and len(identifier) > 1
                    else None
                )
                parent = min(
                    (
                        candidate
                        for candidate, candidate_depth, candidate_identifier in figures
                        if candidate_depth == depth - 1
                        and (
                            candidate_identifier == parent_identifier
                            if parent_identifier is not None
                            else candidate.x0 <= figure.x0
                            and candidate.y0 <= figure.y0
                            and candidate.x1 >= figure.x1
                            and candidate.y1 >= figure.y1
                        )
                    ),
                    key=lambda candidate: candidate.width * candidate.height,
                    default=None,
                )
                if parent is None:
                    boxes.append(figure)
                else:
                    parent._objs.append(figure)
            yield LTPage(
                (0.0, 0.0, layout_width, layout_height),
                page.page_number,
                page.rotation,
                boxes,
            )
            yielded += 1
    finally:
        document.close()


def extract_text(
    pdf_file: PdfInput,
    password: str = "",
    page_numbers: Iterable[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    codec: str = "utf-8",
    laparams: LAParams | None = None,
) -> str:
    """Return text with pdfminer.six-compatible high-level semantics."""
    del codec
    pages = [
        "\n".join(item.get_text() for item in page if isinstance(item, LTText))
        for page in extract_pages(pdf_file, password, page_numbers, maxpages, caching, laparams)
    ]
    return "".join(page_text + ("\n" if page_text else "") + "\f" for page_text in pages)


def extract_text_to_fp(
    inf: BinaryIO | PdfInput,
    outfp: TextIO | BinaryIO,
    output_type: str = "text",
    codec: str = "utf-8",
    laparams: LAParams | None = None,
    maxpages: int = 0,
    page_numbers: Iterable[int] | None = None,
    password: str = "",
    **kwargs: Any,
) -> None:
    """Write locally extracted text to a file-like object.

    Text, XML, and HTML output are supported locally. HOCR and tag output are
    intentionally rejected until their accessibility-specific semantics are
    mapped to core-pdf's structured serializers.
    """
    del kwargs
    if output_type == "text":
        output = extract_text(inf, password, page_numbers, maxpages, True, codec, laparams)
    elif output_type in {"xml", "html"}:
        output = _structured_output(inf, output_type, password, page_numbers, maxpages, laparams)
    else:
        raise ValueError(f"unsupported pdfminer output_type: {output_type}")
    if isinstance(outfp, (BytesIO,)):
        outfp.write(output.encode(codec))
    else:
        cast(TextIO, outfp).write(output)


def _structured_output(
    source: BinaryIO | PdfInput,
    output_type: str,
    password: str,
    page_numbers: Iterable[int] | None,
    maxpages: int,
    laparams: LAParams | None,
) -> str:
    pages = list(extract_pages(source, password, page_numbers, maxpages, True, laparams))
    if output_type == "html":
        html_parts: list[str] = []
        for page in pages:
            html_parts.append(f'<div class="page" data-page="{page.pageid}">')
            for item in page:
                if isinstance(item, LTText) and isinstance(item, LTComponent):
                    bbox = ",".join(str(value) for value in item.bbox)
                    html_parts.append(
                        f'<div class="textbox" data-bbox="{bbox}">{escape(item.get_text())}</div>'
                    )
            html_parts.append("</div>")
        body = "".join(html_parts)
        return f"<!doctype html><html><body>{body}</body></html>"
    xml_parts: list[str] = []
    for page in pages:
        xml_parts.append(f'<page id="{page.pageid}" bbox="{page.bbox}">')
        for item in page:
            if isinstance(item, LTText) and isinstance(item, LTComponent):
                xml_parts.append(f'<textbox bbox="{item.bbox}">{escape(item.get_text())}</textbox>')
        xml_parts.append("</page>")
    return "<pages>" + "".join(xml_parts) + "</pages>"


__all__ = (
    "LAParams",
    "LTAnno",
    "LTChar",
    "LTFigure",
    "LTImage",
    "LTItem",
    "LTPage",
    "LTText",
    "LTTextBox",
    "LTTextBoxHorizontal",
    "LTTextLine",
    "LTTextLineHorizontal",
    "LTTextLineVertical",
    "LTTextBoxVertical",
    "LTTextContainer",
    "extract_pages",
    "extract_text",
    "extract_text_to_fp",
)
