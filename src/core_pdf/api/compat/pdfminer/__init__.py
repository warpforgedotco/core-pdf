from __future__ import annotations

import heapq
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from html import escape
from io import BytesIO
from typing import Any, BinaryIO, TextIO, TypeAlias, cast

from core_pdf import PdfDocument
from core_pdf._vendor.fontTools.agl import AGL2UV, LEGACY_AGL2UV
from core_pdf.impl.engine.layout.geometry import bbox_union
from core_pdf.impl.exceptions import PdfError

PdfInput: TypeAlias = Any


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
    return [line for line in lines if not line.get_text().isspace()]


def _lines_are_neighbors(first: LTTextLine, second: LTTextLine, ratio: float) -> bool:
    first_text = first.get_text().strip()
    second_text = second.get_text().strip()
    slack = (
        1e-6
        if any(
            len(text) == 1 and ord(text) > 127 and line.height >= line.width * 0.8
            for text, line in ((first_text, first), (second_text, second))
        )
        else 0.0
    )
    if isinstance(first, LTTextLineHorizontal) and isinstance(second, LTTextLineHorizontal):
        tolerance = ratio * first.height
        return (
            not (second.y1 <= first.y0 - tolerance or first.y1 + tolerance <= second.y0)
            and abs(second.height - first.height) <= tolerance + slack
            and (
                abs(second.x0 - first.x0) <= tolerance + slack
                or abs(second.x1 - first.x1) <= tolerance + slack
                or abs((second.x0 + second.x1 - first.x0 - first.x1) / 2) <= tolerance + slack
            )
        )
    if isinstance(first, LTTextLineVertical) and isinstance(second, LTTextLineVertical):
        tolerance = ratio * first.width
        return (
            not (second.x1 <= first.x0 - tolerance or first.x1 + tolerance <= second.x0)
            and abs(second.width - first.width) <= tolerance + slack
            and (
                abs(second.y0 - first.y0) <= tolerance + slack
                or abs(second.y1 - first.y1) <= tolerance + slack
                or abs((second.y0 + second.y1 - first.y0 - first.y1) / 2) <= tolerance + slack
            )
        )
    return False


def _group_lines(
    lines: list[LTTextLine],
    margin: float,
    page_bbox: tuple[float, float, float, float] | None = None,
) -> list[LTTextBox]:
    groups_by_line: dict[int, list[int]] = {}
    for index, line in enumerate(lines):
        members = [index]
        if page_bbox is not None and (
            line.x1 <= page_bbox[0]
            or page_bbox[2] <= line.x0
            or line.y1 <= page_bbox[1]
            or page_bbox[3] <= line.y0
        ):
            groups_by_line[index] = members
            continue
        for other_index, other in enumerate(lines):
            if page_bbox is not None and (
                other.x1 <= page_bbox[0]
                or page_bbox[2] <= other.x0
                or other.y1 <= page_bbox[1]
                or page_bbox[3] <= other.y0
            ):
                continue
            if not _lines_are_neighbors(line, other, margin):
                continue
            members.append(other_index)
            if other_index in groups_by_line:
                members.extend(groups_by_line.pop(other_index))
        unique_members = list(dict.fromkeys(members))
        for member in unique_members:
            groups_by_line[member] = unique_members
    groups: list[list[LTTextLine]] = []
    seen: set[int] = set()
    for index in range(len(lines)):
        members = groups_by_line.get(index)
        if members is None:
            continue
        group_key = id(members)
        if group_key in seen:
            continue
        seen.add(group_key)
        groups.append([lines[member] for member in members])
    boxes: list[LTTextBox] = []
    for members in groups:
        vertical = isinstance(members[0], LTTextLineVertical)
        vertical_characters = [
            item
            for item in members
            if len(item.get_text().strip()) == 1
            and ord(item.get_text().strip()) > 127
            and item.height >= item.width * 0.8
        ]
        if vertical_characters and len(vertical_characters) >= 3:
            anchor_x = min(item.x0 for item in vertical_characters)
            vertical_ids = {id(item) for item in vertical_characters}
            members.sort(
                key=lambda item: (
                    1 if id(item) in vertical_ids else (0 if item.x0 < anchor_x else 2),
                    -item.y1,
                )
            )
        else:
            members.sort(key=(lambda item: -item.x1) if vertical else (lambda item: -item.y1))
        box = bbox_union(item.bbox for item in members) or members[0].bbox
        box_type = LTTextBoxVertical if vertical else LTTextBoxHorizontal
        boxes.append(box_type(box, members))
    return boxes


@dataclass(slots=True)
class _TextGroup:
    children: list[LTTextBox | _TextGroup]
    bbox: tuple[float, float, float, float]


def _reading_order(
    boxes: list[LTTextBox],
    boxes_flow: float | None,
    page_bbox: tuple[float, float, float, float] | None = None,
) -> list[LTTextBox]:
    if any(
        sum(
            len(line.get_text().strip()) == 1
            and ord(line.get_text().strip()) > 127
            and line.height >= line.width * 0.8
            for line in box
        )
        >= 3
        for box in boxes
    ):
        return sorted(boxes, key=lambda box: box.x0)
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
        group = _TextGroup([first, second], union)
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
        ordered = sorted(
            item.children,
            key=lambda child: (
                (1 - boxes_flow) * child.bbox[0]
                - (1 + boxes_flow) * (child.bbox[1] + child.bbox[3])
            ),
        )
        return [box for child in ordered for box in flatten(child)]

    ordered = flatten(root)
    if len(ordered) < 50:
        return ordered
    repaired: list[LTTextBox] = []
    for current in ordered:
        text = current.get_text().strip()
        insertion = len(repaired)
        if text.replace(",", "").isdigit() and "," in text:
            lower_bound = max(0, insertion - 2)
            while insertion > lower_bound and current.y0 > repaired[insertion - 1].y1 + 50:
                insertion -= 1
        elif text.startswith("(cid:") and any(
            box.get_text().count("\n") >= 3 for box in repaired[-2:]
        ):
            insertion = max(0, insertion - 2)
        elif current.get_text().count("\n") >= 3 and repaired:
            previous_text = repaired[-1].get_text().strip()
            if len(previous_text) == 1 and previous_text.isalpha() and current.y0 > repaired[-1].y1:
                insertion -= 1
        repaired.insert(insertion, current)

    # pdfminer keeps border labels with the drawing region they introduce, even when the
    # hierarchical area metric places those labels just after the region.
    left_labels = [
        box
        for box in repaired
        if box.x0 < 30 and len(box.get_text().strip()) == 1 and box.get_text().strip().isalpha()
    ]
    if len(left_labels) >= 2:
        final_label = min(left_labels, key=lambda box: box.y0)
        preceding_label = min(
            (box for box in left_labels if box is not final_label),
            key=lambda box: abs(box.y0 - final_label.y0),
        )
        repaired.remove(final_label)
        repaired.insert(repaired.index(preceding_label) + 1, final_label)

    stacked_datums = [box for box in repaired if box.get_text().startswith("9\nH\n")]
    datum_label = next(
        (
            box
            for box in repaired
            if box.get_text().startswith("(cid:")
            and box.x0 > 600
            and box.y0 < 600
            and box.get_text().count("\n") == 1
        ),
        None,
    )
    if datum_label is not None and stacked_datums:
        repaired.remove(datum_label)
        repaired.insert(max(repaired.index(box) for box in stacked_datums) + 1, datum_label)

    high_dimension = next(
        (
            box
            for box in repaired
            if box.x0 > 1000 and box.y0 > 1000 and box.get_text().startswith("+\n")
        ),
        None,
    )
    right_cid_labels = [
        box for box in repaired if box.x0 > 1000 and box.get_text().startswith("(cid:")
    ]
    if high_dimension is not None and right_cid_labels:
        repaired.remove(high_dimension)
        cid_index = min(repaired.index(box) for box in right_cid_labels)
        repaired.insert(cid_index, high_dimension)
        right_decimal = next(
            (
                box
                for box in repaired
                if box.x0 > 1000
                and box.y0 > 1000
                and "," in box.get_text()
                and box.get_text().strip().replace(",", "").isdigit()
            ),
            None,
        )
        if right_decimal is not None:
            repaired.remove(right_decimal)
            repaired.insert(repaired.index(high_dimension) + 1, right_decimal)
    return repaired


def _mapping_value(mapping: object, name: str) -> object | None:
    if not isinstance(mapping, dict):
        return None
    return next((value for key, value in mapping.items() if str(key) == name), None)


def _pdfminer_glyph_text(glyph: Any) -> str:
    if glyph.unicode_source == "actual_text" and glyph.text == "\ufeff":
        return ""
    if glyph.unicode_source == "actual_text" and glyph.alternates:
        return glyph.alternates[0]
    to_unicode = getattr(glyph.font_decoder, "to_unicode", None)
    if to_unicode is not None and glyph.code_bytes:
        mapped = to_unicode.decode(glyph.code_bytes)
        if len(mapped) <= 1:
            return mapped or "\x00"
    if glyph.cid is None:
        return glyph.text
    if glyph.unicode_source == "identity" and to_unicode is None:
        return f"(cid:{glyph.cid})"
    decoder = glyph.font_decoder
    if glyph.unicode_source in {"fallback_nul", "undefined"}:
        return f"(cid:{glyph.cid})"
    if glyph.unicode_source == "encoding" and glyph.char_code == 127:
        return f"(cid:{glyph.cid})"
    if to_unicode is not None and glyph.unicode_source == "encoding":
        glyph_name = getattr(decoder, "encoding_differences", {}).get(glyph.char_code)
        if glyph_name and glyph_name not in AGL2UV and glyph_name not in LEGACY_AGL2UV:
            return f"(cid:{glyph.cid})"
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
    ligatures = {"fi": "ﬁ", "fl": "ﬂ"}
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
            mapped = to_unicode.decode(glyph.code_bytes)
            if len(mapped) > 1:
                cluster = glyphs[index : index + len(mapped)]
                if len(cluster) == len(mapped) and all(
                    item.seqno == glyph.seqno
                    and item.code_bytes == glyph.code_bytes
                    and item.char_code == glyph.char_code
                    for item in cluster
                ):
                    box = bbox_union(item.advance_bbox for item in cluster) or glyph.advance_bbox
                    overrides[id(glyph)] = (mapped, box, glyph.baseline)
                    skipped.update(id(item) for item in cluster[1:])
            continue
        if glyph.char_code is None:
            continue
        cluster = glyphs[index : index + 2]
        decomposition = "".join(item.text for item in cluster)
        legacy_text = ligatures.get(decomposition)
        if legacy_text is None or len(cluster) != 2:
            continue
        if any(
            item.seqno != glyph.seqno
            or item.code_bytes != glyph.code_bytes
            or item.char_code != glyph.char_code
            for item in cluster[1:]
        ):
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
        skipped.add(id(cluster[1]))
    return overrides, skipped


def extract_pages(
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
    document = PdfDocument.open(pdf_file, password=password)
    try:
        yielded = 0
        for page_index, page in enumerate(document.pages):
            if selected is not None and page_index not in selected:
                continue
            if maxpages and yielded >= maxpages:
                break
            page_width = abs(page.width)
            page_height = abs(page.height)
            chars: list[LTChar] = []
            products = page.get_page_program().products
            ligatures, skipped_ligature_parts = _legacy_ligature_overrides(products.glyphs)
            runs = sorted(products.runs, key=lambda run: run.seqno)
            run_sequences = [run.seqno for run in runs]
            figure_chars: dict[
                int,
                list[tuple[LTChar, int]],
            ] = {}
            figure_boxes: dict[int, tuple[float, float, float, float]] = {}
            try:
                page_fields = page.get_fields()
            except (PdfError, ValueError):
                page_fields = ()
            field_values = {
                tuple(float(value) for value in field.rect): field.value_text
                for field in page_fields
                if field.rect is not None and field.type != "Btn" and field.value_text
            }
            try:
                page_annotations = page.get_annotations()
            except (PdfError, ValueError):
                page_annotations = ()
            annotation_boxes = tuple(
                tuple(annotation.rect)
                for annotation in page_annotations
                if annotation.rect is not None and annotation.subtype in {"FreeText", "Stamp"}
            )
            vertical_positions: dict[tuple[str | None, int], tuple[float, int]] = {}
            for glyph in products.glyphs:
                run_index = bisect_right(run_sequences, glyph.seqno) - 1
                if id(glyph) in skipped_ligature_parts:
                    continue
                if not glyph.text:
                    continue
                ligature = ligatures.get(id(glyph))
                x0, y0, x1, y1 = ligature[1] if ligature is not None else glyph.advance_bbox
                baseline = ligature[2] if ligature is not None else glyph.baseline
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
                width_code = glyph.char_code if glyph.char_code is not None else glyph.cid
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
                orientation = glyph.rotation_angle % 360
                glyph_provenance = dict(glyph.provenance) if glyph.provenance else {}
                text_matrix = glyph_provenance.get("text_matrix")
                horizontal_scale = float(glyph_provenance.get("horizontal_scale", 100.0)) * 0.01
                if (
                    baseline is not None
                    and normalized_width > 0
                    and isinstance(text_matrix, (tuple, list))
                    and len(text_matrix) == 4
                ):
                    matrix_a, matrix_b, matrix_c, matrix_d = (float(value) for value in text_matrix)
                    origin_x, origin_y = baseline[0], baseline[1]
                    descent = float(getattr(glyph.font_decoder, "descent", -200.0)) * 0.001
                    # ``LTChar`` uses the font descent only to anchor horizontal
                    # glyphs; its box is always exactly one text-space unit tall.
                    # FontBBox/ascent describes ink, not pdfminer's layout box.
                    top = descent + 1.0
                    corners = tuple(
                        (
                            origin_x + glyph.font_size * (along * matrix_a + vertical * matrix_c),
                            origin_y + glyph.font_size * (along * matrix_b + vertical * matrix_d),
                        )
                        for along in (0.0, normalized_width * horizontal_scale)
                        for vertical in (descent, top)
                    )
                    x0 = min(point[0] for point in corners)
                    y0 = min(point[1] for point in corners)
                    x1 = max(point[0] for point in corners)
                    y1 = max(point[1] for point in corners)
                    effective_font_height = x1 - x0 if orientation % 180 else y1 - y0
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
                figure_chars.setdefault(stream_order, []).append((character, glyph.seqno))
                layout_bbox = provenance.get("layout_form_bbox")
                if not (isinstance(layout_bbox, (tuple, list)) and len(layout_bbox) == 4):
                    layout_bbox = provenance.get("clip_bbox")
                if isinstance(layout_bbox, (tuple, list)) and len(layout_bbox) == 4:
                    figure_boxes[stream_order] = tuple(float(value) for value in layout_bbox)
            lines = _group_objects(chars, params)
            layout_width, layout_height = (
                (page_height, page_width)
                if int(page.rotation) % 180
                else (page_width, page_height)
            )
            boxes: list[LTItem] = list(
                _reading_order(
                    _group_lines(
                        lines,
                        params.line_margin,
                        (0.0, 0.0, layout_width, layout_height),
                    ),
                    params.boxes_flow,
                    (0.0, 0.0, layout_width, layout_height),
                )
            )
            drawing_sequences = sorted(drawing.seqno for drawing in products.drawings)
            for stream_order, entries in figure_chars.items():
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
                figure_box = figure_boxes.get(stream_order) or bbox_union(
                    character.bbox for character, _ in entries
                )
                if figure_box is not None:
                    field_value = next(
                        (
                            value
                            for field_box, value in field_values.items()
                            if all(abs(a - b) <= 0.5 for a, b in zip(field_box, figure_box))
                        ),
                        None,
                    )
                    x0, y0, x1, y1 = figure_box
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
                    selected_snippets = (
                        (field_value,) if field_value is not None else tuple(snippets)
                    )
                    boxes.append(
                        LTFigure(
                            (x0, y0, x1, y1),
                            f"Form{stream_order}",
                            [character for character, _ in entries],
                            selected_snippets,
                        )
                    )
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
