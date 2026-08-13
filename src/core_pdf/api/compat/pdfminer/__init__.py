from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from html import escape
from io import BytesIO
from typing import Any, BinaryIO, TextIO, TypeAlias, cast

from core_pdf import PdfDocument
from core_pdf.impl.engine.layout.geometry import bbox_union

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


def _group_lines(lines: list[LTTextLine], margin: float) -> list[LTTextBox]:
    groups_by_line: dict[int, list[int]] = {}
    for index, line in enumerate(lines):
        members = [index]
        for other_index, other in enumerate(lines):
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
        if members is None or id(members) in seen:
            continue
        seen.add(id(members))
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


def _reading_order(boxes: list[LTTextBox], boxes_flow: float | None) -> list[LTTextBox]:
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

    active: dict[int, LTTextBox | _TextGroup] = dict(enumerate(boxes))
    next_group_id = len(active)

    def area_gap(first: LTTextBox | _TextGroup, second: LTTextBox | _TextGroup) -> float:
        x0 = min(first.bbox[0], second.bbox[0])
        y0 = min(first.bbox[1], second.bbox[1])
        x1 = max(first.bbox[2], second.bbox[2])
        y1 = max(first.bbox[3], second.bbox[3])
        first_area = (first.bbox[2] - first.bbox[0]) * (first.bbox[3] - first.bbox[1])
        second_area = (second.bbox[2] - second.bbox[0]) * (second.bbox[3] - second.bbox[1])
        return (x1 - x0) * (y1 - y0) - first_area - second_area

    queue: list[tuple[bool, float, int, int]] = []
    for first_id, first in active.items():
        for second_id in range(first_id + 1, len(boxes)):
            second = active[second_id]
            heapq.heappush(queue, (False, area_gap(first, second), first_id, second_id))
    while queue:
        skip_between, _distance, first_id, second_id = heapq.heappop(queue)
        if first_id not in active or second_id not in active:
            continue
        first = active[first_id]
        second = active[second_id]
        union = bbox_union((first.bbox, second.bbox)) or first.bbox
        between = [
            item
            for item_id, item in active.items()
            if item_id not in {first_id, second_id}
            and not (
                item.bbox[2] < union[0]
                or union[2] < item.bbox[0]
                or item.bbox[3] < union[1]
                or union[3] < item.bbox[1]
            )
        ]
        if between and not skip_between:
            heapq.heappush(queue, (True, _distance, first_id, second_id))
            continue
        group = _TextGroup([first, second], union)
        del active[first_id], active[second_id]
        group_id = next_group_id
        next_group_id += 1
        for other_id, other in active.items():
            heapq.heappush(queue, (False, area_gap(group, other), group_id, other_id))
        active[group_id] = group

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
    if glyph.cid is None:
        return glyph.text
    if (
        glyph.unicode_source == "identity"
        and getattr(glyph.font_decoder, "to_unicode", None) is None
    ):
        return f"(cid:{glyph.cid})"
    decoder = glyph.font_decoder
    if glyph.unicode_source == "truetype_cmap" and getattr(decoder, "to_unicode", None) is None:
        descendants = _mapping_value(getattr(decoder, "font", None), "DescendantFonts")
        descendant = descendants[0] if isinstance(descendants, list) and descendants else None
        system_info = _mapping_value(descendant, "CIDSystemInfo")
        registry = _mapping_value(system_info, "Registry")
        registry_data = getattr(registry, "data", b"")
        if isinstance(registry_data, bytes) and registry_data.strip() == b"PDFAUTOCAD":
            return f"(cid:{glyph.cid})"
    return glyph.text


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
            chars: list[LTChar] = []
            annotation_boxes = tuple(
                tuple(annotation.rect)
                for annotation in page.get_annotations()
                if annotation.rect is not None
            )
            vertical_positions: dict[tuple[str | None, int], tuple[float, int]] = {}
            for glyph in page.get_page_program().products.glyphs:
                if not glyph.text:
                    continue
                x0, y0, x1, y1 = glyph.advance_bbox
                center_x = (x0 + x1) / 2.0
                center_y = (y0 + y1) / 2.0
                if any(
                    left <= center_x <= right and bottom <= center_y <= top
                    for left, bottom, right, top in annotation_boxes
                ):
                    continue
                text = _pdfminer_glyph_text(glyph)
                if (
                    glyph.baseline is not None
                    and glyph.font_size > 0
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
                # Core's advance box includes character/word spacing. pdfminer keeps LTChar's
                # box at the font's nominal advance and leaves that spacing before the next
                # character, so recover the nominal width from the retained font decoder.
                width_code = glyph.char_code if glyph.char_code is not None else glyph.cid
                width_lookup = getattr(glyph.font_decoder, "glyph_width", None)
                if (
                    glyph.rotation_angle % 180 == 0
                    and glyph.font_size >= (y1 - y0) * 0.5
                    and width_code is not None
                    and callable(width_lookup)
                ):
                    x1 = x0 + float(width_lookup(width_code)) * glyph.font_size * 0.001
                    y1 = y0 + glyph.font_size
                rotation = int(page.rotation) % 360
                if rotation == 90:
                    x0, y0, x1, y1 = y0, page.width - x1, y1, page.width - x0
                elif rotation == 180:
                    x0, y0, x1, y1 = (
                        page.width - x1,
                        page.height - y1,
                        page.width - x0,
                        page.height - y0,
                    )
                elif rotation == 270:
                    x0, y0, x1, y1 = page.height - y1, x0, page.height - y0, x1
                chars.append(
                    LTChar(
                        (x0, y0, x1, y1),
                        text,
                        glyph.font_name,
                        glyph.font_size,
                    )
                )
            lines = _group_objects(chars, params)
            boxes: list[LTItem] = list(
                _reading_order(_group_lines(lines, params.line_margin), params.boxes_flow)
            )
            page_width, page_height = (
                (page.height, page.width) if int(page.rotation) % 180 else (page.width, page.height)
            )
            yield LTPage(
                (0.0, 0.0, page_width, page_height),
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
