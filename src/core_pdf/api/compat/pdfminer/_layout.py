"""PDFMiner layout records, grouping, and reading order."""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass, field

from core_pdf.impl._impl.model.geometry import bbox_union


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
            current.append(item)
            current_vertical = True
        elif horizontal and not vertical:
            current.append(item)
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
        line_members = groups_by_line.get(id(line))
        if line_members is None:
            continue
        group_key = id(line_members)
        if group_key in seen:
            continue
        seen.add(group_key)
        groups.append(line_members)
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
    # Heap entries use object identity like pdfminer itself. Keep merged groups
    # alive until ordering completes so CPython cannot recycle an id while a
    # stale entry for the former object is still queued.
    retained_groups: list[_TextGroup] = []

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
        active_first = active[first_id]
        active_second = active[second_id]
        union = bbox_union((active_first.bbox, active_second.bbox)) or active_first.bbox
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
            isinstance(active_first, LTTextBoxVertical)
            or isinstance(active_second, LTTextBoxVertical)
            or (isinstance(active_first, _TextGroup) and active_first.vertical)
            or (isinstance(active_second, _TextGroup) and active_second.vertical)
        )
        group = _TextGroup([active_first, active_second], union, vertical)
        retained_groups.append(group)
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
