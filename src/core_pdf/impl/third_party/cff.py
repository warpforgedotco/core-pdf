from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, inf


@dataclass(frozen=True)
class CFFGlyphFeature:
    cells: tuple[tuple[int, int], ...]
    aspect: float
    contours: int
    bitmap: tuple[int, ...] = ()


EMPTY_FEATURE = CFFGlyphFeature((), 0.0, 0, ())
FEATURE_GRID_WIDTH = 18
FEATURE_GRID_HEIGHT = 24


class CFFFont:
    __slots__ = (
        "data",
        "top_dict",
        "charstrings",
        "cid_to_gid",
        "global_subrs",
        "local_subrs",
        "fd_select",
    )

    def __init__(self, data: bytes | memoryview) -> None:
        self.data = data.tobytes() if isinstance(data, memoryview) else data
        if len(self.data) < 4 or self.data[0] != 1:
            raise ValueError("invalid CFF font program")
        pos = self.data[2]
        ignored_names, pos = self._read_index(pos)
        top_index, pos = self._read_index(pos)
        ignored_strings, pos = self._read_index(pos)
        global_subrs, pos = self._read_index(pos)
        self.global_subrs = tuple(global_subrs)
        if not top_index:
            raise ValueError("invalid CFF top dict")
        self.top_dict = self._parse_dict(top_index[0])
        charstrings_off = self.top_dict.get(17, [None])[0]
        if not isinstance(charstrings_off, (int, float)):
            raise ValueError("invalid CFF CharStrings offset")
        self.charstrings, ignored_pos = self._read_index(int(charstrings_off))
        charset_off = self.top_dict.get(15, [0])[0]
        self.cid_to_gid = self._read_charset(
            int(charset_off) if isinstance(charset_off, (int, float)) else 0,
            len(self.charstrings),
        )
        self.fd_select = self._read_fd_select()
        self.local_subrs = self._read_local_subrs()

    def _read_index(self, pos: int) -> tuple[list[bytes], int]:
        data = self.data
        if pos + 2 > len(data):
            raise ValueError("invalid CFF INDEX")
        count = int.from_bytes(data[pos : pos + 2], "big")
        pos += 2
        if count == 0:
            return ([], pos)
        if pos >= len(data):
            raise ValueError("invalid CFF INDEX")
        off_size = data[pos]
        pos += 1
        if off_size < 1 or off_size > 4:
            raise ValueError("invalid CFF INDEX")
        offsets_end = pos + (count + 1) * off_size
        if offsets_end > len(data):
            raise ValueError("invalid CFF INDEX")
        offsets = [
            int.from_bytes(data[pos + i * off_size : pos + (i + 1) * off_size], "big")
            for i in range(count + 1)
        ]
        pos = offsets_end
        if offsets[0] != 1 or any(b < a for a, b in zip(offsets, offsets[1:])):
            raise ValueError("invalid CFF INDEX")
        base = pos
        end = base + offsets[-1] - 1
        if end > len(data):
            raise ValueError("invalid CFF INDEX")
        return (
            [
                data[base + offsets[i] - 1 : base + offsets[i + 1] - 1]
                for i in range(count)
            ],
            end,
        )

    def _parse_number(
        self, item: bytes, pos: int, *, dict_number: bool = False
    ) -> tuple[float, int]:
        b0 = item[pos]
        if 32 <= b0 <= 246:
            return (float(b0 - 139), pos + 1)
        if 247 <= b0 <= 250:
            if pos + 1 >= len(item):
                raise ValueError("invalid CFF number")
            return (float((b0 - 247) * 256 + item[pos + 1] + 108), pos + 2)
        if 251 <= b0 <= 254:
            if pos + 1 >= len(item):
                raise ValueError("invalid CFF number")
            return (float(-(b0 - 251) * 256 - item[pos + 1] - 108), pos + 2)
        if b0 == 28:
            if pos + 3 > len(item):
                raise ValueError("invalid CFF number")
            return (
                float(int.from_bytes(item[pos + 1 : pos + 3], "big", signed=True)),
                pos + 3,
            )
        if b0 == 29 and dict_number:
            if pos + 5 > len(item):
                raise ValueError("invalid CFF number")
            return (
                float(int.from_bytes(item[pos + 1 : pos + 5], "big", signed=True)),
                pos + 5,
            )
        if b0 == 30 and dict_number:
            return self._parse_real_number(item, pos + 1)
        if b0 == 255 and not dict_number:
            if pos + 5 > len(item):
                raise ValueError("invalid Type 2 number")
            return (
                int.from_bytes(item[pos + 1 : pos + 5], "big", signed=True) / 65536.0,
                pos + 5,
            )
        raise ValueError("invalid CFF number")

    def _parse_real_number(self, item: bytes, pos: int) -> tuple[float, int]:
        parts: list[str] = []
        while pos < len(item):
            byte = item[pos]
            pos += 1
            for nibble in (byte >> 4, byte & 15):
                if nibble == 15:
                    text = "".join(parts) or "0"
                    return (float(text), pos)
                if nibble <= 9:
                    parts.append(str(nibble))
                elif nibble == 10:
                    parts.append(".")
                elif nibble == 11:
                    parts.append("e")
                elif nibble == 12:
                    parts.append("e-")
                elif nibble == 14:
                    parts.append("-")
        raise ValueError("invalid CFF real number")

    def _parse_dict(self, item: bytes) -> dict[int | tuple[int, int], list[float]]:
        result: dict[int | tuple[int, int], list[float]] = {}
        stack: list[float] = []
        pos = 0
        while pos < len(item):
            byte = item[pos]
            if byte <= 21:
                if byte == 12:
                    pos += 1
                    if pos >= len(item):
                        raise ValueError("invalid CFF dict operator")
                    op: int | tuple[int, int] = (12, item[pos])
                else:
                    op = byte
                result[op] = stack
                stack = []
                pos += 1
            else:
                value, pos = self._parse_number(item, pos, dict_number=True)
                stack.append(value)
        return result

    def _read_charset(self, pos: int, glyph_count: int) -> dict[int, int]:
        if glyph_count <= 0:
            return {}
        if glyph_count == 1 or pos in {0, 1, 2}:
            return {gid: gid for gid in range(glyph_count)}
        data = self.data
        if pos >= len(data):
            return {gid: gid for gid in range(glyph_count)}
        fmt = data[pos]
        pos += 1
        cid_to_gid = {0: 0}
        gid = 1
        try:
            if fmt == 0:
                while gid < glyph_count:
                    if pos + 2 > len(data):
                        break
                    cid = int.from_bytes(data[pos : pos + 2], "big")
                    pos += 2
                    cid_to_gid.setdefault(cid, gid)
                    gid += 1
            elif fmt in {1, 2}:
                while gid < glyph_count:
                    if pos + 2 > len(data):
                        break
                    first = int.from_bytes(data[pos : pos + 2], "big")
                    pos += 2
                    if fmt == 1:
                        if pos >= len(data):
                            break
                        left = data[pos]
                        pos += 1
                    else:
                        if pos + 2 > len(data):
                            break
                        left = int.from_bytes(data[pos : pos + 2], "big")
                        pos += 2
                    for offset in range(left + 1):
                        if gid >= glyph_count:
                            break
                        cid_to_gid.setdefault(first + offset, gid)
                        gid += 1
            else:
                return {gid: gid for gid in range(glyph_count)}
        except IndexError:
            pass
        return cid_to_gid

    def glyph_id_for_cid(self, cid: int) -> int:
        return self.cid_to_gid.get(cid, cid)

    def _read_fd_select(self) -> tuple[int, ...]:
        glyph_count = len(self.charstrings)
        fdselect_off = self.top_dict.get((12, 37), [None])[0]
        if not isinstance(fdselect_off, (int, float)):
            return (0,) * glyph_count
        pos = int(fdselect_off)
        data = self.data
        if pos >= len(data):
            return (0,) * glyph_count
        fmt = data[pos]
        pos += 1
        fd_select = [0] * glyph_count
        try:
            if fmt == 0:
                if pos + glyph_count <= len(data):
                    return tuple(data[pos : pos + glyph_count])
            elif fmt == 3:
                if pos + 2 > len(data):
                    return tuple(fd_select)
                range_count = int.from_bytes(data[pos : pos + 2], "big")
                pos += 2
                ranges: list[tuple[int, int]] = []
                for ignored in range(range_count):
                    if pos + 3 > len(data):
                        return tuple(fd_select)
                    first = int.from_bytes(data[pos : pos + 2], "big")
                    fd = data[pos + 2]
                    pos += 3
                    ranges.append((first, fd))
                if pos + 2 > len(data):
                    return tuple(fd_select)
                sentinel = int.from_bytes(data[pos : pos + 2], "big")
                for idx, (first, fd) in enumerate(ranges):
                    end = ranges[idx + 1][0] if idx + 1 < len(ranges) else sentinel
                    for gid in range(max(0, first), min(glyph_count, end)):
                        fd_select[gid] = fd
        except IndexError:
            pass
        return tuple(fd_select)

    def _read_local_subrs(self) -> tuple[tuple[bytes, ...], ...]:
        fdarray_off = self.top_dict.get((12, 36), [None])[0]
        if isinstance(fdarray_off, (int, float)):
            try:
                fd_dicts_raw, ignored_pos = self._read_index(int(fdarray_off))
            except ValueError:
                fd_dicts_raw = []
            return tuple(
                tuple(self._read_private_subrs(self._parse_dict(fd_dict_raw)))
                for fd_dict_raw in fd_dicts_raw
            )
        return (tuple(self._read_private_subrs(self.top_dict)),)

    def _read_private_subrs(
        self, font_dict: dict[int | tuple[int, int], list[float]]
    ) -> list[bytes]:
        private = font_dict.get(18)
        if not isinstance(private, list) or len(private) < 2:
            return []
        size, offset = private[:2]
        if not isinstance(size, (int, float)) or not isinstance(offset, (int, float)):
            return []
        private_off = int(offset)
        private_size = int(size)
        if (
            private_off < 0
            or private_size <= 0
            or private_off + private_size > len(self.data)
        ):
            return []
        private_dict = self._parse_dict(
            self.data[private_off : private_off + private_size]
        )
        subrs_off = private_dict.get(19, [None])[0]
        if not isinstance(subrs_off, (int, float)):
            return []
        try:
            subrs, ignored_pos = self._read_index(private_off + int(subrs_off))
        except ValueError:
            return []
        return subrs

    def local_subrs_for_glyph(self, glyph_id: int) -> tuple[bytes, ...]:
        fd_index = self.fd_select[glyph_id] if glyph_id < len(self.fd_select) else 0
        if 0 <= fd_index < len(self.local_subrs):
            return self.local_subrs[fd_index]
        return ()

    def glyph_feature(self, glyph_id: int) -> CFFGlyphFeature:
        try:
            charstring = self.charstrings[glyph_id]
        except IndexError:
            return EMPTY_FEATURE
        return type2_glyph_feature(
            charstring,
            local_subrs=self.local_subrs_for_glyph(glyph_id),
            global_subrs=self.global_subrs,
        )

    def glyph_feature_for_cid(self, cid: int) -> CFFGlyphFeature:
        return self.glyph_feature(self.glyph_id_for_cid(cid))

    def glyph_bitmap(
        self, cid: int, width: int = 24, height: int = 32
    ) -> tuple[int, ...]:
        try:
            glyph_id = self.glyph_id_for_cid(cid)
            charstring = self.charstrings[glyph_id]
        except IndexError:
            return ()
        return type2_glyph_bitmap(
            charstring,
            width=width,
            height=height,
            local_subrs=self.local_subrs_for_glyph(glyph_id),
            global_subrs=self.global_subrs,
        )

    def glyph_bbox(self, cid: int) -> tuple[float, float, float, float] | None:
        try:
            glyph_id = self.glyph_id_for_cid(cid)
            charstring = self.charstrings[glyph_id]
        except IndexError:
            return None
        contours = _type2_glyph_contours(
            charstring,
            local_subrs=self.local_subrs_for_glyph(glyph_id),
            global_subrs=self.global_subrs,
        )
        points = [point for contour in contours for point in contour]
        if not points:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return (min(xs), min(ys), max(xs), max(ys))


def type2_glyph_feature(
    charstring: bytes,
    *,
    local_subrs: tuple[bytes, ...] = (),
    global_subrs: tuple[bytes, ...] = (),
) -> CFFGlyphFeature:
    contours = _type2_glyph_contours(
        charstring,
        local_subrs=local_subrs,
        global_subrs=global_subrs,
    )
    if not contours:
        return EMPTY_FEATURE

    points = [point for contour in contours for point in contour]
    if not points:
        return EMPTY_FEATURE
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    cells = {
        (
            max(0, min(17, round((px - min_x) / width * 17))),
            max(0, min(23, round((py - min_y) / height * 23))),
        )
        for px, py in points
    }
    bitmap = _rasterize_contours(contours, width=18, height=24)
    return CFFGlyphFeature(
        tuple(sorted(cells)), round(width / height, 2), len(contours), bitmap
    )


def type2_glyph_bitmap(
    charstring: bytes,
    *,
    width: int = 24,
    height: int = 32,
    local_subrs: tuple[bytes, ...] = (),
    global_subrs: tuple[bytes, ...] = (),
) -> tuple[int, ...]:
    contours = _type2_glyph_contours(
        charstring,
        local_subrs=local_subrs,
        global_subrs=global_subrs,
    )
    if not contours:
        return ()
    return _rasterize_contours(contours, width=width, height=height)


def _type2_glyph_contours(
    charstring: bytes,
    *,
    local_subrs: tuple[bytes, ...] = (),
    global_subrs: tuple[bytes, ...] = (),
) -> list[list[tuple[float, float]]]:
    stack: list[float] = []
    contours: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    x = 0.0
    y = 0.0
    stem_count = 0
    font = object.__new__(CFFFont)
    subr_bias = _type2_subr_bias(len(local_subrs))
    gsubr_bias = _type2_subr_bias(len(global_subrs))

    def flush_contour() -> None:
        nonlocal current
        if current:
            contours.append(current)
            current = []

    def move(dx: float, dy: float) -> None:
        nonlocal x, y, current
        flush_contour()
        x += dx
        y += dy
        current = [(x, y)]

    def line(dx: float, dy: float) -> None:
        nonlocal x, y
        x += dx
        y += dy
        current.append((x, y))

    def curve(
        dx1: float, dy1: float, dx2: float, dy2: float, dx3: float, dy3: float
    ) -> None:
        nonlocal x, y
        x0, y0 = x, y
        x1, y1 = x + dx1, y + dy1
        x2, y2 = x1 + dx2, y1 + dy2
        x3, y3 = x2 + dx3, y2 + dy3
        for t in (0.25, 0.5, 0.75, 1.0):
            mt = 1.0 - t
            current.append(
                (
                    mt**3 * x0 + 3 * mt * mt * t * x1 + 3 * mt * t * t * x2 + t**3 * x3,
                    mt**3 * y0 + 3 * mt * mt * t * y1 + 3 * mt * t * t * y2 + t**3 * y3,
                )
            )
        x, y = x3, y3

    def clear_stack() -> None:
        del stack[:]

    def execute(program: bytes, depth: int = 0) -> bool:
        nonlocal stem_count
        if depth > 10:
            return False
        pos = 0
        try:
            while pos < len(program):
                byte = program[pos]
                if byte > 31 or byte in {28, 255}:
                    value, pos = CFFFont._parse_number(font, program, pos)
                    stack.append(value)
                    continue
                pos += 1
                if byte in (1, 3, 18, 23):
                    stem_count += len(stack) // 2
                    clear_stack()
                elif byte in (19, 20):
                    stem_count += len(stack) // 2
                    clear_stack()
                    pos += (stem_count + 7) // 8
                elif byte == 4:
                    if len(stack) > 1:
                        del stack[:-1]
                    move(0.0, stack[-1] if stack else 0.0)
                    clear_stack()
                elif byte == 21:
                    if len(stack) > 2:
                        del stack[:-2]
                    dx = stack[-2] if len(stack) >= 2 else 0.0
                    dy = stack[-1] if stack else 0.0
                    move(dx, dy)
                    clear_stack()
                elif byte == 22:
                    if len(stack) > 1:
                        del stack[:-1]
                    move(stack[-1] if stack else 0.0, 0.0)
                    clear_stack()
                elif byte == 5:
                    for i in range(0, len(stack) - 1, 2):
                        line(stack[i], stack[i + 1])
                    clear_stack()
                elif byte == 6:
                    horizontal = True
                    for value in stack:
                        line(value, 0.0) if horizontal else line(0.0, value)
                        horizontal = not horizontal
                    clear_stack()
                elif byte == 7:
                    vertical = True
                    for value in stack:
                        line(0.0, value) if vertical else line(value, 0.0)
                        vertical = not vertical
                    clear_stack()
                elif byte == 8:
                    for i in range(0, len(stack) - 5, 6):
                        curve(*stack[i : i + 6])
                    clear_stack()
                elif byte == 10:
                    if stack:
                        subr_index = int(stack.pop()) + subr_bias
                        if 0 <= subr_index < len(local_subrs):
                            if not execute(local_subrs[subr_index], depth + 1):
                                return False
                elif byte == 11:
                    return True
                elif byte == 14:
                    flush_contour()
                    return False
                elif byte == 24:
                    line_count = len(stack) - 6
                    for i in range(0, line_count - 1, 2):
                        line(stack[i], stack[i + 1])
                    if line_count >= 0:
                        curve(*stack[line_count : line_count + 6])
                    clear_stack()
                elif byte == 25:
                    line_count = len(stack) % 6
                    for i in range(0, line_count - 1, 2):
                        line(stack[i], stack[i + 1])
                    for i in range(line_count, len(stack) - 5, 6):
                        curve(*stack[i : i + 6])
                    clear_stack()
                elif byte == 26:
                    if len(stack) % 2:
                        line(stack.pop(0), 0.0)
                    for i in range(0, len(stack) - 3, 4):
                        curve(
                            0.0, stack[i], stack[i + 1], stack[i + 2], 0.0, stack[i + 3]
                        )
                    clear_stack()
                elif byte == 27:
                    if len(stack) % 2:
                        line(0.0, stack.pop(0))
                    for i in range(0, len(stack) - 3, 4):
                        curve(
                            stack[i], 0.0, stack[i + 1], stack[i + 2], stack[i + 3], 0.0
                        )
                    clear_stack()
                elif byte == 29:
                    if stack:
                        subr_index = int(stack.pop()) + gsubr_bias
                        if 0 <= subr_index < len(global_subrs):
                            if not execute(global_subrs[subr_index], depth + 1):
                                return False
                elif byte in (30, 31):
                    horizontal = byte == 31
                    args = list(stack)
                    clear_stack()
                    while len(args) >= 4:
                        if horizontal:
                            dx1 = args.pop(0)
                            dy1 = 0.0
                            dx2 = args.pop(0)
                            dy2 = args.pop(0)
                            dy3 = args.pop(0)
                            dx3 = args.pop(0) if len(args) == 1 else 0.0
                        else:
                            dx1 = 0.0
                            dy1 = args.pop(0)
                            dx2 = args.pop(0)
                            dy2 = args.pop(0)
                            dx3 = args.pop(0)
                            dy3 = args.pop(0) if len(args) == 1 else 0.0
                        curve(dx1, dy1, dx2, dy2, dx3, dy3)
                        horizontal = not horizontal
                else:
                    clear_stack()
            return True
        except IndexError, ValueError:
            return False

    if not execute(charstring):
        return contours
    flush_contour()
    return contours


def _type2_subr_bias(count: int) -> int:
    if count < 1240:
        return 107
    if count < 33900:
        return 1131
    return 32768


def _rasterize_contours(
    contours: list[list[tuple[float, float]]], *, width: int, height: int
) -> tuple[int, ...]:
    points = [point for contour in contours for point in contour]
    if not points or width <= 0 or height <= 0:
        return ()
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    glyph_width = max(max_x - min_x, 1.0)
    glyph_height = max(max_y - min_y, 1.0)
    edges: list[tuple[float, float, float, float]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        normalized = [
            (
                (px - min_x) / glyph_width * (width - 1),
                (py - min_y) / glyph_height * (height - 1),
            )
            for px, py in contour
        ]
        previous = normalized[-1]
        for point in normalized:
            x0, y0 = previous
            x1, y1 = point
            if y0 != y1:
                edges.append((x0, y0, x1, y1))
            previous = point
    if not edges:
        return ()
    rows: list[int] = []
    for y in range(height - 1, -1, -1):
        intersections: list[float] = []
        row = 0
        y_mid = y + 0.5
        for x0, y0, x1, y1 in edges:
            if (y0 > y_mid) == (y1 > y_mid):
                continue
            intersections.append(x0 + (x1 - x0) * (y_mid - y0) / (y1 - y0))
        intersections.sort()
        for index in range(0, len(intersections) - 1, 2):
            start_x = max(0, ceil(intersections[index] - 0.5))
            end_x = min(width - 1, ceil(intersections[index + 1] - 0.5) - 1)
            for x in range(start_x, end_x + 1):
                row |= 1 << x
        rows.append(row)
    return tuple(rows)


@lru_cache(maxsize=4096)
def _feature_cell_distance_map(cells: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    if not cells:
        return ()
    limit = FEATURE_GRID_WIDTH + FEATURE_GRID_HEIGHT
    distances = [limit] * (FEATURE_GRID_WIDTH * FEATURE_GRID_HEIGHT)
    for x, y in cells:
        if 0 <= x < FEATURE_GRID_WIDTH and 0 <= y < FEATURE_GRID_HEIGHT:
            distances[y * FEATURE_GRID_WIDTH + x] = 0
    for y in range(FEATURE_GRID_HEIGHT):
        row = y * FEATURE_GRID_WIDTH
        for x in range(FEATURE_GRID_WIDTH):
            idx = row + x
            best = distances[idx]
            if x:
                best = min(best, distances[idx - 1] + 1)
            if y:
                best = min(best, distances[idx - FEATURE_GRID_WIDTH] + 1)
            distances[idx] = best
    for y in range(FEATURE_GRID_HEIGHT - 1, -1, -1):
        row = y * FEATURE_GRID_WIDTH
        for x in range(FEATURE_GRID_WIDTH - 1, -1, -1):
            idx = row + x
            best = distances[idx]
            if x + 1 < FEATURE_GRID_WIDTH:
                best = min(best, distances[idx + 1] + 1)
            if y + 1 < FEATURE_GRID_HEIGHT:
                best = min(best, distances[idx + FEATURE_GRID_WIDTH] + 1)
            distances[idx] = best
    return tuple(distances)


def _average_nearest_cell_distance(
    cells: tuple[tuple[int, int], ...], distance_map: tuple[int, ...]
) -> float:
    total = 0.0
    count = 0
    for x, y in cells:
        if 0 <= x < FEATURE_GRID_WIDTH and 0 <= y < FEATURE_GRID_HEIGHT:
            total += distance_map[y * FEATURE_GRID_WIDTH + x]
            count += 1
    return total / count if count else inf


def feature_distance(left: CFFGlyphFeature, right: CFFGlyphFeature) -> float:
    if not left.cells or not right.cells:
        return inf
    left_map = _feature_cell_distance_map(left.cells)
    right_map = _feature_cell_distance_map(right.cells)

    return (
        _average_nearest_cell_distance(left.cells, right_map)
        + _average_nearest_cell_distance(right.cells, left_map)
        + _bitmap_distance(left.bitmap, right.bitmap) * 0.75
        + abs(left.aspect - right.aspect) * 2.0
        + abs(left.contours - right.contours) * 0.2
    )


def _bitmap_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    intersection = 0
    union = 0
    for left_row, right_row in zip(left, right, strict=True):
        intersection += (left_row & right_row).bit_count()
        union += (left_row | right_row).bit_count()
    if union == 0:
        return 0.0
    return 1.0 - intersection / union


SUSPICIOUS_TO_UNICODE = {"\ufffd", "£", "•"}
REPAIRABLE_TO_UNICODE = SUSPICIOUS_TO_UNICODE | {"5", "H"}
LEGITIMATE_MULTI_CHAR_GLYPHS = {"ff", "fi", "fl", "ffi", "ffl", "st"}


def is_repairable_to_unicode_label(label: str) -> bool:
    if len(label) == 1:
        return label in REPAIRABLE_TO_UNICODE
    if label in LEGITIMATE_MULTI_CHAR_GLYPHS:
        return False
    if any(ch in SUSPICIOUS_TO_UNICODE for ch in label):
        return True
    if len(label) > 3:
        return True
    return any(not (ch.isalnum() or ch.isspace()) for ch in label)


def _repair_candidate(
    glyph_id: int,
    label: str,
    features: dict[int, CFFGlyphFeature],
    labels: dict[int, str],
) -> str | None:
    feature = features.get(glyph_id, EMPTY_FEATURE)
    if not feature.cells:
        return None
    candidates: list[tuple[float, str]] = []
    same_label = inf
    for other_id, other_label in labels.items():
        if other_id == glyph_id or len(other_label) != 1:
            continue
        if not (other_label.isalnum() or other_label in ".-+"):
            continue
        other_feature = features.get(other_id, EMPTY_FEATURE)
        if not other_feature.cells:
            continue
        distance = feature_distance(feature, other_feature)
        if other_label == label:
            same_label = min(same_label, distance)
        else:
            candidates.append((distance, other_label))
    if not candidates:
        return None
    best_distance, best_label = min(candidates, key=lambda item: item[0])
    if (label in SUSPICIOUS_TO_UNICODE or len(label) > 1) and best_distance < 2.3:
        return best_label
    if label == "5" and best_label == "S" and best_distance < 1.9:
        return best_label
    if label == "H" and best_label == "M" and best_distance < 1.8:
        return best_label
    if same_label < inf and best_distance + 0.35 < same_label and best_distance < 2.0:
        return best_label
    return None


@lru_cache(maxsize=64)
def cff_unicode_repairs_for_data(
    font_data: bytes, mapping_items: tuple[tuple[bytes, int, str], ...]
) -> tuple[tuple[bytes, str], ...]:
    font = CFFFont(font_data)
    glyph_count = len(font.charstrings)
    if glyph_count < 2:
        return ()
    labels: dict[int, str] = {}
    gid_to_codes: dict[int, list[bytes]] = {}
    for code_bytes, cid, value in mapping_items:
        gid = font.glyph_id_for_cid(cid)
        if gid < glyph_count:
            labels[gid] = value
            gid_to_codes.setdefault(gid, []).append(code_bytes)
    if not labels:
        return ()
    useful_gids = {
        gid
        for gid, label in labels.items()
        if (
            is_repairable_to_unicode_label(label)
            or (len(label) == 1 and (label.isalnum() or label in ".-+"))
        )
    }
    if not useful_gids:
        return ()
    features = {gid: font.glyph_feature(gid) for gid in useful_gids}
    repairs: dict[bytes, str] = {}
    for glyph_id, label in labels.items():
        if not is_repairable_to_unicode_label(label):
            continue
        replacement = _repair_candidate(glyph_id, label, features, labels)
        if replacement is not None and replacement != label:
            for code_bytes in gid_to_codes.get(glyph_id, ()):
                repairs[code_bytes] = replacement
    return tuple(sorted(repairs.items()))


@lru_cache(maxsize=64)
def cff_font_for_data(font_data: bytes) -> CFFFont:
    return CFFFont(font_data)


__all__ = (
    "CFFFont",
    "CFFGlyphFeature",
    "REPAIRABLE_TO_UNICODE",
    "cff_font_for_data",
    "cff_unicode_repairs_for_data",
    "feature_distance",
    "is_repairable_to_unicode_label",
    "type2_glyph_bitmap",
)
