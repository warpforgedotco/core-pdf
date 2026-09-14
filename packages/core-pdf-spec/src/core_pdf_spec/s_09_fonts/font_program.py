"""CFF tables, dictionaries, charsets, and Type 2 program semantics."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite, sqrt

from core_pdf_spec._vendor.font_data.cff_tables import (
    CFF_EXPERT_STRINGS,
    CFF_EXPERT_SUBSET_STRINGS,
    CFF_ISO_ADOBE_STRINGS,
    CFF_STANDARD_STRINGS,
)
from core_pdf_spec._vendor.font_data.encoding_names import STANDARD_ENCODING_GLYPH_NAMES
from core_pdf_spec.s_08_graphics.matrix import Matrix

STANDARD_GLYPH_SIDS = {name: sid for sid, name in enumerate(CFF_STANDARD_STRINGS)}


CFF_STANDARD_STRING_COUNT = len(CFF_STANDARD_STRINGS)


TYPE2_MAX_SUBR_DEPTH = 10


internal_TYPE2_MAX_STACK = 48


internal_TYPE2_TRANSIENT_SIZE = 32


DEFAULT_CFF_FONT_MATRIX = Matrix(0.001, 0.0, 0.0, 0.001, 0.0, 0.0)


CFF_EXPERT_ENCODING_CODES = tuple(
    code
    for code in (*range(32, 127), *range(161, 256))
    if code
    not in {
        35,
        64,
        70,
        71,
        72,
        74,
        75,
        80,
        81,
        85,
        92,
        164,
        165,
        171,
        173,
        174,
        176,
        177,
        180,
        181,
        185,
        186,
        187,
        198,
        199,
    }
)


def internal_cff_offset(values: list[float] | None, context: str) -> int:
    """Validate a single DICT offset before conversion or indexing."""
    if not values or len(values) != 1:
        raise ValueError(f"invalid CFF {context}")
    value = values[0]
    if not isfinite(value) or value < 0 or int(value) != value:
        raise ValueError(f"invalid CFF {context}")
    return int(value)


def cff_font_matrix(
    font_dict: dict[int | tuple[int, int], list[float]],
) -> Matrix | None:
    values = font_dict.get((12, 7))
    if values is None:
        return None
    if not isinstance(values, list):
        raise ValueError("invalid CFF FontMatrix")
    try:
        return Matrix.from_operand(values)
    except ValueError as exc:
        raise ValueError("invalid CFF FontMatrix") from exc


class CFFFont:
    """One CFF 1 font with strict table readers and Type 2 program data.

    The ``read_*``, ``parse_*``, ``dict_offset``, ``local_subrs_for_glyph``,
    and ``font_matrix`` methods are supported
    extension points. Readers return values without changing parser state and
    raise ValueError for malformed input. Construction initializes ``data``
    first, then the string/global-subroutine indexes, ``top_dict`` and
    ``is_cid_keyed``, ``charstrings``, ``cid_to_gid``, ``fd_select``,
    ``font_dicts``, and ``local_subrs`` in that order. An overriding reader may
    rely on earlier fields only; callers must discard a failed construction.
    """

    __slots__ = (
        "data",
        "top_dict",
        "charstrings",
        "cid_to_gid",
        "custom_string_sids",
        "is_cid_keyed",
        "global_subrs",
        "local_subrs",
        "fd_select",
        "font_dicts",
    )

    def __init__(self, data: bytes | memoryview | None) -> None:
        if data is None:
            raise ValueError("missing CFF font program")
        # Keep a caller-owned read-only view when one is provided.  INDEX
        # entries are still materialized as bytes below because they escape
        # the parser and remain stable identifiers within the font program.
        self.data = data
        pos = self.read_header()
        ignored_names, pos = self.read_index(pos)
        top_index, pos = self.read_index(pos)
        custom_strings, pos = self.read_index(pos)
        self.custom_string_sids = {
            value.decode("latin-1"): CFF_STANDARD_STRING_COUNT + index
            for index, value in enumerate(custom_strings)
        }
        global_subrs, pos = self.read_index(pos)
        self.global_subrs = tuple(global_subrs)
        if not top_index:
            raise ValueError("invalid CFF top dict")
        self.top_dict = self.parse_dict(top_index[0])
        self.is_cid_keyed = (12, 30) in self.top_dict
        self.charstrings, ignored_pos = self.read_index(self.dict_offset(17))
        self.cid_to_gid = self.read_charset(self.dict_offset(15, default=0), len(self.charstrings))
        self.fd_select = self.read_fd_select()
        self.font_dicts = self.read_font_dicts()
        self.local_subrs = self.read_local_subrs()

    def read_header(self) -> int:
        """Return the first INDEX offset after validating the CFF 1 header."""
        if len(self.data) < 4 or self.data[0] != 1:
            raise ValueError("invalid CFF font program")
        size = self.data[2]
        if not 4 <= size <= len(self.data) or not 1 <= self.data[3] <= 4:
            raise ValueError("invalid CFF header")
        return size

    def dict_offset(self, operator: int, *, default: int | None = None) -> int:
        """Read a nonnegative integral Top DICT offset; default applies only to absence."""
        values = self.top_dict.get(operator)
        if values is None and default is not None:
            return default
        return internal_cff_offset(values, "dictionary offset")

    def read_index(self, pos: int) -> tuple[list[bytes], int]:
        data = memoryview(self.data)
        if pos < 0 or pos + 2 > len(data):
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
            [bytes(data[base + offsets[i] - 1 : base + offsets[i + 1] - 1]) for i in range(count)],
            end,
        )

    @staticmethod
    def parse_number(item: bytes, pos: int, *, dict_number: bool = False) -> tuple[float, int]:
        if not 0 <= pos < len(item):
            raise ValueError("invalid CFF number offset")
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
            return CFFFont.parse_real_number(item, pos + 1)
        if b0 == 255 and not dict_number:
            if pos + 5 > len(item):
                raise ValueError("invalid Type 2 number")
            return (
                int.from_bytes(item[pos + 1 : pos + 5], "big", signed=True) / 65536.0,
                pos + 5,
            )
        raise ValueError("invalid CFF number")

    @staticmethod
    def parse_real_number(item: bytes, pos: int) -> tuple[float, int]:
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
                elif nibble == 13:
                    # 0xd is reserved by the CFF real-number encoding. Treating
                    # it as whitespace silently joins the surrounding digits.
                    raise ValueError("invalid CFF real number")
                elif nibble == 14:
                    parts.append("-")
        raise ValueError("invalid CFF real number")

    def parse_dict(self, item: bytes) -> dict[int | tuple[int, int], list[float]]:
        result, trailing = self.read_dict_entries(item)
        if trailing:
            raise ValueError("unterminated CFF dictionary operands")
        return result

    def read_dict_entries(
        self, item: bytes
    ) -> tuple[dict[int | tuple[int, int], list[float]], list[float]]:
        """Return operator entries and any unconsumed trailing operands."""
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
                value, pos = self.parse_number(item, pos, dict_number=True)
                stack.append(value)
        return result, stack

    def glyph_id_for_cid(self, cid: int) -> int:
        if self.is_cid_keyed:
            return self.cid_to_gid.get(cid, 0)
        return cid

    def glyph_id_for_name(self, name: str) -> int:
        sid = STANDARD_GLYPH_SIDS.get(name)
        if sid is None:
            sid = self.custom_string_sids.get(name)
        if sid is None:
            return 0
        return self.cid_to_gid.get(sid, 0)

    def has_glyph_id(self, gid: int) -> bool:
        return 0 <= gid < len(self.charstrings)

    def read_local_subrs(self) -> tuple[tuple[bytes, ...], ...]:
        if self.is_cid_keyed:
            return tuple(tuple(self.read_private_subrs(font_dict)) for font_dict in self.font_dicts)
        return (tuple(self.read_private_subrs(self.top_dict)),)

    def read_charset(self, pos: int, glyph_count: int) -> dict[int, int]:
        if glyph_count < 1:
            raise ValueError("CFF charset has no .notdef glyph")
        if pos in {0, 1, 2}:
            if self.is_cid_keyed:
                raise ValueError("CID-keyed CFF font uses a predefined charset")
            names = (CFF_ISO_ADOBE_STRINGS, CFF_EXPERT_STRINGS, CFF_EXPERT_SUBSET_STRINGS)[pos]
            if glyph_count > len(names):
                raise ValueError("CFF predefined charset is too short")
            return {STANDARD_GLYPH_SIDS[name]: gid for gid, name in enumerate(names[:glyph_count])}
        if glyph_count == 1:
            return {0: 0}
        if not 0 <= pos < len(self.data):
            raise ValueError("invalid CFF charset offset")
        fmt = self.data[pos]
        pos += 1
        if fmt not in {0, 1, 2}:
            raise ValueError("invalid CFF charset format")
        mapping = {0: 0}
        gid = 1
        while gid < glyph_count:
            size = 2 if fmt == 0 else 3 if fmt == 1 else 4
            if pos + size > len(self.data):
                raise ValueError("truncated CFF charset")
            first = int.from_bytes(self.data[pos : pos + 2], "big")
            count = 1 if fmt == 0 else 1 + int.from_bytes(self.data[pos + 2 : pos + size], "big")
            if gid + count > glyph_count:
                raise ValueError("CFF charset range exceeds glyph count")
            for offset in range(count):
                if first + offset in mapping:
                    raise ValueError("duplicate CFF charset entry")
                mapping[first + offset] = gid + offset
            gid += count
            pos += size
        return mapping

    def read_encoding_codes(self, pos: int) -> dict[int, int]:
        if not 0 < pos < len(self.data):
            raise ValueError("invalid CFF encoding offset")
        raw_format = self.data[pos]
        fmt = raw_format & 127
        if fmt not in {0, 1} or pos + 1 >= len(self.data):
            raise ValueError("invalid CFF encoding")
        count = self.data[pos + 1]
        pos += 2
        codes: dict[int, int] = {}
        gid = 1
        for _ in range(count):
            size = 1 if fmt == 0 else 2
            if pos + size > len(self.data):
                raise ValueError("truncated CFF encoding")
            first = self.data[pos]
            length = 1 if fmt == 0 else self.data[pos + 1] + 1
            if first + length > 256 or gid + length > len(self.charstrings):
                raise ValueError("CFF encoding range exceeds font")
            for offset in range(length):
                if first + offset in codes:
                    raise ValueError("duplicate CFF encoding code")
                codes[first + offset] = gid + offset
            pos += size
            gid += length
        if raw_format & 128:
            if pos >= len(self.data):
                raise ValueError("truncated CFF encoding supplements")
            count = self.data[pos]
            pos += 1
            for _ in range(count):
                if pos + 3 > len(self.data):
                    raise ValueError("truncated CFF encoding supplement")
                code = self.data[pos]
                sid = int.from_bytes(self.data[pos + 1 : pos + 3], "big")
                if sid not in self.cid_to_gid:
                    raise ValueError("CFF encoding supplement references missing glyph")
                codes[code] = self.cid_to_gid[sid]
                pos += 3
        return codes

    def builtin_encoding(self) -> dict[int, str]:
        if self.is_cid_keyed:
            return {}
        offset = self.dict_offset(16, default=0)
        if offset == 0:
            return {
                code: name
                for code, name in enumerate(STANDARD_ENCODING_GLYPH_NAMES)
                if name != ".notdef"
            }
        if offset == 1:
            return {
                code: name
                for code, name in zip(
                    CFF_EXPERT_ENCODING_CODES, CFF_EXPERT_STRINGS[1:], strict=True
                )
                if STANDARD_GLYPH_SIDS[name] in self.cid_to_gid
            }
        reverse = {gid: sid for sid, gid in self.cid_to_gid.items()}
        names = dict(enumerate(CFF_STANDARD_STRINGS)) | {
            sid: name for name, sid in self.custom_string_sids.items()
        }
        return {code: names[reverse[gid]] for code, gid in self.read_encoding_codes(offset).items()}

    def read_fd_select(self) -> tuple[int, ...]:
        count = len(self.charstrings)
        if not self.is_cid_keyed:
            return (0,) * count
        values = self.top_dict.get((12, 37))
        pos = internal_cff_offset(values, "FDSelect offset")
        if not 0 <= pos < len(self.data):
            raise ValueError("invalid CFF FDSelect offset")
        fmt = self.data[pos]
        pos += 1
        if fmt == 0 and pos + count <= len(self.data):
            return tuple(self.data[pos : pos + count])
        if fmt != 3 or pos + 2 > len(self.data):
            raise ValueError("invalid CFF FDSelect")
        ranges = int.from_bytes(self.data[pos : pos + 2], "big")
        pos += 2
        if not ranges or pos + ranges * 3 + 2 > len(self.data):
            raise ValueError("truncated CFF FDSelect")
        entries = [
            (
                int.from_bytes(self.data[pos + i * 3 : pos + i * 3 + 2], "big"),
                self.data[pos + i * 3 + 2],
            )
            for i in range(ranges)
        ]
        sentinel = int.from_bytes(self.data[pos + ranges * 3 : pos + ranges * 3 + 2], "big")
        if entries[0][0] != 0 or sentinel != count:
            raise ValueError("invalid CFF FDSelect bounds")
        selection: list[int] = []
        for index, (first, fd) in enumerate(entries):
            last = entries[index + 1][0] if index + 1 < len(entries) else sentinel
            if last <= first:
                raise ValueError("invalid CFF FDSelect range")
            selection.extend([fd] * (last - first))
        return tuple(selection)

    def read_font_dicts(self) -> tuple[dict[int | tuple[int, int], list[float]], ...]:
        if not self.is_cid_keyed:
            return ()
        values = self.top_dict.get((12, 36))
        items, _ = self.read_index(internal_cff_offset(values, "FDArray offset"))
        return tuple(self.parse_dict(item) for item in items)

    def read_private_subrs(
        self, font_dict: dict[int | tuple[int, int], list[float]]
    ) -> list[bytes]:
        private = font_dict.get(18)
        if private is None:
            return []
        if len(private) != 2 or any(
            not isfinite(value) or int(value) != value for value in private
        ):
            raise ValueError("invalid CFF Private dictionary")
        size, offset = map(int, private)
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise ValueError("invalid CFF Private bounds")
        private_dict = self.parse_dict(bytes(self.data[offset : offset + size]))
        subrs = private_dict.get(19)
        if subrs is None:
            return []
        items, _ = self.read_index(offset + internal_cff_offset(subrs, "Subrs offset"))
        return items

    def local_subrs_for_glyph(self, glyph_id: int) -> tuple[bytes, ...]:
        if not 0 <= glyph_id < len(self.charstrings):
            raise ValueError("invalid CFF glyph id")
        fd = self.fd_select[glyph_id]
        if not 0 <= fd < len(self.local_subrs):
            raise ValueError("invalid CFF font dictionary index")
        return self.local_subrs[fd]

    def font_matrix(self, glyph_id: int) -> Matrix:
        if not self.has_glyph_id(glyph_id) or glyph_id >= len(self.fd_select):
            raise ValueError("invalid CFF glyph id")
        top = cff_font_matrix(self.top_dict)
        fd = self.fd_select[glyph_id]
        if self.font_dicts and not 0 <= fd < len(self.font_dicts):
            raise ValueError("invalid CFF font dictionary index")
        child = cff_font_matrix(self.font_dicts[fd]) if self.font_dicts else None
        if top is None:
            return child or DEFAULT_CFF_FONT_MATRIX
        return top if child is None else child.multiply(top)


def cubic_extrema_times(p0: float, p1: float, p2: float, p3: float) -> tuple[float, ...]:
    """Return the interior extrema parameters of one cubic coordinate."""
    # The derivative's Bernstein coefficients are the adjacent control-point
    # differences. If they have one sign, the coordinate is monotone and no
    # quadratic root solving is needed (including constant coordinates).
    if p0 <= p1 <= p2 <= p3 or p0 >= p1 >= p2 >= p3:
        return ()
    a = -p0 + 3.0 * p1 - 3.0 * p2 + p3
    b = 2.0 * (p0 - 2.0 * p1 + p2)
    c = p1 - p0
    epsilon = 1e-12
    if abs(a) <= epsilon:
        if abs(b) <= epsilon:
            return ()
        root = -c / b
        return (root,) if 0.0 < root < 1.0 else ()
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return ()
    root_delta = sqrt(discriminant)
    roots = ((-b - root_delta) / (2.0 * a), (-b + root_delta) / (2.0 * a))
    return tuple(dict.fromkeys(root for root in roots if 0.0 < root < 1.0))


def cubic_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    mt = 1.0 - t
    # Every coefficient is shared by the x and y line, so bind them once. Keep
    # the ** form: mt**3 and mt*mt*mt disagree on about a quarter of random
    # floats, which would move the golden rasters.
    mt3 = mt**3
    t3 = t**3
    mt2t = 3.0 * mt * mt * t
    mtt2 = 3.0 * mt * t * t
    return (
        mt3 * p0[0] + mt2t * p1[0] + mtt2 * p2[0] + t3 * p3[0],
        mt3 * p0[1] + mt2t * p1[1] + mtt2 * p2[1] + t3 * p3[1],
    )


def internal_execute_type2_flex(
    operator: int,
    operands: list[float],
    curve: Callable[[float, float, float, float, float, float], None],
) -> None:
    """Execute one of the four escaped Type 2 flex operators."""
    match operator:
        case 34:  # hflex
            dx1, dx2, dy2, dx3, dx4, dx5, dx6 = operands
            curve(dx1, 0.0, dx2, dy2, dx3, 0.0)
            curve(dx4, 0.0, dx5, -dy2, dx6, 0.0)
        case 35:  # flex
            (
                dx1,
                dy1,
                dx2,
                dy2,
                dx3,
                dy3,
                dx4,
                dy4,
                dx5,
                dy5,
                dx6,
                dy6,
                ignored_flex_depth,
            ) = operands
            curve(dx1, dy1, dx2, dy2, dx3, dy3)
            curve(dx4, dy4, dx5, dy5, dx6, dy6)
        case 36:  # hflex1
            dx1, dy1, dx2, dy2, dx3, dx4, dx5, dy5, dx6 = operands
            dy6 = -(dy1 + dy2 + dy5)
            curve(dx1, dy1, dx2, dy2, dx3, 0.0)
            curve(dx4, 0.0, dx5, dy5, dx6, dy6)
        case 37:  # flex1
            dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, dx5, dy5, d6 = operands
            dx = dx1 + dx2 + dx3 + dx4 + dx5
            dy = dy1 + dy2 + dy3 + dy4 + dy5
            if abs(dx) > abs(dy):
                dx6, dy6 = d6, -dy
            else:
                dx6, dy6 = -dx, d6
            curve(dx1, dy1, dx2, dy2, dx3, dy3)
            curve(dx4, dy4, dx5, dy5, dx6, dy6)
        case _:
            raise ValueError("invalid Type 2 flex operator")


def internal_type2_subr_bias(count: int) -> int:
    if count < 1240:
        return 107
    if count < 33900:
        return 1131
    return 32768


def execute_type2_charstring(  # noqa: C901 - direct dispatch mirrors Type 2 operators
    charstring: bytes,
    *,
    local_subrs: tuple[bytes, ...],
    global_subrs: tuple[bytes, ...],
    move: Callable[[float, float], None],
    line: Callable[[float, float], None],
    curve: Callable[[float, float, float, float, float, float], None],
    flush_contour: Callable[[], None],
    has_current_point: Callable[[], bool],
    seac: Callable[[int, int, float, float], None],
    random_value: Callable[[], float],
) -> bool:
    """Execute Type 2 operators against a caller-owned geometric path sink.

    Return true when the program exhausts without endchar; malformed programs
    raise. Sampling, partial-path retention and random sources are caller choices.
    """
    stack: list[float] = []
    transient = [0.0] * internal_TYPE2_TRANSIENT_SIZE
    stem_count = 0
    width_resolved = False
    subr_bias = internal_type2_subr_bias(len(local_subrs))
    gsubr_bias = internal_type2_subr_bias(len(global_subrs))

    def push(value: float) -> None:
        if len(stack) >= internal_TYPE2_MAX_STACK or not isfinite(value):
            raise ValueError("invalid Type 2 operand stack")
        stack.append(float(value))

    def require_integer(value: float) -> int:
        integer = int(value)
        if value != integer:
            raise ValueError("Type 2 operator requires an integer")
        return integer

    def pop_integer() -> int:
        return require_integer(stack.pop())

    def execute_escaped_operator(operator: int) -> None:
        match operator:
            case 0:  # dotsection -- deprecated no-op with a clearing stack contract
                stack.clear()
            case 3:  # and
                second = stack.pop()
                first = stack.pop()
                push(float(first != 0.0 and second != 0.0))
            case 4:  # or
                second = stack.pop()
                first = stack.pop()
                push(float(first != 0.0 or second != 0.0))
            case 5:  # not
                push(float(stack.pop() == 0.0))
            case 9:  # abs
                push(abs(stack.pop()))
            case 10:  # add
                second = stack.pop()
                push(stack.pop() + second)
            case 11:  # sub
                second = stack.pop()
                push(stack.pop() - second)
            case 12:  # div
                second = stack.pop()
                push(stack.pop() / second)
            case 14:  # neg
                push(-stack.pop())
            case 15:  # eq
                second = stack.pop()
                push(float(stack.pop() == second))
            case 18:  # drop
                stack.pop()
            case 20:  # put
                index = pop_integer()
                value = stack.pop()
                if not 0 <= index < len(transient):
                    raise ValueError("invalid Type 2 transient-array index")
                transient[index] = value
            case 21:  # get
                index = pop_integer()
                if not 0 <= index < len(transient):
                    raise ValueError("invalid Type 2 transient-array index")
                push(transient[index])
            case 22:  # ifelse
                value2 = stack.pop()
                value1 = stack.pop()
                choice2 = stack.pop()
                choice1 = stack.pop()
                push(choice1 if value1 <= value2 else choice2)
            case 23:  # random
                push(random_value())
            case 24:  # mul
                second = stack.pop()
                push(stack.pop() * second)
            case 26:  # sqrt
                push(sqrt(stack.pop()))
            case 27:  # dup
                push(stack[-1])
            case 28:  # exch
                stack[-1], stack[-2] = stack[-2], stack[-1]
            case 29:  # index
                index = max(pop_integer(), 0)
                if index >= len(stack):
                    raise ValueError("invalid Type 2 stack index")
                push(stack[-index - 1])
            case 30:  # roll
                shift = pop_integer()
                count = pop_integer()
                if count < 0 or count > len(stack):
                    raise ValueError("invalid Type 2 roll count")
                if count:
                    shift %= count
                    if shift:
                        values = stack[-count:]
                        stack[-count:] = values[-shift:] + values[:-shift]
            case 34 | 35 | 36 | 37:  # hflex / flex / hflex1 / flex1
                if not has_current_point():
                    raise ValueError("Type 2 flex operator has no current point")
                internal_execute_type2_flex(operator, stack, curve)
                stack.clear()
            case _:
                raise ValueError("unsupported Type 2 escaped operator")

    def execute(  # noqa: C901 - keeping operator cases together makes the bytecode contract auditable
        program: bytes, depth: int = 0
    ) -> bool:
        """Interpret one Type 2 charstring, appending to the enclosing contour state.

        Return ``True`` on exhaustion or ``return``, and ``False`` when ``endchar``
        has flushed the contour. Malformed programs raise.

        The branches below are keyed by raw Type 2 operator bytes; each carries the
        operator's spec name. Operands are values above 31, plus 28 (a two-byte
        integer) and 255 (a 16.16 fixed-point number).
        """
        nonlocal stem_count, width_resolved
        if depth > TYPE2_MAX_SUBR_DEPTH:
            raise ValueError("invalid Type 2 charstring")
        pos = 0
        try:
            while pos < len(program):
                byte = program[pos]
                if byte > 31 or byte in {28, 255}:
                    value, pos = CFFFont.parse_number(program, pos)
                    push(value)
                    continue
                pos += 1
                match byte:
                    case 1 | 3 | 18 | 23:  # hstem, vstem, hstemhm, vstemhm
                        operand_count = len(stack)
                        if not width_resolved and operand_count % 2:
                            operand_count -= 1
                        if operand_count < 2 or operand_count % 2:
                            raise ValueError("invalid Type 2 charstring")
                        stem_count += operand_count // 2
                        if stem_count > 96:
                            raise ValueError("invalid Type 2 charstring")
                        width_resolved = True
                        stack.clear()
                    case 4 | 22:  # vmoveto / hmoveto
                        if len(stack) == 1:
                            displacement = stack[0]
                        elif not width_resolved and len(stack) == 2:
                            displacement = stack[1]
                        else:
                            raise ValueError("invalid Type 2 charstring")
                        width_resolved = True
                        if byte == 4:
                            move(0.0, displacement)
                        else:
                            move(displacement, 0.0)
                        stack.clear()
                    case 5:  # rlineto
                        if not has_current_point() or len(stack) < 2 or len(stack) % 2:
                            raise ValueError("invalid Type 2 charstring")
                        for i in range(0, len(stack) - 1, 2):
                            line(stack[i], stack[i + 1])
                        stack.clear()
                    case 6 | 7:  # hlineto / vlineto -- alternating axes
                        if not has_current_point() or not stack:
                            raise ValueError("invalid Type 2 charstring")
                        horizontal = byte == 6
                        for value in stack:
                            line(value, 0.0) if horizontal else line(0.0, value)
                            horizontal = not horizontal
                        stack.clear()
                    case 8:  # rrcurveto
                        if not has_current_point() or len(stack) < 6 or len(stack) % 6:
                            raise ValueError("invalid Type 2 charstring")
                        for i in range(0, len(stack) - 5, 6):
                            curve(*stack[i : i + 6])
                        stack.clear()
                    case 10 | 29:  # callsubr / callgsubr
                        if not stack:
                            raise ValueError("invalid Type 2 charstring")
                        subrs, bias = (
                            (local_subrs, subr_bias) if byte == 10 else (global_subrs, gsubr_bias)
                        )
                        subr_index = pop_integer() + bias
                        if not 0 <= subr_index < len(subrs):
                            raise ValueError("invalid Type 2 charstring")
                        # Type 2, 4.2 note 6 permits endchar in a subroutine;
                        # it completes the glyph through every enclosing call.
                        if not execute(subrs[subr_index], depth + 1):
                            return False
                    case 11:  # return -- leave this subroutine, caller keeps going
                        return True
                    case 12:  # two-byte escaped operator
                        if pos >= len(program):
                            raise ValueError("invalid Type 2 charstring")
                        escaped_operator = program[pos]
                        pos += 1
                        execute_escaped_operator(escaped_operator)
                    case 14:  # endchar -- glyph complete
                        arguments = list(stack)
                        if not width_resolved:
                            if len(arguments) in {1, 5}:
                                arguments = arguments[1:]
                            elif len(arguments) not in {0, 4}:
                                raise ValueError("invalid Type 2 charstring")
                            width_resolved = True
                        elif len(arguments) not in {0, 4}:
                            raise ValueError("invalid Type 2 charstring")
                        stack.clear()
                        flush_contour()
                        if arguments:
                            seac(
                                require_integer(arguments[2]),
                                require_integer(arguments[3]),
                                arguments[0],
                                arguments[1],
                            )
                        return False  # endchar completes the glyph
                    case 19 | 20:  # hintmask, cntrmask -- skip trailing mask bytes
                        operand_count = len(stack)
                        if not width_resolved and operand_count % 2:
                            operand_count -= 1
                        if operand_count % 2:
                            raise ValueError("invalid Type 2 charstring")
                        stem_count += operand_count // 2
                        if stem_count <= 0 or stem_count > 96:
                            raise ValueError("invalid Type 2 charstring")
                        mask_bytes = (stem_count + 7) // 8
                        if pos + mask_bytes > len(program):
                            raise ValueError("invalid Type 2 charstring")
                        width_resolved = True
                        stack.clear()
                        pos += mask_bytes
                    case 21:  # rmoveto
                        if len(stack) == 2:
                            dx, dy = stack
                        elif not width_resolved and len(stack) == 3:
                            dx, dy = stack[1:]
                        else:
                            raise ValueError("invalid Type 2 charstring")
                        width_resolved = True
                        move(dx, dy)
                        stack.clear()
                    case 24:  # rcurveline -- curves followed by exactly one line
                        if not has_current_point() or len(stack) < 8 or (len(stack) - 2) % 6:
                            raise ValueError("invalid Type 2 charstring")
                        curve_args = stack[:-2]
                        for i in range(0, len(curve_args) - 5, 6):
                            curve(*curve_args[i : i + 6])
                        line(stack[-2], stack[-1])
                        stack.clear()
                    case 25:  # rlinecurve -- lines followed by exactly one curve
                        if not has_current_point() or len(stack) < 8 or (len(stack) - 6) % 2:
                            raise ValueError("invalid Type 2 charstring")
                        line_args = stack[:-6]
                        for i in range(0, len(line_args) - 1, 2):
                            line(line_args[i], line_args[i + 1])
                        curve(*stack[-6:])
                        stack.clear()
                    case 26 | 27:  # vvcurveto / hhcurveto
                        if (
                            not has_current_point()
                            or len(stack) < 4
                            or len(stack) % 4 not in {0, 1}
                        ):
                            raise ValueError("invalid Type 2 charstring")
                        first_offset = stack.pop(0) if len(stack) % 2 else 0.0
                        for i in range(0, len(stack) - 3, 4):
                            first, dx2, dy2, last = stack[i : i + 4]
                            if byte == 26:
                                curve(first_offset, first, dx2, dy2, 0.0, last)
                            else:
                                curve(first, first_offset, dx2, dy2, last, 0.0)
                            first_offset = 0.0
                        stack.clear()
                    case 30 | 31:  # vhcurveto / hvcurveto -- alternating tangents
                        if (
                            not has_current_point()
                            or len(stack) < 4
                            or len(stack) % 4 not in {0, 1}
                        ):
                            raise ValueError("invalid Type 2 charstring")
                        horizontal = byte == 31
                        args = list(stack)
                        stack.clear()
                        while len(args) >= 4:
                            if horizontal:  # this segment starts horizontal
                                dx1 = args.pop(0)
                                dy1 = 0.0
                                dx2 = args.pop(0)
                                dy2 = args.pop(0)
                                dy3 = args.pop(0)
                                dx3 = args.pop(0) if len(args) == 1 else 0.0
                            else:  # this segment starts vertical
                                dx1 = 0.0
                                dy1 = args.pop(0)
                                dx2 = args.pop(0)
                                dy2 = args.pop(0)
                                dx3 = args.pop(0)
                                dy3 = args.pop(0) if len(args) == 1 else 0.0
                            curve(dx1, dy1, dx2, dy2, dx3, dy3)
                            horizontal = not horizontal
                    case _:
                        raise ValueError("invalid Type 2 charstring")
            return True
        except (ArithmeticError, IndexError, ValueError) as exc:
            raise ValueError("invalid Type 2 charstring") from exc

    return execute(charstring)


__all__ = [
    "STANDARD_GLYPH_SIDS",
    "CFF_STANDARD_STRING_COUNT",
    "TYPE2_MAX_SUBR_DEPTH",
    "DEFAULT_CFF_FONT_MATRIX",
    "CFF_EXPERT_ENCODING_CODES",
    "cff_font_matrix",
    "CFFFont",
    "cubic_extrema_times",
    "cubic_point",
    "execute_type2_charstring",
]
