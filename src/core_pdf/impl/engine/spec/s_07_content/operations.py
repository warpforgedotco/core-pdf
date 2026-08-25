# SPDX-License-Identifier: AGPL-3.0-only
"""Tokenize and dispatch content-stream operators."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, TypeAlias, TypeVar, cast, overload

from core_pdf.impl.engine.spec.s_07_content.inline_images import (
    InlineImage,
    parse_inline_image,
    recover_inline_image_position,
)
from core_pdf.impl.engine.spec.s_07_objects.object_cache import CachedPdfObject
from core_pdf.impl.engine.spec.s_07_syntax.content_operators import (
    GRAPHICS_STATE_OPERATORS,
    IMAGE_OPERATORS,
    TEXT_ONLY_SKIP_DOUBLE,
    TEXT_ONLY_SKIP_SINGLE,
    TEXT_OPERATORS,
    VECTOR_PAINT_OPERATORS,
    VECTOR_PATH_OPERATORS,
)
from core_pdf.impl.engine.spec.s_07_syntax.lexer import (
    EMPTY_SIMPLE_TJ_ARRAY,
    PdfLexer,
)
from core_pdf.impl.engine.spec.s_07_syntax.lexer_helpers import full_source_bytes
from core_pdf.impl.engine.spec.s_07_syntax.scanning import (
    skip_comment,
    skip_hex_string,
    skip_literal_string,
    skip_name,
)
from core_pdf.impl.engine.spec.s_07_syntax.tokens import SEPARATOR_TABLE, WS_TABLE
from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.primitives import PdfName, PdfString

PdfName_of = PdfName.of
FONT_DIGIT_NAMES = tuple(PdfName_of(b"F" + bytes((48 + i,))) for i in range(10))
CS_DIGIT_NAMES = tuple(PdfName_of(b"CS" + bytes((48 + i,))) for i in range(10))
TT_DIGIT_NAMES = tuple(PdfName_of(b"TT" + bytes((48 + i,))) for i in range(10))
P_NAME = PdfName_of(b"P")

ContentOperand: TypeAlias = CachedPdfObject | InlineImage
ContentOperands: TypeAlias = tuple[ContentOperand, ...]
ContentOperation: TypeAlias = tuple[str, ContentOperands]

WORD_BREAK_OR_WS = SEPARATOR_TABLE

IS_WORD_START = bytes([0 if SEPARATOR_TABLE[i] else 1 for i in range(256)])

TEXT_CLIP_PREFIX_RE = re.compile(
    b"[\x00\t\n\f\r ]*"
    b"[+\\-.0-9][^\x00\t\n\f\r ()<>\\[\\]{}%/]*[\x00\t\n\f\r ]+"
    b"[+\\-.0-9][^\x00\t\n\f\r ()<>\\[\\]{}%/]*[\x00\t\n\f\r ]+"
    b"[+\\-.0-9][^\x00\t\n\f\r ()<>\\[\\]{}%/]*[\x00\t\n\f\r ]+"
    b"[+\\-.0-9][^\x00\t\n\f\r ()<>\\[\\]{}%/]*[\x00\t\n\f\r ]+"
    b"re[\x00\t\n\f\r ]+W[\x00\t\n\f\r ]+n"
)
TEXT_SHOWING_CANDIDATES = (b'"', b"'", b"Tj", b"TJ", b"Do")
TEXT_OR_LEXICAL_MARKER_RE = re.compile(rb"""[%(/<>\[\]"']|T[jJ]|Do|BI""")
CONTAINER_LEXICAL_MARKER_RE = re.compile(rb"[%(<>\[\]]")


@dataclass(frozen=True, slots=True)
class ContentOperatorCounts:
    """Coarse content-stream operator counts for cheap page preflight."""

    text: int = 0
    image: int = 0
    vector_path: int = 0
    vector_paint: int = 0
    graphics_state: int = 0
    unknown: int = 0
    malformed: int = 0

    @property
    def vector(self) -> int:
        return self.vector_path + self.vector_paint

    @property
    def total(self) -> int:
        return (
            self.text
            + self.image
            + self.vector_path
            + self.vector_paint
            + self.graphics_state
            + self.unknown
        )

    def add(self, other: "ContentOperatorCounts") -> "ContentOperatorCounts":
        return ContentOperatorCounts(
            text=self.text + other.text,
            image=self.image + other.image,
            vector_path=self.vector_path + other.vector_path,
            vector_paint=self.vector_paint + other.vector_paint,
            graphics_state=self.graphics_state + other.graphics_state,
            unknown=self.unknown + other.unknown,
            malformed=self.malformed + other.malformed,
        )


def content_stream_may_show_text(data: bytes | memoryview) -> bool:
    data_len = len(data)
    raw_bytes = full_source_bytes(data)
    if raw_bytes is None:
        raw_bytes = bytes(data)

    if not any(raw_bytes.find(candidate) >= 0 for candidate in TEXT_SHOWING_CANDIDATES):
        return False

    pos = 0
    container_depth = 0
    inline_image_lexer: PdfLexer | None = None
    while match := (
        CONTAINER_LEXICAL_MARKER_RE if container_depth else TEXT_OR_LEXICAL_MARKER_RE
    ).search(raw_bytes, pos):
        marker = match.start()
        token = match.group()
        if token == b"%":
            pos = skip_comment(raw_bytes, marker, data_len)
            continue
        if token == b"(":
            pos = skip_literal_string(raw_bytes, marker, data_len)
            continue
        if token == b"<":
            if marker + 1 < data_len and raw_bytes[marker + 1] == 60:
                container_depth += 1
                pos = marker + 2
            else:
                pos = skip_hex_string(raw_bytes, marker, data_len)
            continue
        if token == b">":
            if marker + 1 < data_len and raw_bytes[marker + 1] == 62:
                container_depth = max(0, container_depth - 1)
                pos = marker + 2
            else:
                pos = marker + 1
            continue
        if token == b"[":
            container_depth += 1
            pos = marker + 1
            continue
        if token == b"]":
            container_depth = max(0, container_depth - 1)
            pos = marker + 1
            continue
        if token == b"/":
            pos = skip_name(raw_bytes, marker, data_len)
            continue
        after = match.end()
        delimited = (marker == 0 or SEPARATOR_TABLE[raw_bytes[marker - 1]]) and (
            after == data_len or SEPARATOR_TABLE[raw_bytes[after]]
        )
        if not delimited:
            pos = marker + 1
            continue
        if token != b"BI" and container_depth == 0:
            return True
        if container_depth:
            pos = after
            continue
        if inline_image_lexer is None:
            inline_image_lexer = PdfLexer(raw_bytes)
        inline_image_lexer.pos = after
        try:
            parse_inline_image(inline_image_lexer)
        except PdfParseError:
            pos = after
        else:
            pos = inline_image_lexer.pos

    return False


def count_content_stream_operators(data: bytes | memoryview) -> ContentOperatorCounts:
    """Return coarse operator counts without constructing graphics or text state."""
    text = image = vector_path = vector_paint = graphics_state = unknown = 0

    def collector(operands: OperandWindow, depth: int, operator: str) -> None:
        nonlocal text, image, vector_path, vector_paint, graphics_state, unknown
        if operator in TEXT_OPERATORS:
            text += 1
        elif operator in IMAGE_OPERATORS:
            image += 1
        elif operator in VECTOR_PATH_OPERATORS:
            vector_path += 1
        elif operator in VECTOR_PAINT_OPERATORS:
            vector_paint += 1
        elif operator in GRAPHICS_STATE_OPERATORS:
            graphics_state += 1
        elif operator:
            unknown += 1

    try:
        handlers = CollectedIntegerHandlers(collector)
        dispatch_operations(
            PdfLexer(data),
            CollectedStringHandlers(collector),
            None,
            handlers,
            handlers,
            None,
            0,
        )
    except PdfParseError:
        return ContentOperatorCounts(malformed=1)
    return ContentOperatorCounts(
        text=text,
        image=image,
        vector_path=vector_path,
        vector_paint=vector_paint,
        graphics_state=graphics_state,
        unknown=unknown,
    )


def skip_text_clip_prefix(raw_bytes: bytes | memoryview, pos: int) -> int | None:
    match = TEXT_CLIP_PREFIX_RE.match(raw_bytes, pos)
    if match is None:
        return None
    return match.end()


class OperandWindow:
    __slots__ = ("operands", "count")

    operands: list[ContentOperand] | ContentOperands

    def __init__(
        self,
        operands: list[ContentOperand] | ContentOperands,
        count: int = 0,
    ) -> None:
        self.operands = operands
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __bool__(self) -> bool:
        return self.count > 0

    @overload
    def __getitem__(self, item: int) -> ContentOperand: ...

    @overload
    def __getitem__(self, item: slice) -> list[ContentOperand]: ...

    def __getitem__(self, item: int | slice) -> ContentOperand | list[ContentOperand]:
        if type(item) is int:
            count = self.count
            if 0 <= item < count:
                return self.operands[item]
            if item < 0:
                item += count
            if item < 0 or item >= count:
                raise IndexError(item)
            return self.operands[item]
        if isinstance(item, slice):
            start, stop, step = item.indices(self.count)
            return [self.operands[index] for index in range(start, stop, step)]
        raise TypeError("operand index must be int or slice")

    def __iter__(self) -> Iterator[ContentOperand]:
        for index in range(self.count):
            yield self.operands[index]


class NestedStreamRequest(Exception):
    """Internal control flow used to pause a content stream for a nested one."""


class OperationTarget(Protocol):
    capture_graphics: bool
    capture_glyphs: bool
    capture_clipping: bool
    current_decoder: FontDecoder | None

    def get_decoder(self, *, update_metrics: bool = True) -> FontDecoder: ...

    def append_text(
        self,
        operand: ContentOperand | None = None,
        *,
        data: bytes | memoryview | None = None,
        decoder: FontDecoder | None = None,
        string_syntax: str | None = None,
        compatibility_data: bytes | None = None,
    ) -> None: ...

    def append_tj_array(self, array: ContentOperand) -> None: ...

    def op_BT(self, operands: OperandWindow, depth: int) -> None: ...

    def op_TD(self, operands: OperandWindow, depth: int) -> None: ...

    def op_TD_values(self, tx: float, ty: float) -> None: ...

    def op_Tc(self, operands: OperandWindow, depth: int) -> None: ...

    def op_Tc_values(self, char_space: float) -> None: ...

    def op_Tf(self, operands: OperandWindow, depth: int) -> None: ...

    def op_Tf_values(
        self, font_operand: ContentOperand, font_size_operand: ContentOperand
    ) -> None: ...

    def op_Tm(self, operands: OperandWindow, depth: int) -> None: ...

    def op_Tm_values(self, a: float, b: float, c: float, d_: float, e: float, f: float) -> None: ...

    def op_Tw(self, operands: OperandWindow, depth: int) -> None: ...

    def op_Tw_values(self, word_space: float) -> None: ...

    def op_Td(self, operands: OperandWindow, depth: int) -> None: ...

    def op_Td_values(self, tx: float, ty: float) -> None: ...

    def op_re(self, operands: OperandWindow, depth: int) -> None: ...

    def op_re_values(
        self, x: int | float, y: int | float, width: int | float, height: int | float
    ) -> None: ...

    def op_RG_values(self, red: int | float, green: int | float, blue: int | float) -> None: ...

    def op_rg_values(self, red: int | float, green: int | float, blue: int | float) -> None: ...

    def op_w_value(self, line_width: int | float) -> None: ...

    def op_J_value(self, line_cap: int | float) -> None: ...

    def op_j_value(self, line_join: int | float) -> None: ...

    def op_M_value(self, miter_limit: int | float) -> None: ...

    def op_m_values(self, x: int | float, y: int | float) -> None: ...

    def op_l_values(self, x: int | float, y: int | float) -> None: ...

    def op_paint_stroke(self, operands: OperandWindow, depth: int) -> None: ...

    def op_paint_fill(self, operands: OperandWindow, depth: int) -> None: ...

    def op_ET(self, operands: OperandWindow, depth: int) -> None: ...

    def op_cm(self, operands: OperandWindow, depth: int) -> None: ...

    def op_cm_values(self, a: float, b: float, c: float, d_: float, e: float, f: float) -> None: ...

    def op_q(self, operands: OperandWindow, depth: int) -> None: ...

    def op_Q(self, operands: OperandWindow, depth: int) -> None: ...


BoundOperationHandler: TypeAlias = Callable[[OperandWindow, int], None]
StateOperationHandler: TypeAlias = Callable[[OperationTarget, OperandWindow, int], None]
OperationCollector: TypeAlias = Callable[[OperandWindow, int, str], None]

internal_HandlerT = TypeVar("internal_HandlerT", covariant=True)


class StringHandlerMap(Protocol[internal_HandlerT]):
    def get(self, key: str) -> internal_HandlerT | None: ...


class ByteHandlerMap(Protocol[internal_HandlerT]):
    def get(self, key: bytes) -> internal_HandlerT | None: ...


class SingleHandlerLookup(Protocol[internal_HandlerT]):
    def __getitem__(self, key: int) -> internal_HandlerT | None: ...


class IntHandlerMap(Protocol[internal_HandlerT]):
    def get(self, key: int) -> internal_HandlerT | None: ...


class CollectedOperationHandler:
    __slots__ = ("callback", "op_name")

    def __init__(self, callback: OperationCollector, op_name: str) -> None:
        self.callback = callback
        self.op_name = op_name

    def __call__(self, operands: OperandWindow, depth: int) -> None:
        self.callback(operands, depth, self.op_name)


class CollectedStringHandlers:
    __slots__ = ("callback", "handlers")

    def __init__(self, callback: OperationCollector) -> None:
        self.callback = callback
        self.handlers: dict[str, BoundOperationHandler] = {}

    def get(self, key: str) -> BoundOperationHandler:
        handler = self.handlers.get(key)
        if handler is None:
            handler = CollectedOperationHandler(self.callback, key)
            self.handlers[key] = handler
        return handler


class CollectedIntegerHandlers:
    __slots__ = ("callback", "handlers")

    def __init__(self, callback: OperationCollector) -> None:
        self.callback = callback
        self.handlers: dict[int, BoundOperationHandler] = {}

    def __getitem__(self, key: int) -> BoundOperationHandler:
        handler = self.handlers.get(key)
        if handler is not None:
            return handler
        if key > 255:
            op_name = chr(key >> 8) + chr(key & 0xFF)
        else:
            op_name = chr(key)
        handler = CollectedOperationHandler(self.callback, op_name)
        self.handlers[key] = handler
        return handler

    def get(self, key: int) -> BoundOperationHandler:
        return self[key]


@overload
def dispatch_operations(
    lexer: PdfLexer,
    op_handlers: StringHandlerMap[StateOperationHandler],
    op_handlers_bytes: ByteHandlerMap[StateOperationHandler] | None,
    single_op_handlers: SingleHandlerLookup[StateOperationHandler],
    double_op_handlers: IntHandlerMap[StateOperationHandler],
    handler_target: OperationTarget,
    depth: int,
    operands: list[ContentOperand] | None = None,
) -> None: ...


@overload
def dispatch_operations(
    lexer: PdfLexer,
    op_handlers: StringHandlerMap[BoundOperationHandler],
    op_handlers_bytes: ByteHandlerMap[BoundOperationHandler] | None,
    single_op_handlers: SingleHandlerLookup[BoundOperationHandler],
    double_op_handlers: IntHandlerMap[BoundOperationHandler],
    handler_target: None,
    depth: int,
    operands: list[ContentOperand] | None = None,
) -> None: ...


def dispatch_operations(
    lexer: PdfLexer,
    op_handlers: StringHandlerMap[StateOperationHandler] | StringHandlerMap[BoundOperationHandler],
    op_handlers_bytes: ByteHandlerMap[StateOperationHandler]
    | ByteHandlerMap[BoundOperationHandler]
    | None,
    single_op_handlers: SingleHandlerLookup[StateOperationHandler]
    | SingleHandlerLookup[BoundOperationHandler],
    double_op_handlers: IntHandlerMap[StateOperationHandler] | IntHandlerMap[BoundOperationHandler],
    handler_target: OperationTarget | None,
    depth: int,
    operands: list[ContentOperand] | None = None,
) -> None:
    if operands is None:
        operands = [None] * 16
    op_count = 0
    raw_data = lexer.raw_data
    data_len = lexer.data_len
    raw_bytes: bytes | memoryview
    source_bytes = full_source_bytes(raw_data)
    raw_bytes = source_bytes if source_bytes is not None else raw_data

    word_break_or_ws = WORD_BREAK_OR_WS
    ws_table = WS_TABLE
    is_word_start = IS_WORD_START
    op_get = op_handlers.get
    op_get_bytes = op_handlers_bytes.get if op_handlers_bytes is not None else None
    double_get = double_op_handlers.get
    max_operands = len(operands)
    exact_number_types = (int, float)

    def set_operand_count(count: int) -> None:
        operand_window.count = count if count < max_operands else max_operands

    operand_window = OperandWindow(operands)
    text_only = (
        handler_target is not None
        and not handler_target.capture_graphics
        and not handler_target.capture_glyphs
        and not handler_target.capture_clipping
    )
    should_decipher = lexer.decipher is not None and lexer.current_obj_num is not None
    skipped_clip_q_count = 0

    pos = lexer.pos
    while pos < data_len:
        byte = raw_bytes[pos]

        if ws_table[byte]:
            pos += 1
            while pos < data_len and ws_table[raw_bytes[pos]]:
                pos += 1
            if pos >= data_len:
                break
            byte = raw_bytes[pos]
            if byte == 37:
                pos = lexer.skip_ignored_at(pos)
                if pos >= data_len:
                    break
                byte = raw_bytes[pos]
        elif byte == 37:
            pos = lexer.skip_ignored_at(pos)
            if pos >= data_len:
                break
            byte = raw_bytes[pos]

        if is_word_start[byte]:
            limit = pos + 1024 if pos + 1024 < data_len else data_len
            end = pos + 1
            while end < limit:
                if word_break_or_ws[raw_bytes[end]]:
                    break
                end += 1
            if end == limit:
                lexer.pos = pos
                scanned = lexer.scan_word_at(pos, skip_ignored=False)
                if scanned is None:
                    break
                ignored, end = scanned

            n_raw = end - pos
            pos = end

            if n_raw > 0:
                start_offset = pos - n_raw
                first = raw_bytes[start_offset]
                # If the token has a numeric prefix, parse it directly.
                # This avoids slicing strings and allocating temporary objects.
                if first == 46 or 48 <= first <= 57 or first == 45 or first == 43:
                    # Fast path: Single digit number (e.g. '0', '5')
                    if n_raw == 1 and 48 <= first <= 57:
                        if op_count < max_operands:
                            operands[op_count] = first - 48
                        op_count += 1
                        continue

                    # Fast path: Two-character numeric tokens (e.g. '12', '-5', '.5')
                    if n_raw == 2:
                        b1 = raw_bytes[start_offset + 1]
                        if 48 <= first <= 57 and 48 <= b1 <= 57:
                            if op_count < max_operands:
                                operands[op_count] = (first - 48) * 10 + (b1 - 48)
                            op_count += 1
                            continue
                        if (first == 45 or first == 43) and 48 <= b1 <= 57:
                            val = b1 - 48
                            if op_count < max_operands:
                                operands[op_count] = -val if first == 45 else val
                            op_count += 1
                            continue
                        if first == 46 and 48 <= b1 <= 57:
                            if op_count < max_operands:
                                operands[op_count] = float(raw_bytes[start_offset:pos])
                            op_count += 1
                            continue

                    # Fast path: Three-character numeric tokens (e.g. '123', '-12')
                    elif n_raw == 3:
                        b1 = raw_bytes[start_offset + 1]
                        b2 = raw_bytes[start_offset + 2]
                        if 48 <= first <= 57 and 48 <= b1 <= 57 and 48 <= b2 <= 57:
                            if op_count < max_operands:
                                operands[op_count] = (first - 48) * 100 + (b1 - 48) * 10 + (b2 - 48)
                            op_count += 1
                            continue
                        if (first == 45 or first == 43) and 48 <= b1 <= 57 and 48 <= b2 <= 57:
                            val = (b1 - 48) * 10 + (b2 - 48)
                            if op_count < max_operands:
                                operands[op_count] = -val if first == 45 else val
                            op_count += 1
                            continue

                    # Fast path: Four-character numeric tokens (e.g. '1234', '-123')
                    elif n_raw == 4:
                        b1 = raw_bytes[start_offset + 1]
                        b2 = raw_bytes[start_offset + 2]
                        b3 = raw_bytes[start_offset + 3]
                        if (
                            48 <= first <= 57
                            and 48 <= b1 <= 57
                            and 48 <= b2 <= 57
                            and 48 <= b3 <= 57
                        ):
                            if op_count < max_operands:
                                operands[op_count] = (
                                    (first - 48) * 1000
                                    + (b1 - 48) * 100
                                    + (b2 - 48) * 10
                                    + (b3 - 48)
                                )
                            op_count += 1
                            continue
                        if (
                            (first == 45 or first == 43)
                            and 48 <= b1 <= 57
                            and 48 <= b2 <= 57
                            and 48 <= b3 <= 57
                        ):
                            val = (b1 - 48) * 100 + (b2 - 48) * 10 + (b3 - 48)
                            if op_count < max_operands:
                                operands[op_count] = -val if first == 45 else val
                            op_count += 1
                            continue

                        if 48 <= first <= 57 and b1 == 46 and 48 <= b2 <= 57 and 48 <= b3 <= 57:
                            if op_count < max_operands:
                                operands[op_count] = float(raw_bytes[start_offset:pos])
                            op_count += 1
                            continue

                    elif n_raw == 5:
                        b1 = raw_bytes[start_offset + 1]
                        b2 = raw_bytes[start_offset + 2]
                        b3 = raw_bytes[start_offset + 3]
                        b4 = raw_bytes[start_offset + 4]
                        if (
                            48 <= first <= 57
                            and b1 == 46
                            and 48 <= b2 <= 57
                            and 48 <= b3 <= 57
                            and 48 <= b4 <= 57
                        ):
                            if op_count < max_operands:
                                operands[op_count] = float(raw_bytes[start_offset:pos])
                            op_count += 1
                            continue
                        if (
                            48 <= first <= 57
                            and 48 <= b1 <= 57
                            and b2 == 46
                            and 48 <= b3 <= 57
                            and 48 <= b4 <= 57
                        ):
                            if op_count < max_operands:
                                operands[op_count] = float(raw_bytes[start_offset:pos])
                            op_count += 1
                            continue
                        if (
                            first == 45
                            and 48 <= b1 <= 57
                            and b2 == 46
                            and 48 <= b3 <= 57
                            and 48 <= b4 <= 57
                        ):
                            if op_count < max_operands:
                                operands[op_count] = float(raw_bytes[start_offset:pos])
                            op_count += 1
                            continue

                    elif n_raw == 6:
                        b1 = raw_bytes[start_offset + 1]
                        b2 = raw_bytes[start_offset + 2]
                        b3 = raw_bytes[start_offset + 3]
                        b4 = raw_bytes[start_offset + 4]
                        b5 = raw_bytes[start_offset + 5]
                        if (
                            first == 45
                            and 48 <= b1 <= 57
                            and b2 == 46
                            and 48 <= b3 <= 57
                            and 48 <= b4 <= 57
                            and 48 <= b5 <= 57
                        ):
                            if op_count < max_operands:
                                operands[op_count] = float(raw_bytes[start_offset:pos])
                            op_count += 1
                            continue
                        if (
                            48 <= first <= 57
                            and 48 <= b1 <= 57
                            and 48 <= b2 <= 57
                            and b3 == 46
                            and 48 <= b4 <= 57
                            and 48 <= b5 <= 57
                        ):
                            if op_count < max_operands:
                                operands[op_count] = float(raw_bytes[start_offset:pos])
                            op_count += 1
                            continue
                        if (
                            first == 45
                            and 48 <= b1 <= 57
                            and 48 <= b2 <= 57
                            and b3 == 46
                            and 48 <= b4 <= 57
                            and 48 <= b5 <= 57
                        ):
                            if op_count < max_operands:
                                operands[op_count] = float(raw_bytes[start_offset:pos])
                            op_count += 1
                            continue

                    elif n_raw == 7:
                        b1 = raw_bytes[start_offset + 1]
                        b2 = raw_bytes[start_offset + 2]
                        b3 = raw_bytes[start_offset + 3]
                        b4 = raw_bytes[start_offset + 4]
                        b5 = raw_bytes[start_offset + 5]
                        b6 = raw_bytes[start_offset + 6]
                        if (
                            first == 45
                            and 48 <= b1 <= 57
                            and 48 <= b2 <= 57
                            and 48 <= b3 <= 57
                            and b4 == 46
                            and 48 <= b5 <= 57
                            and 48 <= b6 <= 57
                        ):
                            if op_count < max_operands:
                                operands[op_count] = float(raw_bytes[start_offset:pos])
                            op_count += 1
                            continue

                    # General fallback: validate syntax and parse longer integers or float values
                    digit_start = 1 if first in (43, 45) else 0
                    saw_digit = False
                    saw_dot = False
                    is_valid = True
                    if digit_start == 0:
                        if first == 46:
                            saw_dot = True
                        elif 48 <= first <= 57:
                            saw_digit = True

                    for i in range(1 if digit_start == 0 else digit_start, n_raw):
                        b = raw_bytes[start_offset + i]
                        if 48 <= b <= 57:
                            saw_digit = True
                        elif b == 46 and not saw_dot:
                            saw_dot = True
                        else:
                            is_valid = False
                            break

                    if is_valid and saw_digit:
                        raw = raw_bytes[start_offset:pos]
                        raw_number = raw.tobytes() if type(raw) is memoryview else raw
                        if op_count < max_operands:
                            operands[op_count] = float(raw_number) if saw_dot else int(raw_number)
                        op_count += 1
                        continue

                if n_raw == 2 and raw_bytes[pos - 2] == 66 and raw_bytes[pos - 1] == 73:
                    lexer.pos = pos
                    try:
                        image = parse_inline_image(lexer)
                    except PdfParseError as exc:
                        message = str(exc)
                        if message in (
                            "unterminated inline image",
                            "unterminated inline image data",
                            "inline image keys must be names",
                            "expected inline image data separator",
                        ):
                            recovered_pos = recover_inline_image_position(
                                lexer,
                                pos,
                                (lambda token: op_get_bytes(token) is not None)
                                if op_get_bytes is not None
                                else None,
                            )
                            if recovered_pos is None:
                                if message == "unterminated inline image data":
                                    pos = data_len
                                    break
                                raise
                            pos = recovered_pos
                            op_count = 0
                            continue
                        raise
                    pos = lexer.pos
                    if op_count < max_operands:
                        operands[op_count] = image
                    op_count += 1
                    handler = op_get("BI")
                    if handler is not None:
                        operand_window.count = min(op_count, max_operands)
                        lexer.pos = pos
                        if handler_target is None:
                            cast(BoundOperationHandler, handler)(operand_window, depth)
                        else:
                            cast(StateOperationHandler, handler)(
                                handler_target,
                                operand_window,
                                depth,
                            )
                    op_count = 0
                    continue

                if handler_target is not None and n_raw == 2:
                    op0 = raw_bytes[pos - 2]
                    op1 = raw_bytes[pos - 1]
                    match op0:
                        case 66 if op1 == 84:  # 'B' 'T'
                            handler_target.op_BT(operand_window, depth)
                            op_count = 0
                            continue
                        case 84:  # 'T'
                            match op1:
                                case 68:  # 'D'
                                    if op_count >= 2:
                                        tx, ty = operands[0], operands[1]
                                        if (
                                            type(tx) in exact_number_types
                                            and type(ty) in exact_number_types
                                        ):
                                            handler_target.op_TD_values(
                                                cast(int | float, tx), cast(int | float, ty)
                                            )
                                        else:
                                            set_operand_count(op_count)
                                            handler_target.op_TD(operand_window, depth)
                                    else:
                                        set_operand_count(op_count)
                                        handler_target.op_TD(operand_window, depth)
                                    op_count = 0
                                    continue
                                case 99:  # 'c'
                                    if op_count and operands:
                                        char_space = operands[0]
                                        if type(char_space) in exact_number_types:
                                            handler_target.op_Tc_values(
                                                cast(int | float, char_space)
                                            )
                                        else:
                                            set_operand_count(op_count)
                                            handler_target.op_Tc(operand_window, depth)
                                    op_count = 0
                                    continue
                                case 102:  # 'f'
                                    if op_count >= 2:
                                        handler_target.op_Tf_values(operands[0], operands[1])
                                    else:
                                        set_operand_count(op_count)
                                        handler_target.op_Tf(operand_window, depth)
                                    op_count = 0
                                    continue
                                case 106:  # 'j'
                                    if op_count:
                                        decoder = (
                                            handler_target.current_decoder
                                            if handler_target.current_decoder is not None
                                            else handler_target.get_decoder()
                                        )
                                        # Tj consumes the top value from the
                                        # operand stack.  Earlier values are
                                        # stale operands in malformed streams.
                                        operand = operands[min(op_count, len(operands)) - 1]
                                        if type(operand) is PdfString:
                                            handler_target.append_text(
                                                data=operand.data,
                                                decoder=decoder,
                                                string_syntax=(
                                                    "literal" if operand.is_literal else "hex"
                                                ),
                                                compatibility_data=operand.compatibility_data,
                                            )
                                        elif type(operand) is bytes:
                                            handler_target.append_text(
                                                data=operand, decoder=decoder
                                            )
                                        else:
                                            handler_target.append_text(operand, decoder=decoder)
                                    op_count = 0
                                    continue
                                case 109:  # 'm'
                                    if op_count >= 6:
                                        tm_a, tm_b, tm_c = operands[0], operands[1], operands[2]
                                        tm_d, tm_e, tm_f = operands[3], operands[4], operands[5]
                                        if (
                                            type(tm_a) in exact_number_types
                                            and type(tm_b) in exact_number_types
                                            and type(tm_c) in exact_number_types
                                            and type(tm_d) in exact_number_types
                                            and type(tm_e) in exact_number_types
                                            and type(tm_f) in exact_number_types
                                        ):
                                            handler_target.op_Tm_values(
                                                cast(int | float, tm_a),
                                                cast(int | float, tm_b),
                                                cast(int | float, tm_c),
                                                cast(int | float, tm_d),
                                                cast(int | float, tm_e),
                                                cast(int | float, tm_f),
                                            )
                                        else:
                                            set_operand_count(op_count)
                                            handler_target.op_Tm(operand_window, depth)
                                    else:
                                        set_operand_count(op_count)
                                        handler_target.op_Tm(operand_window, depth)
                                    op_count = 0
                                    continue
                                case 119:  # 'w'
                                    if op_count:
                                        word_space = operands[0]
                                        if type(word_space) in exact_number_types:
                                            handler_target.op_Tw_values(
                                                cast(int | float, word_space)
                                            )
                                        else:
                                            set_operand_count(op_count)
                                            handler_target.op_Tw(operand_window, depth)
                                    op_count = 0
                                    continue
                                case 74:  # 'J'
                                    if op_count:
                                        handler_target.append_tj_array(operands[0])
                                    op_count = 0
                                    continue
                                case 100:  # 'd'
                                    if op_count >= 2:
                                        tx, ty = operands[0], operands[1]
                                        if (
                                            type(tx) in exact_number_types
                                            and type(ty) in exact_number_types
                                        ):
                                            handler_target.op_Td_values(
                                                cast(int | float, tx), cast(int | float, ty)
                                            )
                                        else:
                                            set_operand_count(op_count)
                                            handler_target.op_Td(operand_window, depth)
                                    else:
                                        set_operand_count(op_count)
                                        handler_target.op_Td(operand_window, depth)
                                    op_count = 0
                                    continue
                                # no case _: -- an unmatched op1 falls through to Stage D/E/F,
                                # same as the original if-chain having no trailing `else`.
                        case 82 if op1 == 71 and op_count >= 3:  # 'R' 'G'
                            red, green, blue = operands[0], operands[1], operands[2]
                            if (
                                type(red) in exact_number_types
                                and type(green) in exact_number_types
                                and type(blue) in exact_number_types
                            ):
                                handler_target.op_RG_values(
                                    cast(int | float, red),
                                    cast(int | float, green),
                                    cast(int | float, blue),
                                )
                                op_count = 0
                                continue
                        case 114:  # 'r'
                            match op1:
                                case 103 if op_count >= 3:  # 'g'
                                    red, green, blue = operands[0], operands[1], operands[2]
                                    if (
                                        type(red) in exact_number_types
                                        and type(green) in exact_number_types
                                        and type(blue) in exact_number_types
                                    ):
                                        handler_target.op_rg_values(
                                            cast(int | float, red),
                                            cast(int | float, green),
                                            cast(int | float, blue),
                                        )
                                        op_count = 0
                                        continue
                                case 101:  # 'e'
                                    if (
                                        handler_target.capture_graphics
                                        or handler_target.capture_glyphs
                                        or handler_target.capture_clipping
                                    ):
                                        if op_count >= 4:
                                            rect_x, rect_y = operands[0], operands[1]
                                            rect_width, rect_height = operands[2], operands[3]
                                            if (
                                                type(rect_x) in exact_number_types
                                                and type(rect_y) in exact_number_types
                                                and type(rect_width) in exact_number_types
                                                and type(rect_height) in exact_number_types
                                            ):
                                                handler_target.op_re_values(
                                                    cast(int | float, rect_x),
                                                    cast(int | float, rect_y),
                                                    cast(int | float, rect_width),
                                                    cast(int | float, rect_height),
                                                )
                                            else:
                                                set_operand_count(op_count)
                                                handler_target.op_re(operand_window, depth)
                                        else:
                                            set_operand_count(op_count)
                                            handler_target.op_re(operand_window, depth)
                                    op_count = 0
                                    continue
                                # no case _: here either -- e.g. `op1 == 103` with
                                # `op_count < 3` must fall through, not be swallowed.
                        case 69 if op1 == 84:  # 'E' 'T'
                            handler_target.op_ET(operand_window, depth)
                            op_count = 0
                            continue
                        case 99 if op1 == 109 and op_count >= 6:  # 'c' 'm'
                            m_a, m_b, m_c = operands[0], operands[1], operands[2]
                            m_d, m_e, m_f = operands[3], operands[4], operands[5]
                            if (
                                type(m_a) in exact_number_types
                                and type(m_b) in exact_number_types
                                and type(m_c) in exact_number_types
                                and type(m_d) in exact_number_types
                                and type(m_e) in exact_number_types
                                and type(m_f) in exact_number_types
                            ):
                                handler_target.op_cm_values(
                                    cast(int | float, m_a),
                                    cast(int | float, m_b),
                                    cast(int | float, m_c),
                                    cast(int | float, m_d),
                                    cast(int | float, m_e),
                                    cast(int | float, m_f),
                                )
                            else:
                                set_operand_count(op_count)
                                handler_target.op_cm(operand_window, depth)
                            op_count = 0
                            continue
                        # no case _: at the outer level either.

                if text_only:
                    if n_raw == 1:
                        op0 = raw_bytes[pos - 1]
                        if op0 == 113 and op_count == 0:
                            skipped_pos = skip_text_clip_prefix(raw_bytes, pos)
                            if skipped_pos is not None:
                                skipped_clip_q_count += 1
                                pos = skipped_pos
                                op_count = 0
                                continue
                        if op0 == 81 and skipped_clip_q_count:
                            skipped_clip_q_count -= 1
                            op_count = 0
                            continue
                        if TEXT_ONLY_SKIP_SINGLE[op0]:
                            op_count = 0
                            continue
                    elif n_raw == 2:
                        op_code = (raw_bytes[pos - 2] << 8) | raw_bytes[pos - 1]
                        if TEXT_ONLY_SKIP_DOUBLE[op_code]:
                            op_count = 0
                            continue

                if handler_target is not None and n_raw == 1:
                    op0 = raw_bytes[pos - 1]
                    match op0:
                        case 119 if op_count:  # 'w'
                            line_width = operands[0]
                            if type(line_width) in exact_number_types:
                                handler_target.op_w_value(cast(int | float, line_width))
                                op_count = 0
                                continue
                        case 74 if op_count:  # 'J'
                            line_cap = operands[0]
                            if type(line_cap) in exact_number_types:
                                handler_target.op_J_value(cast(int | float, line_cap))
                                op_count = 0
                                continue
                        case 106 if op_count:  # 'j'
                            line_join = operands[0]
                            if type(line_join) in exact_number_types:
                                handler_target.op_j_value(cast(int | float, line_join))
                                op_count = 0
                                continue
                        case 77 if op_count:  # 'M'
                            miter_limit = operands[0]
                            if type(miter_limit) in exact_number_types:
                                handler_target.op_M_value(cast(int | float, miter_limit))
                                op_count = 0
                                continue
                        case 109 if op_count >= 2:  # 'm'
                            move_x, move_y = operands[0], operands[1]
                            if (
                                type(move_x) in exact_number_types
                                and type(move_y) in exact_number_types
                            ):
                                handler_target.op_m_values(
                                    cast(int | float, move_x),
                                    cast(int | float, move_y),
                                )
                                op_count = 0
                                continue
                        case 108 if op_count >= 2:  # 'l'
                            line_x, line_y = operands[0], operands[1]
                            if (
                                type(line_x) in exact_number_types
                                and type(line_y) in exact_number_types
                            ):
                                handler_target.op_l_values(
                                    cast(int | float, line_x),
                                    cast(int | float, line_y),
                                )
                                op_count = 0
                                continue
                        case 83:  # 'S'
                            handler_target.op_paint_stroke(operand_window, depth)
                            op_count = 0
                            continue
                        case 102 | 70:  # 'f' | 'F'
                            handler_target.op_paint_fill(operand_window, depth)
                            op_count = 0
                            continue
                        case 113:  # 'q'
                            handler_target.op_q(operand_window, depth)
                            op_count = 0
                            continue
                        case 81:  # 'Q'
                            handler_target.op_Q(operand_window, depth)
                            op_count = 0
                            continue

                handler = None
                if n_raw == 1:
                    handler = single_op_handlers[raw_bytes[pos - 1]]
                elif n_raw == 2:
                    handler = double_get((raw_bytes[pos - 2] << 8) | raw_bytes[pos - 1])

                if handler is None:
                    raw = raw_bytes[pos - n_raw : pos]
                    raw_key = raw.tobytes() if type(raw) is memoryview else raw
                    if op_get_bytes is not None:
                        handler = op_get_bytes(raw_key)
                    if handler is None:
                        if raw_key in (
                            b"R",
                            b"obj",
                            b"endobj",
                            b"stream",
                            b"endstream",
                        ):
                            op_count = 0
                            continue
                        op_name = cast(str, lexer.parse_keyword(raw_key))
                        handler = op_get(op_name)

                if handler is not None:
                    operand_window.count = min(op_count, max_operands)
                    lexer.pos = pos
                    if handler_target is None:
                        cast(BoundOperationHandler, handler)(operand_window, depth)
                    else:
                        cast(StateOperationHandler, handler)(handler_target, operand_window, depth)
                op_count = 0
                continue

        lexer.pos = pos
        if byte == 91:
            if (
                handler_target is not None
                and not handler_target.capture_graphics
                and not handler_target.capture_glyphs
            ):
                if pos + 1 < data_len and raw_bytes[pos + 1] == 93:
                    if op_count < max_operands:
                        operands[op_count] = cast(ContentOperand, EMPTY_SIMPLE_TJ_ARRAY)
                    pos += 2
                    op_count += 1
                    continue
                simple_tj_array = lexer.parse_simple_tj_array()
                if simple_tj_array is not None:
                    if op_count < max_operands:
                        operands[op_count] = cast(ContentOperand, simple_tj_array)
                else:
                    operand_start = pos
                    try:
                        if op_count < max_operands:
                            operands[op_count] = cast(ContentOperand, lexer.parse_array())
                        else:
                            lexer.parse_array()
                    except PdfParseError as exc:
                        if str(exc) == "unterminated array" and lexer.pos >= data_len:
                            pos = data_len
                            break
                        if lexer.pos > operand_start:
                            pos = lexer.pos
                            continue
                        raise
            else:
                operand_start = pos
                try:
                    if op_count < max_operands:
                        operands[op_count] = cast(ContentOperand, lexer.parse_array())
                    else:
                        lexer.parse_array()
                except PdfParseError as exc:
                    if str(exc) == "unterminated array" and lexer.pos >= data_len:
                        pos = data_len
                        break
                    if lexer.pos > operand_start:
                        pos = lexer.pos
                        continue
                    raise
            pos = lexer.pos
            op_count += 1
            continue
        if byte == 60:
            if pos + 1 < data_len and raw_bytes[pos + 1] == 60:
                operand_start = pos
                legacy_pdfminer_mode = bool(
                    handler_target is not None
                    and getattr(
                        getattr(handler_target, "document", None),
                        "legacy_pdfminer_text_operators",
                        False,
                    )
                )
                try:
                    parse_dictionary = (
                        lexer.parse_dictionary
                        if legacy_pdfminer_mode
                        else lexer.parse_dictionary_or_stream
                    )
                    previous_recovery = lexer.recover_malformed_objects
                    if legacy_pdfminer_mode:
                        lexer.recover_malformed_objects = False
                    try:
                        if op_count < max_operands:
                            operands[op_count] = cast(ContentOperand, parse_dictionary())
                        else:
                            parse_dictionary()
                    finally:
                        lexer.recover_malformed_objects = previous_recovery
                except PdfParseError as exc:
                    if lexer.pos >= data_len or str(exc) == "unexpected end of PDF input":
                        trailing = raw_bytes[operand_start:]
                        if legacy_pdfminer_mode and (
                            b"endobj" in trailing or re.search(rb"(?m)^xref\b", trailing)
                        ):
                            raise ValueError("invalid pdfminer content dictionary") from exc
                        pos = data_len
                        break
                    if legacy_pdfminer_mode:
                        raise ValueError("invalid pdfminer content dictionary") from exc
                    if lexer.pos > operand_start:
                        pos = lexer.pos
                        continue
                    raise
            else:
                raw_string = lexer.read_hex_string()
                if should_decipher:
                    raw_string = lexer.apply_decipher(raw_string)
                value = PdfString(raw_string, is_literal=False)
                if op_count < max_operands:
                    operands[op_count] = value
            pos = lexer.pos
            op_count += 1
            continue
        if byte == 40:
            literal_start = lexer.pos
            raw_string = lexer.read_string()
            literal_end = lexer.pos
            lexer.pos = literal_start
            compatibility_string = lexer.read_string(drop_unknown_escapes=True)
            lexer.pos = literal_end
            if should_decipher:
                raw_string = lexer.apply_decipher(raw_string)
                compatibility_string = lexer.apply_decipher(compatibility_string)
            string_value = (
                raw_string
                if text_only
                else PdfString(
                    raw_string,
                    is_literal=True,
                    compatibility_data=compatibility_string,
                )
            )
            if op_count < max_operands:
                operands[op_count] = string_value
            pos = lexer.pos
            op_count += 1
            continue
        if byte == 47:
            if (
                pos + 3 <= data_len
                and raw_bytes[pos + 1] == 70
                and 48 <= raw_bytes[pos + 2] <= 57
                and (pos + 3 == data_len or SEPARATOR_TABLE[raw_bytes[pos + 3]])
            ):
                if op_count < max_operands:
                    operands[op_count] = FONT_DIGIT_NAMES[raw_bytes[pos + 2] - 48]
                pos += 3
            elif (
                pos + 4 <= data_len
                and raw_bytes[pos + 1] == 67
                and raw_bytes[pos + 2] == 83
                and 48 <= raw_bytes[pos + 3] <= 57
                and (pos + 4 == data_len or SEPARATOR_TABLE[raw_bytes[pos + 4]])
            ):
                if op_count < max_operands:
                    operands[op_count] = CS_DIGIT_NAMES[raw_bytes[pos + 3] - 48]
                pos += 4
            elif (
                pos + 4 <= data_len
                and raw_bytes[pos + 1] == 84
                and raw_bytes[pos + 2] == 84
                and 48 <= raw_bytes[pos + 3] <= 57
                and (pos + 4 == data_len or SEPARATOR_TABLE[raw_bytes[pos + 4]])
            ):
                if op_count < max_operands:
                    operands[op_count] = TT_DIGIT_NAMES[raw_bytes[pos + 3] - 48]
                pos += 4
            elif (
                pos + 2 <= data_len
                and raw_bytes[pos + 1] == 80
                and (pos + 2 == data_len or SEPARATOR_TABLE[raw_bytes[pos + 2]])
            ):
                if op_count < max_operands:
                    operands[op_count] = P_NAME
                pos += 2
            else:
                name_value = PdfName_of(lexer.read_name())
                if op_count < max_operands:
                    operands[op_count] = name_value
                pos = lexer.pos
            op_count += 1
            continue
        if byte == 62:
            pos = pos + 2 if pos + 1 < data_len and raw_bytes[pos + 1] == 62 else pos + 1
            continue
        if byte == 93:
            pos += 1
            continue

        pos += 1

    lexer.pos = pos


def iter_content_operations(lexer: PdfLexer) -> Iterator[ContentOperation]:

    results: list[ContentOperation] = []

    def collector(operands: OperandWindow, depth: int, op_name: str) -> None:
        results.append((op_name, tuple(operands)))

    dispatch_operations(
        lexer,
        CollectedStringHandlers(collector),
        None,
        CollectedIntegerHandlers(collector),
        CollectedIntegerHandlers(collector),
        None,
        0,
    )
    yield from results


def validate_inline_images(data: bytes | memoryview) -> None:
    """Validate inline-image boundaries without disabling normal stream recovery."""
    raw_bytes = full_source_bytes(data)
    if raw_bytes is None:
        raw_bytes = bytes(data)
    data_len = len(raw_bytes)
    pos = 0
    container_depth = 0
    lexer = PdfLexer(raw_bytes)
    while match := TEXT_OR_LEXICAL_MARKER_RE.search(raw_bytes, pos):
        marker = match.start()
        token = match.group()
        if token == b"%":
            pos = skip_comment(raw_bytes, marker, data_len)
            continue
        if token == b"(":
            pos = skip_literal_string(raw_bytes, marker, data_len)
            continue
        if token == b"<":
            if marker + 1 < data_len and raw_bytes[marker + 1] == 60:
                container_depth += 1
                pos = marker + 2
            else:
                pos = skip_hex_string(raw_bytes, marker, data_len)
            continue
        if token == b">":
            if marker + 1 < data_len and raw_bytes[marker + 1] == 62:
                container_depth = max(0, container_depth - 1)
                pos = marker + 2
            else:
                pos = marker + 1
            continue
        if token == b"[":
            container_depth += 1
            pos = marker + 1
            continue
        if token == b"]":
            container_depth = max(0, container_depth - 1)
            pos = marker + 1
            continue
        if token == b"/":
            pos = skip_name(raw_bytes, marker, data_len)
            continue
        after = match.end()
        delimited = (marker == 0 or SEPARATOR_TABLE[raw_bytes[marker - 1]]) and (
            after == data_len or SEPARATOR_TABLE[raw_bytes[after]]
        )
        if token != b"BI" or container_depth or not delimited:
            pos = after
            continue
        lexer.pos = after
        parse_inline_image(lexer)
        pos = lexer.pos


__all__ = (
    "ContentOperand",
    "ContentOperands",
    "ContentOperation",
    "ContentOperatorCounts",
    "OperandWindow",
    "content_stream_may_show_text",
    "count_content_stream_operators",
    "dispatch_operations",
    "iter_content_operations",
    "validate_inline_images",
)
