# SPDX-License-Identifier: AGPL-3.0-only
"""Tokenize and dispatch content-stream operators."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Protocol, TypeAlias, TypeVar, cast, overload

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.primitives import PdfName, PdfString
from core_pdf.impl.spec.s_07_content.inline_images import (
    InlineImage,
    parse_inline_image,
    recover_inline_image_position,
)
from core_pdf.impl.spec.s_07_content.operator_tables import (
    TEXT_ONLY_SKIP_DOUBLE,
    TEXT_ONLY_SKIP_SINGLE,
)
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.types import CachedPdfObject
from core_pdf.impl.spec.s_07_syntax_primitives.scanning import (
    full_source_bytes,
    is_number_word_bytes,
    skip_comment,
    skip_hex_string,
    skip_literal_string,
    skip_name,
)
from core_pdf.impl.spec.s_07_syntax_primitives.tokens import SEPARATOR_TABLE, WS_TABLE

PdfName_of = PdfName.of
FONT_DIGIT_NAMES = tuple(PdfName_of(b"F" + bytes((48 + i,))) for i in range(10))
CS_DIGIT_NAMES = tuple(PdfName_of(b"CS" + bytes((48 + i,))) for i in range(10))
TT_DIGIT_NAMES = tuple(PdfName_of(b"TT" + bytes((48 + i,))) for i in range(10))
P_NAME = PdfName_of(b"P")

ContentOperand: TypeAlias = CachedPdfObject | InlineImage
ContentOperands: TypeAlias = tuple[ContentOperand, ...]
ContentOperation: TypeAlias = tuple[str, ContentOperands]


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


def _advance_past_lexical_markers(
    raw_bytes: bytes,
    pos: int,
    data_len: int,
    container_depth: int,
    container_re: re.Pattern[bytes],
) -> tuple[int, int, bytes, int, bool] | None:
    while match := (container_re if container_depth else TEXT_OR_LEXICAL_MARKER_RE).search(
        raw_bytes, pos
    ):
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
        delimited = bool(
            (marker == 0 or SEPARATOR_TABLE[raw_bytes[marker - 1]])
            and (after == data_len or SEPARATOR_TABLE[raw_bytes[after]])
        )
        return marker, after, token, container_depth, delimited
    return None


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
    while scan := _advance_past_lexical_markers(
        raw_bytes, pos, data_len, container_depth, CONTAINER_LEXICAL_MARKER_RE
    ):
        marker, after, token, container_depth, delimited = scan
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


BoundOperationHandler: TypeAlias = Callable[[OperandWindow, int], None]
StateOperationHandler: TypeAlias = Callable[[OperationTarget, OperandWindow, int], None]
OperationCollector: TypeAlias = Callable[[OperandWindow, int, str], None]

internal_HandlerT = TypeVar("internal_HandlerT", covariant=True)


class StringHandlerMap(Protocol[internal_HandlerT]):
    def get(self, key: str) -> internal_HandlerT | None: ...


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


@overload
def dispatch_operations(
    lexer: PdfLexer,
    op_handlers: StringHandlerMap[StateOperationHandler],
    handler_target: OperationTarget,
    depth: int,
    operands: list[ContentOperand] | None = None,
) -> None: ...


@overload
def dispatch_operations(
    lexer: PdfLexer,
    op_handlers: StringHandlerMap[BoundOperationHandler],
    handler_target: None,
    depth: int,
    operands: list[ContentOperand] | None = None,
) -> None: ...


def dispatch_operations(
    lexer: PdfLexer,
    op_handlers: StringHandlerMap[StateOperationHandler] | StringHandlerMap[BoundOperationHandler],
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

    word_break_or_ws = SEPARATOR_TABLE
    ws_table = WS_TABLE
    is_word_start = IS_WORD_START
    op_get = op_handlers.get
    max_operands = len(operands)

    operand_window = OperandWindow(operands)
    text_only = (
        handler_target is not None
        and not handler_target.capture_graphics
        and not handler_target.capture_glyphs
        and not handler_target.capture_clipping
    )
    should_decipher = lexer.decipher is not None and lexer.current_obj_num is not None
    # Fixed when the document is opened, but it was previously resolved by a
    # double getattr chain on every ``<<`` token of every content stream.
    legacy_pdfminer_mode = bool(
        handler_target is not None
        and getattr(
            getattr(handler_target, "document", None),
            "legacy_pdfminer_text_operators",
            False,
        )
    )
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
                raw = raw_bytes[pos - n_raw : pos]
                raw_key = raw.tobytes() if type(raw) is memoryview else raw
                if is_number_word_bytes(raw_key):
                    if op_count < max_operands:
                        operands[op_count] = float(raw_key) if b"." in raw_key else int(raw_key)
                    op_count += 1
                    continue

                if raw_key == b"BI":
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
                                (lambda token: op_get(token.decode("latin-1")) is not None)
                                if handler_target is not None
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

                # Skip irrelevant graphics operators before the normal handler lookup.
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
                        operands[op_count] = cast(ContentOperand, ())
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
    while scan := _advance_past_lexical_markers(
        raw_bytes, pos, data_len, container_depth, TEXT_OR_LEXICAL_MARKER_RE
    ):
        _marker, after, token, container_depth, delimited = scan
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
    "OperandWindow",
    "content_stream_may_show_text",
    "dispatch_operations",
    "iter_content_operations",
    "validate_inline_images",
)
