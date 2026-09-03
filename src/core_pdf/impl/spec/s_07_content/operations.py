# SPDX-License-Identifier: AGPL-3.0-only
"""Tokenize and dispatch content-stream operators."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Protocol, TypeAlias, cast

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.primitives import PdfName, PdfString
from core_pdf.impl.spec.s_07_content.inline_images import (
    InlineImage,
    parse_inline_image,
    recover_inline_image_position,
)
from core_pdf.impl.spec.s_07_content.operator_tables import TEXT_ONLY_SKIP_OPERATORS
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
from core_pdf.impl.spec.s_07_syntax_primitives.tokens import SEPARATOR_TABLE

PdfName_of = PdfName.of

ContentOperand: TypeAlias = CachedPdfObject | InlineImage
ContentOperands: TypeAlias = tuple[ContentOperand, ...]
ContentOperation: TypeAlias = tuple[str, ContentOperands]


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


class NestedStreamRequest(Exception):
    """Internal control flow used to pause a content stream for a nested one."""


class OperationTarget(Protocol):
    capture_graphics: bool
    capture_glyphs: bool
    capture_clipping: bool


OperationHandler: TypeAlias = Callable[[ContentOperands, int], None]


def dispatch_operations(
    lexer: PdfLexer,
    get_handler: Callable[[str], OperationHandler | None],
    target: OperationTarget | None,
    depth: int,
) -> None:
    operands: list[ContentOperand] = []

    def append_operand(value: ContentOperand) -> None:
        if len(operands) < 16:
            operands.append(value)

    raw_data = lexer.raw_data
    data_len = lexer.data_len
    raw_bytes: bytes | memoryview
    source_bytes = full_source_bytes(raw_data)
    raw_bytes = source_bytes if source_bytes is not None else raw_data

    text_only = (
        target is not None
        and not target.capture_graphics
        and not target.capture_glyphs
        and not target.capture_clipping
    )
    should_decipher = lexer.decipher is not None and lexer.current_obj_num is not None
    # Fixed when the document is opened, but it was previously resolved by a
    # double getattr chain on every ``<<`` token of every content stream.
    legacy_pdfminer_mode = bool(
        target is not None
        and getattr(
            getattr(target, "document", None),
            "legacy_pdfminer_text_operators",
            False,
        )
    )
    skipped_clip_q_count = 0

    pos = lexer.pos
    while pos < data_len:
        pos = lexer.skip_ignored_at(pos)
        if pos >= data_len:
            break
        byte = raw_bytes[pos]

        if not SEPARATOR_TABLE[byte]:
            scanned = lexer.scan_word_at(pos, skip_ignored=False)
            if scanned is None:
                break
            raw_key, pos = scanned

            if raw_key:
                if is_number_word_bytes(raw_key):
                    append_operand(float(raw_key) if b"." in raw_key else int(raw_key))
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
                                (lambda token: get_handler(token.decode("latin-1")) is not None)
                                if target is not None
                                else None,
                            )
                            if recovered_pos is None:
                                if message == "unterminated inline image data":
                                    pos = data_len
                                    break
                                raise
                            pos = recovered_pos
                            operands.clear()
                            continue
                        raise
                    pos = lexer.pos
                    append_operand(image)
                    handler = get_handler("BI")
                    if handler is not None:
                        lexer.pos = pos
                        handler(tuple(operands), depth)
                    operands.clear()
                    continue

                # Skip irrelevant graphics operators before the normal handler lookup.
                if text_only:
                    if raw_key == b"q" and not operands:
                        skipped_pos = skip_text_clip_prefix(raw_bytes, pos)
                        if skipped_pos is not None:
                            skipped_clip_q_count += 1
                            pos = skipped_pos
                            operands.clear()
                            continue
                    if raw_key == b"Q" and skipped_clip_q_count:
                        skipped_clip_q_count -= 1
                        operands.clear()
                        continue
                    if raw_key in TEXT_ONLY_SKIP_OPERATORS:
                        operands.clear()
                        continue

                if raw_key in (
                    b"R",
                    b"obj",
                    b"endobj",
                    b"stream",
                    b"endstream",
                ):
                    operands.clear()
                    continue
                op_name = cast(str, lexer.parse_keyword(raw_key))
                handler = get_handler(op_name)

                if handler is not None:
                    lexer.pos = pos
                    handler(tuple(operands), depth)
                operands.clear()
                continue

        lexer.pos = pos
        if byte == 91:
            operand_start = pos
            try:
                append_operand(cast(ContentOperand, lexer.parse_array()))
            except PdfParseError as exc:
                if str(exc) == "unterminated array" and lexer.pos >= data_len:
                    pos = data_len
                    break
                if lexer.pos > operand_start:
                    pos = lexer.pos
                    continue
                raise
            pos = lexer.pos
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
                        append_operand(cast(ContentOperand, parse_dictionary()))
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
                append_operand(value)
            pos = lexer.pos
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
            append_operand(string_value)
            pos = lexer.pos
            continue
        if byte == 47:
            append_operand(PdfName_of(lexer.read_name()))
            pos = lexer.pos
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

    def get_handler(op_name: str) -> OperationHandler:
        def collect(operands: ContentOperands, _depth: int) -> None:
            results.append((op_name, operands))

        return collect

    dispatch_operations(
        lexer,
        get_handler,
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
    "content_stream_may_show_text",
    "dispatch_operations",
    "iter_content_operations",
    "validate_inline_images",
)
