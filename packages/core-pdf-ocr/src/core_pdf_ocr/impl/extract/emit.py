# SPDX-License-Identifier: AGPL-3.0-only
"""Normalize recognition artifacts before shared page composition."""

from __future__ import annotations

import re

from core_pdf.impl.extract import emit as native_emit
from core_pdf.impl.extract.contracts import ParsedBlock
from core_pdf.impl.extract.emit import internal_wordlike_pipe_token
from core_pdf.impl.output.model import Block, Figure, Page, Table
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing
from core_pdf_ocr.impl.extract.table_reconcile import internal_project_text_and_tables

internal_NUMERIC_PIPE_TOKEN = re.compile(r"^[($+-]?\d+(?:[.,]\d+)?[%)]?$")


def internal_numeric_pipe_token(token: str) -> bool:
    return bool(internal_NUMERIC_PIPE_TOKEN.match(token.strip()))


def internal_corrupt_ocr_block(block: Block) -> bool:
    if block.provenance != ("ocr",):
        return False
    text = block.text.strip()
    if not text:
        return True
    nonspace = [character for character in text if not character.isspace()]
    return len(nonspace) <= 2 and not any(character.isalnum() for character in nonspace)


def internal_ocr_artifact_token(token: str, line_tokens: list[str]) -> bool:
    if token in {"'", "[", "!"}:
        return True
    if (
        len(line_tokens) <= 2
        and len(token) == 2
        and token.startswith("0")
        and token[1].isdigit()
        and not any(internal_wordlike_pipe_token(line_token) for line_token in line_tokens)
    ):
        return True
    return token == "•" and not any(
        internal_wordlike_pipe_token(line_token) for line_token in line_tokens
    )


def internal_remove_sparse_ocr_artifacts(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        tokens = line.split()
        original_tokens = tokens
        if len(tokens) == 2 and tokens[0] == ">" and re.fullmatch(r"\d+(?:[.,]\d+)?", tokens[1]):
            lines.append(tokens[1])
            continue
        artifact_flags = [internal_ocr_artifact_token(token, tokens) for token in tokens]
        if any(artifact_flags):
            tokens = [
                token
                for token, is_artifact in zip(tokens, artifact_flags, strict=True)
                if not is_artifact
            ]
        if "|" not in tokens:
            lines.append(" ".join(tokens) if tokens != original_tokens else line)
            continue
        non_pipe_tokens = [token for token in tokens if token != "|"]
        if not non_pipe_tokens:
            continue
        if (
            all(internal_numeric_pipe_token(token) for token in non_pipe_tokens)
            or sum(internal_wordlike_pipe_token(token) for token in non_pipe_tokens) <= 1
        ):
            lines.append(" ".join(non_pipe_tokens))
            continue
        lines.append(line)
    return "\n".join(lines)


def internal_normalize_emitted_text(text: str, source: str) -> str:
    normalized = native_emit.internal_normalize_emitted_text(text, source)
    if source == "ocr":
        normalized = internal_remove_sparse_ocr_artifacts(normalized)
    return normalized


def internal_normalized_blocks(
    parsed_blocks: tuple[ParsedBlock, ...],
    drawings: tuple[CapturedDrawing, ...],
) -> list[Block]:
    return native_emit.internal_normalized_blocks(
        parsed_blocks, drawings, normalize_text=internal_normalize_emitted_text
    )


def assemble_page(
    blocks: tuple[ParsedBlock, ...],
    *,
    page_number: int,
    width: float,
    height: float,
    rotation: int,
    route: str,
    tables: tuple[Table, ...] = (),
    figures: tuple[Figure, ...] = (),
    diagnostics: tuple[str, ...] = (),
    full_page_image: bool = False,
    drawings: tuple[CapturedDrawing, ...] = (),
) -> Page:
    normalized_blocks = internal_normalized_blocks(blocks, drawings)
    normalized_blocks = native_emit.internal_remove_off_page_blocks(
        [
            block
            for block in native_emit.internal_remove_corrupt_native_blocks(normalized_blocks)
            if not internal_corrupt_ocr_block(block)
        ],
        width,
        height,
    )
    normalized_blocks, projected_tables = internal_project_text_and_tables(
        normalized_blocks, tables
    )
    return native_emit.internal_compose_page(
        blocks,
        normalized_blocks,
        projected_tables,
        page_number=page_number,
        width=width,
        height=height,
        rotation=rotation,
        route=route,
        figures=figures,
        diagnostics=diagnostics,
        full_page_image=full_page_image,
    )
