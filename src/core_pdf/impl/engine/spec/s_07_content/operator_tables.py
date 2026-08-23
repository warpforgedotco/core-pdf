# SPDX-License-Identifier: AGPL-3.0-only
"""Precomputed dispatch tables for PDF content-stream operators."""

from __future__ import annotations

from typing import Any

from core_pdf.impl.engine.spec.s_07_syntax.content_operators import (
    TEXT_ONLY_NOOP_OPS,
    TEXT_ONLY_OP,
    TEXT_OP,
)

__all__ = (
    "TEXT_ONLY_NOOP_OPS",
    "TEXT_ONLY_OP",
    "TEXT_OP",
    "build_operator_tables",
)


def build_operator_tables(
    target: type[Any],
    *,
    capture_graphics: bool,
    capture_clipping: bool,
) -> tuple[
    dict[str, Any],
    dict[bytes, Any],
    list[Any | None],
    dict[int, Any],
]:
    operator_map = TEXT_OP if capture_graphics or capture_clipping else TEXT_ONLY_OP
    handlers = {name: getattr(target, method) for name, method in operator_map.items()}
    byte_handlers = {name.encode("latin-1"): handler for name, handler in handlers.items()}
    single_handlers: list[Any | None] = [None] * 256
    double_handlers: dict[int, Any] = {}
    for name, handler in handlers.items():
        if len(name) == 1:
            single_handlers[ord(name)] = handler
        elif len(name) == 2:
            double_handlers[(ord(name[0]) << 8) | ord(name[1])] = handler
    return handlers, byte_handlers, single_handlers, double_handlers
