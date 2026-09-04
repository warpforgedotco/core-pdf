# SPDX-License-Identifier: AGPL-3.0-only
"""Operator metadata and dispatch tables for PDF content streams."""

from __future__ import annotations

from typing import Any

from core_pdf.impl.spec.s_07_syntax_primitives.content_operators import (
    CONTENT_OPERATOR_HANDLERS,
)

#: Content-stream operator name -> the ``TextState`` method that implements it.
#: The table itself lives at the spec floor because the filter layer recognizes
#: content streams from the same vocabulary and may not import this package.
#: Add new operators there; this stays the module content code binds through.
OPERATOR_SPECS = CONTENT_OPERATOR_HANDLERS


__all__ = (
    "OPERATOR_SPECS",
    "build_operator_handlers",
)


def build_operator_handlers(target: Any) -> dict[str, Any]:
    """Bind each content operator name to its target method."""
    return {name: getattr(target, handler) for name, handler in OPERATOR_SPECS.items()}
