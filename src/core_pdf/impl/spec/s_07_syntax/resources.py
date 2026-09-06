# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve resource dictionaries without traversing their unused entries."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfValueResolver


def resolve_resource_dict(
    value: object,
    resolver: PdfValueResolver,
    *,
    on_invalid: Callable[[object], PdfDict | None] | None = None,
) -> PdfDict | None:
    """Keep entry references intact so a lookup retains the source's identity."""
    resolved = resolver.resolve(value)
    if resolved is None:
        return None
    if isinstance(resolved, dict):
        return cast(PdfDict, resolved)
    if on_invalid is not None:
        return on_invalid(resolved)
    raise PdfParseError("resource value must be a dictionary")


__all__ = ("resolve_resource_dict",)
