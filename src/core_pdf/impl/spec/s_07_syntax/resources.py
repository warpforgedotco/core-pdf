# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve resource dictionaries without traversing their unused entries."""

from __future__ import annotations

from typing import cast

from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfValueResolver


def resolve_resource_dict(value: object, resolver: PdfValueResolver) -> PdfDict | None:
    """Keep entry references intact so a lookup retains the source's identity."""
    resolved = resolver.resolve(value)
    return cast(PdfDict, resolved) if isinstance(resolved, dict) else None


__all__ = ("resolve_resource_dict",)
