# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import cast

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfValueResolver


def resolve_resource_dict(
    value: object,
    resolver: PdfValueResolver,
) -> PdfDict | None:
    resolved = resolver.resolve(value)
    if resolved is None:
        return None
    if isinstance(resolved, dict):
        return cast(PdfDict, resolved)
    raise PdfParseError("resource value must be a dictionary")


__all__ = ("resolve_resource_dict",)
