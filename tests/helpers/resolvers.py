# SPDX-License-Identifier: AGPL-3.0-only
"""An in-memory ``PdfValueResolver`` for spec-level tests that hold direct objects."""

from __future__ import annotations

from typing import cast

from core_pdf.impl.primitives import PdfName, PdfString
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
    parse_float,
    parse_int,
)
from core_pdf.impl.spec.s_07_syntax_primitives.text_string import decode_pdf_text_string


class IdentityResolver:
    """Resolves nothing: every value is already direct.

    Implements the whole ``PdfValueResolver`` protocol so a test that hands a
    dictionary of direct objects to a spec helper keeps working when that helper
    starts calling a resolver method it did not use before.
    """

    def __init__(self) -> None:
        # TextState reads this straight off the resolver: TextResolver declares
        # it, so a double standing in for one has to carry it.
        self.kw_cache: dict[bytes, object] = {}

    def resolve(self, ref: object) -> object:
        return ref

    def deep_resolve(self, value: object, seen: set[int] | None = None) -> object:
        return value

    def resolve_dict(self, value: object) -> PdfDict | None:
        return cast(PdfDict, value) if isinstance(value, dict) else None

    def resolve_box(self, value: object) -> tuple[float, float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        parts = [parse_float(item, None) for item in value]
        if any(part is None for part in parts):
            return None
        return cast(tuple[float, float, float, float], tuple(parts))

    def resolve_font_dict(self, font: PdfDict) -> PdfDict:
        return font

    def resolve_float(self, value: object, default: float | None = 0.0) -> float | None:
        return parse_float(value, default)

    def resolve_name(self, value: object) -> str | None:
        return normalize_pdf_name(value)

    def resolve_name_like_value(self, resolved: object) -> str | None:
        name = normalize_pdf_name(resolved)
        if name is not None:
            return name
        if isinstance(resolved, PdfString):
            return decode_pdf_text_string(resolved.data)
        return None

    def resolve_name_or_text(self, value: object, *, name_like: bool = False) -> str | None:
        text = self.resolve_name(value)
        if text is None and name_like:
            text = self.resolve_name_like_value(value)
        return text or self.resolve_str(value)

    def resolve_int(self, value: object, default: int | None = None) -> int | None:
        return parse_int(value, default)

    def resolve_str(self, value: object) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, PdfName):
            return value.value
        return None


__all__ = ("IdentityResolver",)
