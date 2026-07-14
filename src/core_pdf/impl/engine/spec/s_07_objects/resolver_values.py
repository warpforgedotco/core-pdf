# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import cast

from core_pdf.impl.engine.spec.s_07_objects.coercion import (
    normalize_pdf_name,
    parse_float,
    parse_float_strict,
    parse_int,
)
from core_pdf.impl.engine.spec.s_07_objects.object_cache import (
    CachedPdfObject,
    DeepObjectCache,
)
from core_pdf.impl.engine.spec.s_09_fonts.encoding import decode_pdf_text_string
from core_pdf.impl.objects import MISSING, PdfName, PdfReference, PdfStream, PdfString
from core_pdf.impl.types import PdfDict, PdfObject


TERMINAL_TYPES = {int, float, str, bool, type(None), PdfName, bytes}


class ResolverValueMixin:
    __slots__ = ()

    deep_cache: DeepObjectCache

    def resolve(self, ref: object) -> object:
        raise NotImplementedError

    def deep_resolve(self, value: object, seen: set[int] | None = None) -> object:
        t = type(value)
        terminal_types = TERMINAL_TYPES
        if t in terminal_types:
            return value

        if t is PdfReference:
            res = self.resolve(value)
            if type(res) in (dict, list, PdfStream, tuple, PdfReference):
                return self.deep_resolve(res, seen)
            return res

        if t not in (dict, list, tuple, PdfStream):
            return value

        val_id = id(value)
        deep_cache = self.deep_cache
        cached = deep_cache.get(val_id, MISSING)
        if cached is not MISSING:
            return cached

        if seen is None:
            seen = set()

        if t is PdfStream:
            stream = cast(PdfStream, value)
            resolved_dict = self.deep_resolve(stream.dictionary, seen)
            if resolved_dict is stream.dictionary:
                deep_cache[val_id] = stream
                return stream
            res = stream.replace(dictionary=resolved_dict)
            deep_cache[val_id] = res
            return res

        marker = id(value)
        if marker in seen:
            return value
        seen.add(marker)

        if t is list:
            items = cast(list[object], value)
            if len(items) > 64 and set(map(type, items)).issubset(terminal_types):
                deep_cache[val_id] = cast(CachedPdfObject, items)
                return items
            for item in items:
                if type(item) not in terminal_types:
                    break
            else:
                deep_cache[val_id] = cast(CachedPdfObject, items)
                return items
            res_list: list[object] = []
            changed = False
            append = res_list.append
            for item in items:
                if type(item) in terminal_types:
                    append(item)
                    continue
                resolved_item = self.deep_resolve(item, seen)
                append(resolved_item)
                if resolved_item is not item:
                    changed = True
            res = res_list if changed else items
            deep_cache[val_id] = cast(CachedPdfObject, res)
            return res

        if t is dict:
            mapping = cast(PdfDict, value)
            for item in mapping.values():
                if type(item) not in terminal_types:
                    break
            else:
                deep_cache[val_id] = cast(CachedPdfObject, mapping)
                return mapping
            res_dict: PdfDict = {}
            changed = False
            for key, item in mapping.items():
                resolved_item = self.deep_resolve(item, seen)
                res_dict[key] = cast(PdfObject, resolved_item)
                if resolved_item is not item:
                    changed = True
            res = res_dict if changed else mapping
            deep_cache[val_id] = cast(CachedPdfObject, res)
            return res

        if t is tuple:
            tuple_items = cast(tuple[object, ...], value)
            res = [self.deep_resolve(item, seen) for item in tuple_items]
            deep_cache[val_id] = cast(CachedPdfObject, res)
            return res

        return value

    def resolve_dict(self, value: object) -> PdfDict | None:
        resolved = self.deep_resolve(value)
        return resolved if isinstance(resolved, dict) else None

    def resolve_box(self, value: object) -> tuple[float, float, float, float] | None:
        resolved = self.deep_resolve(value)
        if resolved is None:
            return None
        if isinstance(resolved, (list, tuple)) and len(resolved) == 4:
            try:
                return (
                    parse_float_strict(resolved[0]),
                    parse_float_strict(resolved[1]),
                    parse_float_strict(resolved[2]),
                    parse_float_strict(resolved[3]),
                )
            except ValueError:
                raise ValueError("invalid box value")
        raise ValueError("invalid box value")

    def resolve_font_dict(self, font: PdfDict) -> PdfDict:
        resolved_font = self.deep_resolve(font)
        if not isinstance(resolved_font, dict):
            raise ValueError("invalid font dictionary")
        return resolved_font

    def resolve_list(self, value: object) -> list[object] | None:
        resolved = self.deep_resolve(value)
        return resolved if isinstance(resolved, list) else None

    def resolve_float(self, value: object, default: float | None = 0.0) -> float | None:
        if type(value) is int:
            return float(value)
        if type(value) is float:
            return value
        if type(value) is bool:
            return default
        return parse_float(self.resolve(value), default=default)

    def resolve_name(self, value: object) -> str | None:
        return normalize_pdf_name(value) or normalize_pdf_name(self.resolve(value))

    def resolve_name_like_value(self, resolved: object) -> str | None:
        val = self.resolve(resolved)
        name = normalize_pdf_name(val)
        if name is not None:
            return name
        if type(val) is PdfString:
            return decode_pdf_text_string(val.data)
        return None

    def resolve_int(self, value: object, default: int | None = None) -> int | None:
        if type(value) is int:
            return value
        return parse_int(self.resolve(value), default)

    def resolve_str(self, value: object) -> str | None:
        if type(value) is str:
            return value

        resolved = self.deep_resolve(value)
        if type(resolved) is PdfString:
            return decode_pdf_text_string(resolved.data)
        if type(resolved) is bytes:
            return decode_pdf_text_string(resolved)
        if type(resolved) is str:
            return resolved
        return None
