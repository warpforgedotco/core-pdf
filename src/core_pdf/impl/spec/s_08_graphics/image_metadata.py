# SPDX-License-Identifier: AGPL-3.0-only
"""PDF image metadata shared by display-list and raster consumers."""

from __future__ import annotations

from typing import Any, TypeGuard

from core_pdf.impl.model.geometry import rect_tuple
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
    parse_float,
    parse_int,
)
from core_pdf.impl.spec.s_07_syntax_primitives.pdfdict import lookup_dict_key


def pdf_int(value: Any, default: int) -> int:
    if type(value) is bool:
        return default
    parsed = parse_int(value, None)
    return default if parsed is None else parsed


def pdf_float(value: Any, default: float) -> float:
    if type(value) is bool:
        return default
    parsed = parse_float(value, None)
    return default if parsed is None else parsed


def pdf_positive_int(value: Any, default: int = 0) -> int:
    parsed = pdf_int(value, default)
    return parsed if parsed > 0 else default


def pdf_number(value: Any) -> TypeGuard[int | float]:
    return type(value) is int or type(value) is float


def image_filter_names(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [name for item in value if (name := normalize_pdf_name(item))]
    name = normalize_pdf_name(value)
    return [name] if name else []


def image_color_space_name(value: Any) -> str | None:
    prefixes: list[str] = []
    seen: set[int] = set()
    current = value
    while True:
        name = normalize_pdf_name(current)
        if name is not None:
            return ":".join((*prefixes, name))
        if not isinstance(current, (list, tuple)) or not current:
            return ":".join(prefixes) if prefixes else None
        marker = id(current)
        if marker in seen:
            return ":".join(prefixes) if prefixes else None
        seen.add(marker)
        kind = normalize_pdf_name(current[0])
        if kind == "Indexed":
            prefixes.append("Indexed")
            if len(current) <= 1:
                return ":".join(prefixes)
            current = current[1]
            continue
        if kind == "ICCBased":
            prefixes.append("ICCBased")
            if len(current) <= 1 or not isinstance(current[1], dict):
                return ":".join(prefixes)
            alternate = lookup_dict_key(current[1], "Alternate")
            if alternate is None:
                return ":".join(prefixes)
            current = alternate
            continue
        if kind is None:
            return ":".join(prefixes) if prefixes else None
        return ":".join((*prefixes, kind))


def image_display_metadata(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    dictionary = data.get("dictionary")
    if not isinstance(dictionary, dict):
        return {}

    width = pdf_positive_int(lookup_dict_key(dictionary, "Width"))
    height = pdf_positive_int(lookup_dict_key(dictionary, "Height"))
    image_mask = lookup_dict_key(dictionary, "ImageMask") is True
    bpc = pdf_positive_int(
        lookup_dict_key(dictionary, "BitsPerComponent"),
        1 if image_mask else 0,
    )
    bbox = data.get("bbox")
    metadata: dict[str, Any] = {
        "kind": kind,
        "width": width,
        "height": height,
        "pixels": width * height if width > 0 and height > 0 else 0,
        "bits_per_component": bpc if bpc > 0 else None,
        "color_space": image_color_space_name(lookup_dict_key(dictionary, "ColorSpace")),
        "filters": image_filter_names(lookup_dict_key(dictionary, "Filter")),
        "image_mask": image_mask,
        "has_mask": lookup_dict_key(dictionary, "Mask") is not None,
        "has_soft_mask": lookup_dict_key(dictionary, "SMask") is not None,
    }

    raw_data = data.get("raw_data", data.get("data"))
    if isinstance(raw_data, (bytes, bytearray, memoryview)):
        metadata["raw_bytes"] = len(raw_data)

    bbox_tuple = rect_tuple(bbox)
    if bbox_tuple is not None:
        x0, y0, x1, y1 = bbox_tuple
        display_width = abs(x1 - x0)
        display_height = abs(y1 - y0)
        metadata["display_width"] = display_width
        metadata["display_height"] = display_height
        metadata["display_area"] = display_width * display_height

    return metadata


__all__ = (
    "image_color_space_name",
    "image_display_metadata",
    "image_filter_names",
    "pdf_float",
    "pdf_int",
    "pdf_number",
    "pdf_positive_int",
)
