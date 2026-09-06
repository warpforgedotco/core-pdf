# SPDX-License-Identifier: AGPL-3.0-only
"""Conversions from PDF values into host and compatibility representations."""

from __future__ import annotations

from collections.abc import Callable

from core_pdf.impl.types import PdfString


def is_pdf_null(value: object) -> bool:
    return value is None or type(value).__name__ == "NullObject"


def coerce_to_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, PdfString):
        return value.data
    if isinstance(value, str):
        return value.encode("latin-1")
    raise TypeError(f"cannot coerce {type(value).__name__} to bytes")


def coerce_value(value: object, string_decoder: Callable[[bytes], object] | None = None) -> object:
    def decode_scalar(item: object) -> object:
        if string_decoder is not None:
            if isinstance(item, PdfString):
                return string_decoder(item.data)
            if isinstance(item, bytes):
                return string_decoder(item)
        return item

    def walk(item: object) -> object:
        decoded_item = decode_scalar(item)
        if decoded_item is not item:
            return decoded_item

        if isinstance(item, dict):
            changed = False
            coerced_items: list[tuple[object, object]] = []
            for child_key, child_value in item.items():
                coerced_child = walk(child_value)
                coerced_items.append((child_key, coerced_child))
                if coerced_child is not child_value:
                    changed = True
            if not changed:
                return item
            return dict(coerced_items)

        if isinstance(item, (list, tuple)):
            changed = False
            coerced_seq_items: list[object] = []
            for child_value in item:
                coerced_child = walk(child_value)
                coerced_seq_items.append(coerced_child)
                if coerced_child is not child_value:
                    changed = True
            if not changed:
                return item
            return list(coerced_seq_items)

        return decoded_item

    return walk(value)
