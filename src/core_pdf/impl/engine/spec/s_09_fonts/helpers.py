"""Pure font decoding helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from core_pdf.impl.engine.spec.s_07_syntax.primitives import parse_name
from core_pdf.impl.engine.spec.s_09_fonts.encoding import PDFDOC_ENCODING_TABLE
from core_pdf.impl.engine.spec.s_09_fonts.glyphs import glyph_name_to_unicode

EncodingFallback = Callable[[int], str]


def fallback_with_pdfdoc(b: int) -> str:
    return PDFDOC_ENCODING_TABLE[b]


def decode_with_table(data: bytes, table: tuple[str, ...]) -> str:
    return "".join(map(table.__getitem__, data))


def decode_chunks_with_table(chunks: list[bytes], table: tuple[str, ...]) -> list[str]:
    out = []
    for chunk in chunks:
        if len(chunk) == 1:
            out.append(table[chunk[0]])
        else:
            out.append(decode_with_table(chunk, table))
    return out


def build_decode_table(
    fallback_fn: EncodingFallback,
    differences: dict[int, str] | None = None,
) -> tuple[str, ...]:
    gtn = glyph_name_to_unicode
    if not differences:
        return tuple(fallback_fn(b) for b in range(256))
    table = [fallback_fn(b) for b in range(256)]
    for code, glyph_name in differences.items():
        table[code] = gtn(glyph_name)
    return tuple(table)


@lru_cache(maxsize=256)
def cached_decode_table(
    key: str, differences_items: tuple[tuple[int, str], ...]
) -> tuple[str, ...]:
    differences = dict(differences_items)
    return build_decode_table(ENCODING_FALLBACKS.get(key, fallback_with_pdfdoc), differences)


def parse_differences(
    value: Any, resolve_name: Callable[[Any], str | None] | None = None
) -> dict[int, str]:
    differences: dict[int, str] = {}
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid encoding differences array")
    code = 0
    for item in value:
        if isinstance(item, int):
            if item < 0 or item > 255:
                raise ValueError("invalid encoding differences array")
            code = item
            continue
        if resolve_name is not None:
            glyph_name = resolve_name(item)
        else:
            glyph_name = parse_name(item)
        if glyph_name is None:
            raise ValueError("invalid encoding differences array")
        differences[code] = glyph_name
        code += 1
    return differences


ENCODING_FALLBACKS: dict[str, EncodingFallback] = {
    "Type3": lambda b: "",
    "WinAnsiEncoding": lambda b: bytes([b]).decode("cp1252", "replace"),
    "MacRomanEncoding": lambda b: bytes([b]).decode("mac_roman", "replace"),
}
