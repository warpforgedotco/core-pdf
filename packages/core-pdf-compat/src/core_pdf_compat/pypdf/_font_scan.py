# Follows pypdf._cmap and pypdf.generic._font of pypdf 7.0.0, Copyright (c) 2006, Mathieu
# Fenniak, under its BSD-style license.
"""pypdf's warnings while it reads a font's encoding and widths. core-pdf reads the font."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from core_pdf.impl.base.types import PdfName, PdfString
from core_pdf.impl.base.values import recover_pdf_name
from core_pdf_compat._text_state import legacy_base_table, predefined_encoding_codec
lazy from core_pdf_compat.pypdf.errors import LimitReachedError

cmap_logger = logging.getLogger("core_pdf_compat.pypdf._cmap")
font_logger = logging.getLogger("core_pdf_compat.pypdf._font")

# The keys of pypdf's charset_encoding.
CHARSET_ENCODINGS = frozenset(
    {
        "StandardEncoding",
        "WinAnsiEncoding",
        "MacRomanEncoding",
        "PDFDocEncoding",
        "Symbol",
        "ZapfDingbats",
    }
)

MAX_CID_WIDTH_ENTRY_COUNT = 65_536
MAX_WIDTH_ENTRY_COUNT = 100_000

Resolve = Callable[[object], object]


def scan_encoding(
    font: dict[object, object], resolve: Resolve, present: Callable[[object], Any] = str
) -> None:
    """Log what pypdf's _parse_encoding logs for an encoding it does not know, or malformed
    /Differences. ``present`` turns a core value into pypdf's object, which a warning formats."""
    if "Encoding" not in font:
        return
    encoding = resolve(font["Encoding"])
    unknown: str | None = None
    if isinstance(encoding, PdfName):
        name = recover_pdf_name(encoding) or ""
        if name not in CHARSET_ENCODINGS and predefined_encoding_codec(name) is None:
            unknown = f"/{name}"
    elif isinstance(encoding, PdfString):
        text = str(present(encoding))
        if "-UCS2-" not in text:
            unknown = text  # pypdf looks the string up as a name it lacks.
    elif isinstance(encoding, dict) and "BaseEncoding" in encoding:
        base = resolve(encoding["BaseEncoding"])
        if not isinstance(base, PdfName) or recover_pdf_name(base) not in CHARSET_ENCODINGS:
            # pypdf formats the encoding it has built so far, still an empty list.
            cmap_logger.error("Advanced encoding %s not implemented yet", [])
    if unknown is not None:
        subtype = resolve(font.get("Subtype"))
        composite = isinstance(subtype, PdfName) and recover_pdf_name(subtype) == "Type0"
        replacement: object = "utf-16-be" if composite else legacy_base_table("StandardEncoding")
        cmap_logger.error(
            "Advanced encoding %s not implemented yet, using %s instead.", unknown, replacement
        )
    if isinstance(encoding, dict) and "Differences" in encoding:
        differences = resolve(encoding["Differences"])
        if not isinstance(differences, (list, tuple)):
            cmap_logger.warning(
                "Font encoding differences are not an array: %s", present(differences)
            )


def _numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _sequence(value: object) -> bool:
    # pypdf's names and strings are Python strings, so they count as sequences.
    return isinstance(value, (list, tuple, PdfName, PdfString))


def _check_range(start: int, end: int) -> None:
    if end < start:
        raise LimitReachedError(f"Invalid CID width range: {start}..{end}.")
    if end - start > MAX_CID_WIDTH_ENTRY_COUNT:
        raise LimitReachedError(
            f"CID width range too large: {end - start} > {MAX_CID_WIDTH_ENTRY_COUNT}."
        )


def _check_count(count: int) -> None:
    if count > MAX_WIDTH_ENTRY_COUNT:
        raise LimitReachedError(f"Too many character widths: {count} > {MAX_WIDTH_ENTRY_COUNT}.")


def scan_cid_widths(
    descendant: dict[object, object],
    resolve: Resolve,
    present: Callable[[object], Any],
    widths: dict[int, float],
) -> bool:
    """Walk /W as pypdf's _collect_cid_character_widths does, with its warnings and limits.

    The widths pypdf reads go into ``widths``. The result is whether the walk logged a
    warning: only then can its widths differ from core-pdf's. ``present`` turns a core
    value into pypdf's object, which the warning formats.
    """
    array = resolve(descendant.get("W"))
    if not isinstance(array, (list, tuple)):
        return False
    count = len(array)
    skip = 0
    entries = 0
    irregular = False
    segments: list[tuple[int, Sequence[object]]] = []
    for index, raw_entry in enumerate(array):
        if skip:
            skip -= 1
            continue
        entry = resolve(raw_entry)
        if not _numeric(entry):
            font_logger.warning(
                "Expected numeric value for width, got %s. Ignoring it.", present(entry)
            )
            irregular = True
            continue
        start = int(_number(entry))
        following = resolve(array[index + 1]) if index + 1 < count else None
        if _sequence(following):
            listed: Sequence[object] = following if isinstance(following, (list, tuple)) else ()
            _check_range(start, start + len(listed))
            entries += len(listed)
            _check_count(entries)
            segments.append((start, listed))
            skip = 1
        elif _numeric(following) and index + 2 < count and _numeric(resolve(array[index + 2])):
            stop = int(_number(following))
            width = _number(resolve(array[index + 2]))
            _check_range(start, stop + 1)
            entries += stop - start + 1
            _check_count(entries)
            segments.append((start, (width,) * (stop - start + 1)))
            skip = 2
        else:
            font_logger.warning("Invalid font width definition. Last element: %s.", present(entry))
            irregular = True
    if irregular:
        for start, listed in segments:
            for code, raw_width in enumerate(listed, start=start):
                value = resolve(raw_width)
                if _numeric(value):
                    widths[code] = _number(value)
    return irregular


# A simple font's character codes are single bytes, so /Widths has at most 256 entries.
MAX_SIMPLE_FONT_CHARACTER_CODE = 255


def scan_simple_widths(
    font: dict[object, object], resolve: Resolve, present: Callable[[object], Any]
) -> None:
    """Log what pypdf's _parse_widths logs for a simple font's /Widths, and raise its limit."""
    if "Widths" not in font:
        return
    widths = resolve(font["Widths"])
    if not isinstance(widths, (list, tuple)):
        font_logger.warning("Expected an array for /Widths, got %s. Ignoring it.", present(widths))
        return
    limit = MAX_SIMPLE_FONT_CHARACTER_CODE + 1
    if len(widths) > limit:
        raise LimitReachedError(f"Too many character widths: {len(widths)} > {limit}.")
    first_char = resolve(font.get("FirstChar", 0))
    if not _numeric(first_char):
        font_logger.warning("Ignoring invalid /FirstChar %s.", present(first_char))
        return
    first = int(_number(first_char))
    if first < 0:
        font_logger.warning("Ignoring invalid /FirstChar %d < 0.", first)
        return
    if first + len(widths) - 1 > MAX_SIMPLE_FONT_CHARACTER_CODE:
        font_logger.warning(
            "Ignoring invalid character codes > %d (starting at %d).",
            MAX_SIMPLE_FONT_CHARACTER_CODE,
            max(first, MAX_SIMPLE_FONT_CHARACTER_CODE + 1),
        )
