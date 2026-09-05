"""Shared text-state transitions for compatibility projections."""

from __future__ import annotations

import math
from collections.abc import Sequence

from core_pdf.impl._impl.model.text import is_neutral_character, is_rtl_character
from core_pdf.impl.spec.s_08_graphics.matrix import multiply_affine
from core_pdf.impl.spec.s_09_fonts.helpers import build_decode_table

# pypdf's _predefined_cmap, which is what both legacy projections have to
# agree with: the pypdf facade emulates pypdf directly, and the LlamaIndex one
# emulates llama_index's PDFReader, which wraps pypdf. The codec choices are
# pypdf's rather than the specification's -- GBK for GB-EUC, cp950 for Big5 --
# so they are transcribed, not derived.
internal_PREDEFINED_ENCODING_CODECS = {
    "Identity-H": "utf-16-be",
    "Identity-V": "utf-16-be",
    "GB-EUC-H": "gbk",
    "GB-EUC-V": "gbk",
    "GBpc-EUC-H": "gb2312",
    "GBpc-EUC-V": "gb2312",
    "GBK-EUC-H": "gbk",
    "GBK-EUC-V": "gbk",
    "GBK2K-H": "gb18030",
    "GBK2K-V": "gb18030",
    "ETen-B5-H": "cp950",
    "ETen-B5-V": "cp950",
    "ETenms-B5-H": "cp950",
    "ETenms-B5-V": "cp950",
    "UniCNS-UTF16-H": "utf-16-be",
    "UniCNS-UTF16-V": "utf-16-be",
    "UniGB-UTF16-H": "gb18030",
    "UniGB-UTF16-V": "gb18030",
    "90ms-RKSJ-H": "cp932",
    "90ms-RKSJ-V": "cp932",
    "UniJIS-UTF16-H": "utf-16-be",
    "UniJIS-UTF16-V": "utf-16-be",
}


def internal_orientation(matrix: Sequence[float]) -> int:
    """Return the legacy extraction orientation for an affine matrix."""
    if matrix[3] > 1e-6:
        return 0
    if matrix[3] < -1e-6:
        return 180
    return 90 if matrix[1] > 0 else 270


def internal_legacy_base_table(name: str) -> list[str]:
    """Build the encoding table expected by the legacy text projections."""
    table = list(build_decode_table(name, ()))
    if name == "StandardEncoding":
        table = [value or chr(code) for code, value in enumerate(table)]
        table[174] = "ﬁ"
        table[175] = "ﬂ"
    elif name == "WinAnsiEncoding":
        for code in (127, 129, 141, 143, 144, 157):
            table[code] = chr(code)
        table[160] = "\xa0"
        table[173] = "\xad"
    elif name == "MacRomanEncoding":
        table[127] = "\x7f"
        table[202] = "\xa0"
        table[219] = "€"
        table[222] = "ﬁ"
        table[223] = "ﬂ"
        table[240] = "\uf8ff"
    return table


def internal_flush_text(output_parts: list[str], text: str, output_last: str) -> tuple[str, str]:
    """Move pending text to an append-only output buffer."""
    if text:
        output_parts.append(text)
        return "", text[-1]
    return text, output_last


def internal_append_directional_text(text: str, rtl: bool, value: str) -> tuple[str, bool]:
    """Apply the legacy left-to-right/right-to-left run transition."""
    if len(value) != 1 or is_neutral_character(value):
        return (value + text if rtl else text + value), rtl
    if is_rtl_character(value):
        return value + (text if rtl else ""), True
    return ("" if rtl else text) + value, False


def internal_positioned_text(
    output_parts: list[str],
    text: str,
    output_last: str,
    *,
    previous_text_matrix: Sequence[float],
    previous_current_matrix: Sequence[float],
    text_matrix: Sequence[float],
    current_matrix: Sequence[float],
    line_height: float,
    font_size: float,
    space_width: float,
    string_width: float,
) -> tuple[str, str]:
    """Insert the newline or space implied by a text-position transition."""
    trailing = text[-1:] or output_last[-1:]
    if not trailing:
        return text, output_last

    previous = multiply_affine(previous_text_matrix, previous_current_matrix)
    current = multiply_affine(text_matrix, current_matrix)
    delta_x = current[4] - previous[4]
    delta_y = current[5] - previous[5]
    previous_scale_x = math.hypot(previous_text_matrix[0], previous_text_matrix[1])
    previous_scale_y = math.hypot(previous_text_matrix[2], previous_text_matrix[3])
    current_scale_y = math.hypot(text_matrix[2], text_matrix[3])
    moved_height, moved_width = (
        (delta_y, delta_x) if internal_orientation(current) in (0, 180) else (delta_x, delta_y)
    )
    if abs(moved_height) > 0.8 * min(
        line_height * previous_scale_y,
        font_size * current_scale_y,
    ):
        if trailing != "\n":
            output_parts.append(text + "\n")
            return "", "\n"
    elif (
        moved_width >= (font_size * space_width / 1000.0 + string_width) * previous_scale_x
        and trailing != " "
    ):
        return text + " ", output_last
    return text, output_last


def internal_ensure_line_break(output_parts: list[str], output_last: str) -> str:
    """End non-empty output at a line boundary."""
    if output_last and output_last != "\n":
        output_parts.append("\n")
        return "\n"
    return output_last
