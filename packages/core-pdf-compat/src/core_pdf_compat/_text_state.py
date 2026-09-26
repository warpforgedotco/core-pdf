from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence

from core_pdf.impl.fonts_helpers import build_decode_table
from core_pdf.impl.text import is_neutral_character, is_rtl_character
from core_pdf_spec.s_08_graphics.matrix import multiply_affine

IDENTITY_MATRIX = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
EMBEDDED_FONT_PROGRAM_KEYS = ("FontFile", "FontFile2", "FontFile3")

PREDEFINED_ENCODING_CODECS = {
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


def orientation(matrix: Sequence[float]) -> int:
    if matrix[3] > 1e-6:
        return 0
    if matrix[3] < -1e-6:
        return 180
    return 90 if matrix[1] > 0 else 270


def legacy_base_table(name: str) -> list[str]:
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


def flush_text(output_parts: list[str], text: str, output_last: str) -> tuple[str, str]:
    if text:
        output_parts.append(text)
        return "", text[-1]
    return text, output_last


def append_directional_text(text: str, rtl: bool, value: str) -> tuple[str, bool]:
    if len(value) != 1 or is_neutral_character(value):
        return (value + text if rtl else text + value), rtl
    if is_rtl_character(value):
        return value + (text if rtl else ""), True
    return ("" if rtl else text) + value, False


def positioned_text(
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
        (delta_y, delta_x) if orientation(current) in (0, 180) else (delta_x, delta_y)
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


def ensure_line_break(output_parts: list[str], output_last: str) -> str:
    if output_last and output_last != "\n":
        output_parts.append("\n")
        return "\n"
    return output_last


def embedded_font_program_count(descriptor: Mapping[object, object]) -> int:
    return sum(descriptor.get(key) is not None for key in EMBEDDED_FONT_PROGRAM_KEYS)


def type1_encoding_entries(segment: bytes) -> Iterator[tuple[int, str]]:
    """The code and glyph name of each `dup <code> /<name> put` line, in order.

    segment is the clear text of a Type 1 program after its /Encoding key;
    lines that are not such an entry, and codes outside a byte, are skipped.
    """
    for line in segment.replace(b"\r", b"\n").split(b"\n"):
        if not line.startswith(b"dup"):
            continue
        words = [word for word in line.split(b" ") if word]
        if len(words) < 3 or (len(words) > 3 and words[3] != b"put"):
            continue
        try:
            code = int(words[1])
        except ValueError:
            continue
        if 0 <= code <= 255:
            yield code, words[2].removeprefix(b"/").decode("latin-1")


class TextMachine[FontT]:
    """The text and graphics state the legacy text extractors share.

    It holds what pypdf's and LlamaIndex's operator walks keep alike, and
    runs the operators they treat alike: BT, ET, q, Q, cm and TL. Each facade
    keeps its own font selection, positioning and showing operators, which
    differ between them down to the float arithmetic of T*.
    """

    def __init__(self, fonts: Mapping[str, FontT]) -> None:
        self.fonts = fonts
        self.font: FontT | None = None
        self.font_size = 12.0
        self.half_space_width = 125.0
        self.leading = 0.0
        self.cm = list(IDENTITY_MATRIX)
        self.tm = list(IDENTITY_MATRIX)
        self.previous_cm = self.cm.copy()
        self.previous_tm = self.tm.copy()
        self.stack: list[tuple[list[float], FontT | None, float, float]] = []
        self.text = ""
        self.output_parts: list[str] = []
        self.output_last = ""
        self.rtl = False
        self.accumulated_width = 0.0
        self.actual_height = 0.0

    def flush(self) -> None:
        self.text, self.output_last = flush_text(self.output_parts, self.text, self.output_last)

    def positioned(self, string_width: float) -> None:
        self.text, self.output_last = positioned_text(
            self.output_parts,
            self.text,
            self.output_last,
            previous_text_matrix=self.previous_tm,
            previous_current_matrix=self.previous_cm,
            text_matrix=self.tm,
            current_matrix=self.cm,
            line_height=self.actual_height,
            font_size=self.font_size,
            space_width=self.half_space_width,
            string_width=string_width,
        )
        self.previous_tm = self.tm.copy()
        self.previous_cm = self.cm.copy()

    def apply_state_operator(self, operator: str, operands: Sequence[object]) -> bool:
        """Run operator if it is one both extractors treat alike; say whether it was."""
        match operator:
            case "BT":
                self.tm = list(IDENTITY_MATRIX)
                self.flush()
            case "ET":
                self.flush()
            case "q":
                self.stack.append((self.cm.copy(), self.font, self.font_size, self.leading))
            case "Q":
                if self.stack:
                    self.cm, self.font, self.font_size, self.leading = self.stack.pop()
                else:
                    self.cm = list(IDENTITY_MATRIX)
            case "cm":
                self.flush()
                try:
                    values = [float(value) for value in operands[:6]]  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                except TypeError, ValueError:
                    values = []
                self.cm = (
                    list(multiply_affine(values, self.cm))
                    if len(values) == 6
                    else list(IDENTITY_MATRIX)
                )
            case "TL":
                scale_x = math.hypot(self.tm[0], self.tm[2])
                self.leading = float(operands[0]) * self.font_size * scale_x if operands else 0.0  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            case _:
                return False
        return True
