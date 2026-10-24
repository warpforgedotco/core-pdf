from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence

from core_pdf.impl.fonts_helpers import build_decode_table
from core_pdf.impl.text import is_rtl_character
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, multiply_affine

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
    # UniGB covers the GB18030 character set, but its codes are UTF-16BE
    # (ISO 32000-2:2020, 9.7.5.2).
    "UniGB-UTF16-H": "utf-16-be",
    "UniGB-UTF16-V": "utf-16-be",
    "90ms-RKSJ-H": "cp932",
    "90ms-RKSJ-V": "cp932",
    "UniJIS-UTF16-H": "utf-16-be",
    "UniJIS-UTF16-V": "utf-16-be",
}


def predefined_encoding_codec(name: str) -> str | None:
    codec = PREDEFINED_ENCODING_CODECS.get(name)
    if codec is None and "-UCS2-" in name:
        return "utf-16-be"
    return codec


type LineSpan = tuple[int, float, float]


def number_operand(operands: Sequence[object], index: int, default: float) -> float:
    if index < len(operands):
        value = operands[index]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return default


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


# pypdf's neutral characters keep the current insertion order. Unlike core's,
# they include the Arabic-Indic and Extended Arabic-Indic digits, which pypdf
# no longer reverses (py-pdf/pypdf#4077).
NEUTRAL_CHARACTER_RANGES = (
    ("\x00", "\x2f"),
    ("\x3a", "\x40"),
    ("\u0660", "\u066d"),
    ("\u06f0", "\u06f9"),
    ("\u2000", "\u206f"),
    ("\u20a0", "\u21ff"),
)


def is_neutral_character(character: str) -> bool:
    return any(start <= character <= end for start, end in NEUTRAL_CHARACTER_RANGES)


def append_directional_text(text: str, rtl: bool, value: str) -> tuple[str, bool, str]:
    """Add ``value`` to the run ``text`` in display order.

    Returns the run, its direction, and the run a direction change completed
    ("" when the direction did not change).
    """
    if len(value) != 1 or is_neutral_character(value):
        return (value + text if rtl else text + value), rtl, ""
    if is_rtl_character(value):
        return (value + text, True, "") if rtl else (value, True, text)
    return (value, False, text) if rtl else (text + value, False, "")


def positioned_text(
    output_parts: list[str],
    text: str,
    output_last: str,
    line_span: LineSpan | None,
    *,
    previous_text_matrix: Sequence[float],
    previous_current_matrix: Sequence[float],
    text_matrix: Sequence[float],
    current_matrix: Sequence[float],
    line_height: float,
    font_size: float,
    space_width: float,
    string_width: float,
) -> tuple[str, str, LineSpan]:
    trailing = text[-1:] or output_last[-1:]
    previous = multiply_affine(previous_text_matrix, previous_current_matrix)
    current = multiply_affine(text_matrix, current_matrix)
    delta_x = current[4] - previous[4]
    delta_y = current[5] - previous[5]
    # The deltas are in the space of text matrix x CTM, so the scales they are
    # compared against come from the same combined matrices.
    previous_scale_x = math.hypot(previous[0], previous[1])
    previous_scale_y = math.hypot(previous[2], previous[3])
    current_scale_y = math.hypot(current[2], current[3])
    horizontal = orientation(current) in (0, 180)
    moved_height, moved_width = (delta_y, delta_x) if horizontal else (delta_x, delta_y)
    # The baselines already on the current line form a span; a fragment whose
    # baseline falls inside it (a row label centered between the two lines of
    # a wrapped cell) continues the line instead of breaking it.
    axis_index = 5 if horizontal else 4
    axis = current[axis_index]
    if line_span is None or line_span[0] != axis_index:
        distance = abs(moved_height)
    else:
        lower, upper = line_span[1], line_span[2]
        distance = 0.0 if lower <= axis <= upper else min(abs(axis - lower), abs(axis - upper))
    if distance > 0.8 * min(line_height * previous_scale_y, font_size * current_scale_y):
        if trailing not in ("", "\n"):
            output_parts.append(text + "\n")
            text, output_last = "", "\n"
        return text, output_last, (axis_index, axis, axis)
    if moved_width >= (
        font_size * space_width / 1000.0 + string_width
    ) * previous_scale_x and trailing not in ("", " "):
        text += " "
    if trailing in ("", "\n") or line_span is None or line_span[0] != axis_index:
        return text, output_last, (axis_index, axis, axis)
    return text, output_last, (axis_index, min(line_span[1], axis), max(line_span[2], axis))


def ensure_line_break(output_parts: list[str], output_last: str) -> str:
    if output_last and output_last != "\n":
        output_parts.append("\n")
        return "\n"
    return output_last


def embedded_font_program_count(descriptor: Mapping[object, object]) -> int:
    return sum(descriptor.get(key) is not None for key in EMBEDDED_FONT_PROGRAM_KEYS)


def type1_encoding_entries(segment: bytes) -> Iterator[tuple[int, str]]:
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
        self.line_span: LineSpan | None = None
        self.stack: list[tuple[list[float], FontT | None, float, float]] = []
        self.text = ""
        self.output_parts: list[str] = []
        self.output_last = ""
        self.rtl = False
        self.accumulated_width = 0.0
        self.actual_height = 0.0

    def flush(self) -> None:
        self.text, self.output_last = flush_text(self.output_parts, self.text, self.output_last)

    def add_directional(self, value: str) -> None:
        self.text, self.rtl, completed = append_directional_text(self.text, self.rtl, value)
        # pypdf moves the run a direction change completed straight to the output.
        if completed:
            self.output_parts.append(completed)
            self.output_last = completed[-1]

    def positioned(self, string_width: float) -> None:
        self.text, self.output_last, self.line_span = positioned_text(
            self.output_parts,
            self.text,
            self.output_last,
            self.line_span,
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
                # The leading is in unscaled text space units; T* applies the
                # text matrix to it (PDF 32000-1:2008, 9.3.5).
                self.leading = number_operand(operands, 0, 0.0)
            case _:
                return False
        return True
