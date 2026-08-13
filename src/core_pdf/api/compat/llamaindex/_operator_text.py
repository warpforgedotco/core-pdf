"""PDF text-operator projection for the LlamaIndex compatibility facade."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from core_pdf.impl.engine.spec.s_07_content.operations import iter_content_operations
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.objects import PdfStream, PdfString


def internal_mult(left: list[float], right: list[float]) -> list[float]:
    return [
        left[0] * right[0] + left[1] * right[2],
        left[0] * right[1] + left[1] * right[3],
        left[2] * right[0] + left[3] * right[2],
        left[2] * right[1] + left[3] * right[3],
        left[4] * right[0] + left[5] * right[2] + right[4],
        left[4] * right[1] + left[5] * right[3] + right[5],
    ]


def internal_orientation(matrix: list[float]) -> int:
    if matrix[3] > 1e-6:
        return 0
    if matrix[3] < -1e-6:
        return 180
    return 90 if matrix[1] > 0 else 270


def internal_space_width(
    font: Mapping[object, object], resolver: Any, decoder: FontDecoder
) -> float:
    first_char = resolver.resolve(lookup_dict_key(font, "FirstChar"))
    widths = resolver.resolve(lookup_dict_key(font, "Widths"))
    if not isinstance(first_char, (int, float)) or not isinstance(widths, (list, tuple)):
        return 200.0
    encoding = resolver.resolve(lookup_dict_key(font, "Encoding"))
    difference_space = next(
        (code for code, glyph_name in decoder.differences.items() if glyph_name == "space"),
        None,
    )
    cmap_space = next(
        (
            code[0]
            for code, text in (
                decoder.to_unicode.mappings.items() if decoder.to_unicode is not None else ()
            )
            if len(code) == 1 and text == " "
        ),
        None,
    )
    code = (
        difference_space
        if difference_space is not None
        else cmap_space
        if not isinstance(encoding, dict) and cmap_space is not None
        else 32
    )
    index = code - int(first_char)
    return float(widths[index]) if 0 <= index < len(widths) else 200.0


@dataclass(frozen=True, slots=True)
class internal_Font:
    decoder: FontDecoder
    space_width: float

    def decode(self, data: bytes) -> str:
        if self.decoder.to_unicode is not None:
            if self.decoder.to_unicode.decode_lengths == (1,):
                return "".join(
                    " " if code == 32 else self.decoder.to_unicode.decode(bytes((code,)))
                    for code in data
                )
            return self.decoder.to_unicode.decode(data)
        output: list[str] = []
        for glyph in self.decoder.decode_glyphs(data):
            glyph_name = self.decoder.differences.get(glyph.char_code)
            if self.decoder.is_type3 and glyph_name:
                if glyph_name.isdigit():
                    output.append(f"/{glyph_name}")
                    continue
                if glyph_name.startswith("a") and glyph_name[1:].isdigit():
                    output.append(chr(glyph.char_code))
                    continue
            output.append(" " if glyph.code_bytes == b" " else glyph.unicode)
        return "".join(output)

    def text_width(self, data: bytes) -> float:
        return sum(
            self.space_width if glyph.unicode == " " else self.decoder.glyph_width(glyph.width_code)
            for glyph in self.decoder.decode_glyphs(data)
        )


class internal_TextState:
    def __init__(self, fonts: Mapping[str, internal_Font]) -> None:
        self.fonts = fonts
        self.font: internal_Font | None = None
        self.font_name: str | None = None
        self.font_size = 12.0
        self.half_space_width = 125.0
        self.leading = 0.0
        self.cm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.tm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.previous_cm = self.cm.copy()
        self.previous_tm = self.tm.copy()
        self.memo_cm = self.cm.copy()
        self.memo_tm = self.tm.copy()
        self.stack: list[tuple[list[float], internal_Font | None, str | None, float, float]] = []
        self.text = ""
        self.output = ""
        self.width = 0.0
        self.height = 0.0
        self.rtl = False

    def flush(self) -> None:
        self.output += self.text
        self.text = ""
        self.memo_cm = self.cm.copy()
        self.memo_tm = self.tm.copy()

    def positioned(self, string_width: float) -> None:
        previous = internal_mult(self.previous_tm, self.previous_cm)
        current = internal_mult(self.tm, self.cm)
        orientation = internal_orientation(current)
        delta_x = current[4] - previous[4]
        delta_y = current[5] - previous[5]
        previous_scale_x = math.hypot(self.previous_tm[0], self.previous_tm[1])
        previous_scale_y = math.hypot(self.previous_tm[2], self.previous_tm[3])
        current_scale_y = math.hypot(self.tm[2], self.tm[3])
        moved_height, moved_width = (
            (delta_y, delta_x) if orientation in (0, 180) else (delta_x, delta_y)
        )
        try:
            if abs(moved_height) > 0.8 * min(
                self.height * previous_scale_y, self.font_size * current_scale_y
            ):
                if (self.output + self.text)[-1] != "\n":
                    self.output += self.text + "\n"
                    self.text = ""
            elif (
                moved_width
                >= (self.font_size * self.half_space_width / 1000.0 + string_width)
                * previous_scale_x
                and (self.output + self.text)[-1] != " "
            ):
                self.text += " "
        except (IndexError, ValueError):
            pass
        self.previous_tm = self.tm.copy()
        self.previous_cm = self.cm.copy()
        if not self.text:
            self.memo_tm = self.tm.copy()
            self.memo_cm = self.cm.copy()

    def show(self, data: bytes) -> None:
        if self.font is None:
            decoded = "�" * len(data)
            width = 250.0 * len(data)
        else:
            decoded = self.font.decode(data)
            width = self.font.text_width(data)
        for character in decoded:
            direction = unicodedata.bidirectional(character)
            if direction in {"R", "AL", "AN"}:
                if not self.rtl:
                    self.rtl = True
                    self.text = ""
                self.text = character + self.text
            else:
                if self.rtl and direction not in {"B", "S", "WS", "ON", "BN", "CS", "ES", "ET"}:
                    self.rtl = False
                    self.text = ""
                self.text += character
        self.width += width * self.font_size
        self.height = self.font_size
        self.positioned(0.0)


class OperatorTextProjection:
    """Interpret page and form text operators using core-pdf's object and font engines."""

    def __init__(self, page: Any) -> None:
        self.page = page
        self.resolver = page.document.resolver

    def internal_fonts(self, resources: Mapping[object, object]) -> dict[str, internal_Font]:
        result: dict[str, internal_Font] = {}
        fonts = self.resolver.resolve(lookup_dict_key(resources, "Font"))
        if not isinstance(fonts, dict):
            return result
        for name, raw_font in fonts.items():
            font = self.resolver.resolve(raw_font)
            if not isinstance(font, dict):
                continue
            resolved = self.resolver.resolve_font_dict(font)
            decoder = FontDecoder(cast(dict[str, object], resolved))
            result[str(name)] = internal_Font(
                decoder=decoder,
                space_width=internal_space_width(font, self.resolver, decoder),
            )
        return result

    def internal_streams(self, value: object) -> tuple[PdfStream, ...]:
        resolved = self.resolver.resolve(value)
        if isinstance(resolved, PdfStream):
            return (self.resolver.resolve_stream(resolved),)
        if isinstance(resolved, (list, tuple)):
            return tuple(
                self.resolver.resolve_stream(stream)
                for item in resolved
                if isinstance((stream := self.resolver.resolve(item)), PdfStream)
            )
        return ()

    def internal_extract(
        self,
        streams: tuple[PdfStream, ...],
        resources: Mapping[object, object],
    ) -> str:
        state = internal_TextState(self.internal_fonts(resources))
        xobjects = self.resolver.resolve(lookup_dict_key(resources, "XObject"))
        for stream in streams:
            for operator, raw_operands in iter_content_operations(PdfLexer(stream.data)):
                operands = list(raw_operands)
                if operator == "BT":
                    state.tm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                    state.flush()
                elif operator == "ET":
                    state.flush()
                elif operator == "q":
                    state.stack.append(
                        (
                            state.cm.copy(),
                            state.font,
                            state.font_name,
                            state.font_size,
                            state.leading,
                        )
                    )
                elif operator == "Q":
                    if state.stack:
                        state.cm, state.font, state.font_name, state.font_size, state.leading = (
                            state.stack.pop()
                        )
                    else:
                        state.cm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                elif operator == "cm":
                    state.flush()
                    try:
                        state.cm = internal_mult([float(value) for value in operands[:6]], state.cm)
                    except (TypeError, ValueError):
                        state.cm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                    state.memo_cm = state.cm.copy()
                    state.memo_tm = state.tm.copy()
                elif operator == "TL":
                    scale_x = math.hypot(state.tm[0], state.tm[2])
                    state.leading = (
                        float(operands[0]) * state.font_size * scale_x if operands else 0.0
                    )
                elif operator == "Tf":
                    state.flush()
                    if operands:
                        state.font_name = str(operands[0])
                        state.font = state.fonts.get(state.font_name)
                    if len(operands) > 1:
                        state.font_size = float(operands[1])
                    state.half_space_width = (
                        state.font.space_width / 2.0 if state.font is not None else 125.0
                    )
                elif operator in {"Td", "TD"}:
                    tx = float(operands[0]) if operands else 0.0
                    ty = float(operands[1]) if len(operands) > 1 else 0.0
                    if operator == "TD":
                        scale_x = math.hypot(state.tm[0], state.tm[2])
                        state.leading = -ty * state.font_size * scale_x
                    state.tm[4] += tx * state.tm[0] + ty * state.tm[2]
                    state.tm[5] += tx * state.tm[1] + ty * state.tm[3]
                    state.positioned(state.width / 1000.0)
                    state.width = 0.0
                elif operator == "Tm":
                    try:
                        matrix = [float(value) for value in operands[:6]]
                    except (TypeError, ValueError):
                        matrix = []
                    state.tm = matrix if len(matrix) == 6 else [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                    state.positioned(state.width / 1000.0)
                    state.width = 0.0
                elif operator == "T*":
                    state.tm[4] -= state.leading * state.tm[2]
                    state.tm[5] -= state.leading * state.tm[3]
                    state.positioned(state.width / 1000.0)
                    state.width = 0.0
                elif operator in {"Tj", "'", '"'}:
                    if operator in {"'", '"'}:
                        state.tm[4] -= state.leading * state.tm[2]
                        state.tm[5] -= state.leading * state.tm[3]
                        state.positioned(state.width / 1000.0)
                        state.width = 0.0
                    value = operands[-1] if operands else None
                    state.show(bytes(value.data) if isinstance(value, PdfString) else b"")
                elif operator == "TJ" and operands and isinstance(operands[0], (list, tuple)):
                    threshold = state.half_space_width * 0.95
                    for item in operands[0]:
                        if isinstance(item, PdfString):
                            state.show(bytes(item.data))
                        elif (
                            isinstance(item, (int, float))
                            and abs(float(item)) >= threshold
                            and state.text
                            and state.text[-1] != " "
                        ):
                            state.show(b" ")
                elif operator == "Do" and operands and isinstance(xobjects, dict):
                    state.flush()
                    form = self.resolver.resolve(xobjects.get(operands[0]))
                    if not isinstance(form, PdfStream):
                        form = self.resolver.resolve(xobjects.get(str(operands[0])))
                    if (
                        isinstance(form, PdfStream)
                        and str(lookup_dict_key(form.dictionary, "Subtype")) != "Image"
                    ):
                        form_resources = self.resolver.resolve(
                            lookup_dict_key(form.dictionary, "Resources")
                        )
                        if not isinstance(form_resources, dict):
                            form_resources = resources
                        nested = self.internal_extract(
                            (self.resolver.resolve_stream(form),), form_resources
                        )
                        state.output += nested
        state.flush()
        return state.output

    def extract_text(self) -> str:
        resources = self.page.resolve_resources()
        return self.internal_extract(tuple(self.page.content_streams), resources)
