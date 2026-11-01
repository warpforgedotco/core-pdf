# Follows pypdf._text_extraction._layout_mode, Copyright (c) 2006, Mathieu Fenniak, under its
# BSD-style license.
"""pypdf's "layout" text extraction: a fixed-width page built from text positions."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Iterator, Mapping
from copy import replace
from itertools import groupby
from typing import Any

from core_pdf.impl.types import PdfString
from core_pdf_compat._text_state import orientation
from core_pdf_compat.pypdf.errors import PdfReadError
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, multiply_affine
from core_records import GeneratedRecord

logger = logging.getLogger(__name__)

WHITESPACE_LIMIT = 10_000
NEWLINE_LIMIT = 1_000
NEW_GROUP_SPACE_WIDTHS = 5
CONTROL_CHARACTERS = str.maketrans(dict.fromkeys(range(14, 32), " "))

type Matrix = list[float]
type Operation = tuple[str, tuple[Any, ...]]


def _mult(left: Matrix, right: Matrix) -> Matrix:
    return list(multiply_affine(left, right))


class _Transform(GeneratedRecord):
    matrix: tuple[float, ...]
    is_text: bool = False
    is_render: bool = False


class _FontView:
    """What layout needs from a font. ``font`` is None for one pypdf cannot read."""

    def __init__(self, font: Any) -> None:
        self.font = None if font is None else replace(font, space_width=font.natural_space_width)
        self.interpretable = font is not None
        self.space_width = 250.0 if font is None else font.natural_space_width

    def shown(self, value: bytes) -> tuple[tuple[str, ...], float]:
        return (
            (tuple(value.decode("latin-1")), 0.0)
            if self.font is None
            else self.font.decode_parts(value)
        )


class _TextState:
    def __init__(self) -> None:
        self.stack = [_Transform(tuple(IDENTITY_MATRIX))]
        self.q_queue: Counter[int] = Counter()
        self.q_depth = [0]
        self.character_spacing = 0.0
        self.word_spacing = 0.0
        self.scaling = 100.0
        self.leading = 0.0
        self.rise = 0.0
        self.font_stack: list[tuple[_FontView | None, float]] = []
        self.font: _FontView | None = None
        self.font_size = 0.0

    def set_parameter(self, operator: str, operands: tuple[Any, ...]) -> None:
        if not operands:
            return
        value = operands[0]
        match operator:
            case "Tc":
                self.character_spacing = value
            case "Tw":
                self.word_spacing = value
            case "Tz":
                self.scaling = value
            case "TL":
                self.leading = value
            case "Ts":
                self.rise = value

    def run(self, value: bytes | None = None) -> _Run:
        if self.font is None:
            raise PdfReadError("font not set: is PDF missing a Tf operator?")
        return _Run(self, self.font, value, self.effective_transform())

    def reset_text(self) -> None:
        while self.stack[0].is_text or self.stack[0].is_render:
            self.stack = self.stack[1:]

    def reset_render(self) -> None:
        while self.stack[0].is_render:
            self.stack = self.stack[1:]

    def add_q(self) -> None:
        self.font_stack.append((self.font, self.font_size))
        self.q_depth.append(len(self.q_depth))

    def remove_q(self) -> None:
        self.font, self.font_size = self.font_stack.pop()
        self.reset_text()
        self.stack = self.stack[self.q_queue.pop(self.q_depth.pop(), 0) :]

    def add_cm(self, operands: tuple[float, ...]) -> None:
        self.reset_text()
        self.q_queue.update(self.q_depth[-1:])
        self.stack = [_Transform(tuple(map(float, operands))), *self.stack]

    def add_tm(self, operands: list[float]) -> None:
        self.stack = [_Transform(_complete(operands), is_text=True), *self.stack]

    def add_trm(self, operands: list[float]) -> None:
        self.stack = [_Transform(_complete(operands), is_text=True, is_render=True), *self.stack]

    def effective_transform(self) -> Matrix:
        effective = list(self.stack[0].matrix)
        for transform in self.stack[1:]:
            effective = _mult(effective, list(transform.matrix))
        return effective


def _complete(operands: list[float]) -> tuple[float, ...]:
    if len(operands) == 2:
        return (1.0, 0.0, 0.0, 1.0, float(operands[0]), float(operands[1]))
    return tuple(map(float, operands))


class _Run:
    """A string shown with one text state."""

    def __init__(
        self, state: _TextState, font: _FontView, value: bytes | None, transform: Matrix
    ) -> None:
        self.font = font
        self.font_size = state.font_size
        self.character_spacing = state.character_spacing
        self.word_spacing = state.word_spacing
        self.scaling = state.scaling
        self.rise = state.rise
        self.parts, self.width = font.shown(value) if value else ((), 0.0)
        self.text = "".join(self.parts)
        self.transform = transform
        self.rotated = False
        if orientation(self.transform) in (90, 270):
            self.transform = _mult(
                [1.0, -self.transform[1], -self.transform[2], 1.0, 0.0, 0.0], self.transform
            )
            self.rotated = True
        if orientation(self.transform) == 180 and self.transform[0] < -1e-6:
            self.transform = _mult([-1.0, 0.0, 0.0, -1.0, 0.0, 0.0], self.transform)
            self.rotated = True
        self.displaced_tx = _mult(self.displacement(), self.transform)[4]
        self.tx = self.transform[4]
        size_matrix = [
            self.font_size * self.scaling / 100.0,
            0.0,
            0.0,
            self.font_size,
            0.0,
            self.rise,
        ]
        self.ty = _mult(size_matrix, self.transform)[5]
        self.space_tx = round(self._advance(font.space_width, 1, 1), 3)
        if self.space_tx < 1e-6:
            self.space_tx = round(self._advance(font.space_width, 0, 0), 3)
        self.font_height = self.font_size * math.hypot(self.transform[1], self.transform[3])
        self.flip_vertical = self.transform[3] < -1e-6

    def _advance(self, width: float, characters: int, spaces: int) -> float:
        return (
            self.font_size * (width / 1000.0)
            + characters * self.character_spacing
            + spaces * self.word_spacing
        ) * (self.scaling / 100.0)

    def displacement(self, offset: float = 0.0) -> Matrix:
        tx = self._advance(self.width - offset, len(self.parts), self.parts.count(" "))
        return [1.0, 0.0, 0.0, 1.0, tx, 0.0]


def _bt_group(run: _Run, text: str, displaced_tx: float) -> dict[str, Any]:
    return {
        "tx": run.tx,
        "ty": run.ty,
        "font_size": run.font_size,
        "font_height": run.font_height,
        "text": text,
        "displaced_tx": displaced_tx,
        "flip_sort": -1 if run.flip_vertical else 1,
    }


def _spaces(actual: int) -> int:
    if actual > WHITESPACE_LIMIT:
        logger.warning(
            "Limiting excessive whitespace from %d to %d characters.", actual, WHITESPACE_LIMIT
        )
        return WHITESPACE_LIMIT
    return actual


def _close_text_object(runs: list[_Run], strip_rotated: bool) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    text = ""
    first = 0
    last_displaced_tx = runs[0].displaced_tx
    last_ty = runs[0].ty
    for index, run in enumerate(runs):
        if (strip_rotated and run.rotated) or not run.font.interpretable:
            continue
        # A jump of more than the font height starts a new line.
        if abs(run.ty - last_ty) > run.font_height:
            if text.strip():
                groups.append(_bt_group(runs[first], text, last_displaced_tx))
            first = index
            text = ""
        # A jump back of more than a few spaces starts a new group.
        if last_displaced_tx - run.tx > run.space_tx * NEW_GROUP_SPACE_WIDTHS:
            if text.strip():
                groups.append(_bt_group(runs[first], text, last_displaced_tx))
            first = index
            last_displaced_tx = run.displaced_tx
            text = ""
        excess_tx = round(run.tx - last_displaced_tx, 3) * (index != first)
        spaces = _spaces(int(excess_tx // run.space_tx) if run.space_tx else 0)
        last_ty = run.ty
        text = f"{text}{' ' * spaces}{run.text}"
        last_displaced_tx = run.displaced_tx
    if text:
        groups.append(_bt_group(runs[first], text, last_displaced_tx))
    return groups


class _Walker:
    def __init__(self, operations: Iterator[Operation], fonts: Mapping[str, Any]) -> None:
        self.operations = operations
        self.fonts = fonts
        self.state = _TextState()
        self.strip_rotated = True

    def resolve_font(self, name: object) -> _FontView:
        if str(name) in self.fonts:
            return _FontView(self.fonts[str(name)])
        logger.warning("Font /%s is not in the page resources.", name)
        return _FontView(None)

    def walk(self, end_target: str) -> tuple[list[dict[str, Any]], list[_Run]]:
        state = self.state
        groups: list[dict[str, Any]] = []
        runs: list[_Run] = []
        if end_target == "Q":
            state.add_q()
        for operator, operands in self.operations:
            if operator == end_target:
                if operator == "Q":
                    state.remove_q()
                elif runs:
                    groups.extend(_close_text_object(runs, self.strip_rotated))
                    state.reset_text()
                return groups, runs
            self._step(operator, operands, groups, runs)
        logger.warning("Unbalanced target operations, expected %r.", end_target.encode())
        return groups, runs

    def _step(
        self,
        operator: str,
        operands: tuple[Any, ...],
        groups: list[dict[str, Any]],
        runs: list[_Run],
    ) -> None:
        state = self.state
        match operator:
            case "q" | "BT":
                inner_groups, inner_runs = self.walk("Q" if operator == "q" else "ET")
                groups.extend(inner_groups)
                runs.extend(inner_runs)
            case "cm":
                state.add_cm(operands)
            case "Tj":
                runs.append(state.run(_string(operands[0])))
            case "TJ":
                run = state.run()
                for item in operands[0]:
                    if isinstance(item, PdfString):
                        run = state.run(bytes(item.data))
                        runs.append(run)
                    else:
                        state.add_trm(run.displacement(item))
            case "'":
                state.reset_render()
                state.add_tm([0, -state.leading])
                runs.append(state.run(_string(operands[0])))
            case '"':
                state.reset_render()
                state.word_spacing, state.character_spacing = operands[0], operands[1]
                state.add_tm([0, -state.leading])
                runs.append(state.run(_string(operands[2])))
            case "Td" | "Tm" | "TD" | "T*":
                state.reset_render()
                if operator == "Tm":
                    state.reset_text()
                elif operator == "TD":
                    state.leading = -operands[1]
                matrix = [0, -state.leading] if operator == "T*" else list(operands)
                state.add_tm(matrix)
            case "Tf":
                state.font = self.resolve_font(operands[0])
                state.font_size = operands[1]
            case _:
                state.set_parameter(operator, operands)


def _string(value: object) -> bytes:
    return bytes(value.data) if isinstance(value, PdfString) else b""


def text_show_operations(
    operations: Iterator[Operation], fonts: Mapping[str, Any], strip_rotated: bool
) -> list[dict[str, Any]]:
    walker = _Walker(operations, fonts)
    walker.strip_rotated = strip_rotated
    state = walker.state
    groups: list[dict[str, Any]] = []
    runs: list[_Run] = []
    for operator, operands in operations:
        if operator in {"BT", "q"}:
            inner_groups, inner_runs = walker.walk("ET" if operator == "BT" else "Q")
            groups.extend(inner_groups)
            runs.extend(inner_runs)
        elif operator == "Tf":
            state.font = walker.resolve_font(operands[0])
            state.font_size = operands[1]
        else:
            state.set_parameter(operator, operands)
    if any(run.rotated for run in runs):
        outcome = "Output will be incomplete" if strip_rotated else "Layout will be degraded"
        logger.warning("Rotated text discovered. %s.", outcome)
    if not all(run.font.interpretable for run in runs):
        logger.warning("PDF contains an uninterpretable font. Output will be incomplete.")
    left = min((group["tx"] for group in groups), default=0.0)
    ordered = sorted(
        groups, key=lambda group: (group["ty"] * group["flip_sort"], -group["tx"]), reverse=True
    )
    return [
        {**group, "tx": group["tx"] - left, "displaced_tx": group["displaced_tx"] - left}
        for group in ordered
    ]


def y_coordinate_groups(groups: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_line = {
        ty: sorted(line, key=lambda group: group["tx"])
        for ty, line in groupby(groups, key=lambda group: int(group["ty"] * group["flip_sort"]))
    }
    last_ty = next(iter(by_line))
    last_txs = {int(group["tx"]) for group in by_line[last_ty] if group["text"].strip()}
    for ty in list(by_line)[1:]:
        font_height = min(by_line[key][0]["font_height"] for key in (ty, last_ty))
        txs = {int(group["tx"]) for group in by_line[ty] if group["text"].strip()}
        if not txs & last_txs and abs(ty - last_ty) < font_height:
            by_line[last_ty] = sorted(
                by_line.pop(ty) + by_line[last_ty], key=lambda group: group["tx"]
            )
            last_txs |= txs
        else:
            last_ty = ty
            last_txs = txs
    return by_line


def fixed_char_width(groups: list[dict[str, Any]], scale_weight: float) -> float:
    samples = []
    for group in groups:
        length = len(group["text"]) * scale_weight
        samples.append(((group["displaced_tx"] - group["tx"]) / length, length))
    return sum(width * length for width, length in samples) / sum(length for _, length in samples)


def _blank_lines(gap: int, font_height: float, weight: float) -> int:
    if font_height == 0:
        return 0
    blank_lines = int(abs(gap) / (font_height * weight)) - 1
    if blank_lines > NEWLINE_LIMIT:
        logger.warning("Limiting excessive newlines from %d to %d.", blank_lines, NEWLINE_LIMIT)
        return NEWLINE_LIMIT
    return blank_lines


def fixed_width_page(
    by_line: dict[int, list[dict[str, Any]]],
    char_width: float,
    space_vertically: bool,
    font_height_weight: float,
) -> str:
    lines: list[str] = []
    last_y = 0
    for y, line in by_line.items():
        if space_vertically and lines:
            gap = y - last_y
            lines.extend([""] * _blank_lines(gap, line[0]["font_height"], font_height_weight))
        parts: list[str] = []
        length = 0
        last_displaced = 0.0
        for group in line:
            tx = group["tx"]
            needed = int(tx // char_width) - length
            if needed > 0 and math.ceil(last_displaced) < int(tx):
                needed = _spaces(needed)
                parts.append(" " * needed)
                length += needed
            text = group["text"].translate(CONTROL_CHARACTERS)
            parts.append(text)
            length += len(text)
            last_displaced = group["displaced_tx"]
        full_line = "".join(parts).rstrip()
        if full_line.strip() or (space_vertically and lines):
            lines.append(full_line)
        last_y = y
    return "\n".join(lines)


def layout_text(
    operations: Iterator[Operation],
    fonts: Mapping[str, Any],
    *,
    space_vertically: bool,
    scale_weight: float,
    strip_rotated: bool,
    font_height_weight: float,
) -> str:
    groups = text_show_operations(operations, fonts, strip_rotated)
    if not groups:
        return ""
    by_line = y_coordinate_groups(groups)
    char_width = fixed_char_width(groups, scale_weight)
    return fixed_width_page(by_line, char_width, space_vertically, font_height_weight)
