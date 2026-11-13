# Follows pypdf._text_extraction._layout_mode of pypdf 7.0.0, Copyright (c) 2006, Mathieu
# Fenniak, under its BSD-style license.
"""pypdf's "layout" text extraction: a fixed-width page built from text positions."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections import Counter
from collections.abc import Iterator, Mapping
from copy import replace
from itertools import groupby
from pathlib import Path
from typing import Any

from core_pdf.impl.base.types import PdfString
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX, multiply_affine
from core_pdf_compat._text_state import orientation
from core_pdf_compat.pypdf.errors import PdfReadError
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
        self.font = (
            None
            if font is None
            else (
                font
                if font.space_width == font.natural_space_width
                else replace(
                    font,
                    space_width=font.natural_space_width,
                    simple_byte_table=None,
                    parts_cache=None,
                )
            )
        )
        self.interpretable = font is not None and font.interpretable
        self.space_width = 250.0 if font is None else font.natural_space_width

    def counted(self, value: bytes) -> tuple[int, int]:
        """The characters and spaces pypdf's layout mode counts in the bytes, for Tc and Tw."""
        if self.font is None:
            return len(value), value.count(b" ")
        word, space = self.font.layout_word(value)
        return len(word), word.count(space) if space else 0

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
        # A missing or non-numeric operand leaves the parameter as it is.
        value = operands[0] if operands else None
        match operator:
            case "Tc":
                self.character_spacing = _as_float(value, self.character_spacing)
            case "Tw":
                self.word_spacing = _as_float(value, self.word_spacing)
            case "Tz":
                self.scaling = _as_float(value, self.scaling)
            case "TL":
                self.leading = _as_float(value, self.leading)
            case "Ts":
                self.rise = _as_float(value, self.rise)

    def set_font(self, font: _FontView, operands: tuple[Any, ...]) -> None:
        """pypdf's set_font_operation: a missing or non-numeric size keeps the current one."""
        self.font = font
        self.font_size = _as_float(operands[1] if len(operands) > 1 else None, self.font_size)

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
        # pypdf 7.0.0 passes the 6 operands nearest to the operator to new_transform.
        self.stack = [_Transform(_padded(operands[-6:])), *self.stack]

    def add_tm(self, operands: list[float]) -> None:
        matrix = _complete(operands)
        top = self.stack[0]
        if top.is_text and not top.is_render:
            # pypdf folds a text matrix into the one on top when that renders nothing.
            combined = tuple(_mult(list(matrix), list(top.matrix)))
            self.stack = [_Transform(combined, is_text=True), *self.stack[1:]]
        else:
            self.stack = [_Transform(matrix, is_text=True), *self.stack]

    def add_trm(self, operands: list[float]) -> None:
        self.stack = [_Transform(_complete(operands), is_text=True, is_render=True), *self.stack]

    def effective_transform(self) -> Matrix:
        effective = list(self.stack[0].matrix)
        for transform in self.stack[1:]:
            effective = _mult(effective, list(transform.matrix))
        return effective


def _as_float(value: object, default: float) -> float:
    """pypdf's TextStateManager._as_float: an operand as a float, else the default."""
    if isinstance(value, bool):
        return default  # pypdf's BooleanObject is no number.
    try:
        return float(value)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    except TypeError, ValueError:
        return default


def _padded(operands: Any) -> tuple[float, ...]:
    """pypdf's raw_transform: the numbers given, and the identity's for the rest."""
    values = (*operands, *IDENTITY_MATRIX[len(operands) :])
    return tuple(_as_float(value, default) for value, default in zip(values, IDENTITY_MATRIX))


def _complete(operands: list[float]) -> tuple[float, ...]:
    if len(operands) == 2:
        return _padded([1.0, 0.0, 0.0, 1.0, *operands])
    # pypdf 7.0.0 keeps the 6 operands nearest to the operator.
    return _padded(operands[-6:])


class _Run:
    """A string shown with one text state."""

    def __init__(
        self, state: _TextState, font: _FontView, value: bytes | None, transform: Matrix
    ) -> None:
        self.font = font
        self.value = value
        self.font_size = state.font_size
        self.character_spacing = state.character_spacing
        self.word_spacing = state.word_spacing
        self.scaling = state.scaling
        self.rise = state.rise
        self.parts, self.width = font.shown(value) if value else ((), 0.0)
        self.characters, self.spaces = font.counted(value) if value else (0, 0)
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
        # pypdf applies the character spacing to each character.
        return (
            self.font_size * (width / 1000.0)
            + characters * self.character_spacing
            + spaces * self.word_spacing
        ) * (self.scaling / 100.0)

    def displacement(self, offset: float = 0.0) -> Matrix:
        tx = self._advance(self.width - offset, self.characters, self.spaces)
        return [1.0, 0.0, 0.0, 1.0, tx, 0.0]

    def to_dict(self) -> dict[str, Any]:
        """The fields of pypdf's TextStateParams.to_dict, for layout_mode_debug_path."""
        value = b"" if self.value is None else self.value
        return {
            "value": value,
            "font_size": self.font_size,
            "Tc": self.character_spacing,
            "Tw": self.word_spacing,
            "Tz": self.scaling,
            "TL": 0.0,
            "Ts": self.rise,
            "transform": list(self.transform),
            "tx": self.tx,
            "ty": self.ty,
            "displaced_tx": self.displaced_tx,
            "space_tx": self.space_tx,
            "font_height": self.font_height,
            "flip_vertical": self.flip_vertical,
            "rotated": self.rotated,
            "text": self.text,
            "_raw_chars": "".join(map(chr, value)),
        }


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
        # space_tx is in text space and excess_tx in page space: pypdf scales the space width
        # by the horizontal transform, which is 0 for text with no width.
        page_space_tx = run.space_tx * abs(run.transform[0])
        spaces = _spaces(round(excess_tx / page_space_tx) if excess_tx > 0 and page_space_tx else 0)
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
            self._step(operator, operands, groups, runs, end_target)
        logger.warning("Unbalanced target operations, expected %r.", end_target.encode())
        return groups, runs

    def _step(
        self,
        operator: str,
        operands: tuple[Any, ...],
        groups: list[dict[str, Any]],
        runs: list[_Run],
        end_target: str,
    ) -> None:
        state = self.state
        match operator:
            case "q":
                inner_groups, inner_runs = self.walk("Q")
                groups.extend(inner_groups)
                runs.extend(inner_runs)
            case "BT" if end_target == "Q":
                inner_groups, inner_runs = self.walk("ET")
                groups.extend(inner_groups)
                runs.extend(inner_runs)
            case "BT":
                # Text objects do not nest and end at the first ET: a BT inside one only
                # resets the text matrix.
                state.reset_text()
            case "cm":
                state.add_cm(operands)
            case "Tj" if operands and isinstance(operands[0], PdfString):
                runs.append(state.run(bytes(operands[0].data)))
            case "TJ" if operands and isinstance(operands[0], list):
                run = state.run()
                # Only strings and numbers are TJ array items.
                for item in operands[0]:
                    if isinstance(item, PdfString):
                        run = state.run(bytes(item.data))
                        runs.append(run)
                    elif _is_number(item):
                        state.add_trm(run.displacement(item))
            case "'":
                state.reset_render()
                state.add_tm([0, -state.leading])
                self._show_operand(operands, 0, runs)
            case '"' if len(operands) >= 3:
                state.reset_render()
                state.set_parameter("Tw", operands[0:1])
                state.set_parameter("Tc", operands[1:2])
                state.add_tm([0, -state.leading])
                self._show_operand(operands, 2, runs)
            case "Td" | "Tm" | "TD" | "T*":
                state.reset_render()
                if operator == "Tm":
                    state.reset_text()
                elif operator == "TD" and len(operands) >= 2 and _is_number(operands[1]):
                    # A missing or non-numeric ty leaves the leading as it is.
                    state.leading = -operands[1]
                matrix = [0, -state.leading] if operator == "T*" else list(operands)
                state.add_tm(matrix)
            case "Tf":
                state.set_font(self.resolve_font(operands[0] if operands else ""), operands)
            case _:
                state.set_parameter(operator, operands)

    def _show_operand(self, operands: tuple[Any, ...], index: int, runs: list[_Run]) -> None:
        """Show the operand like Tj if it is a string; anything else shows no text."""
        value = operands[index] if index < len(operands) else None
        if isinstance(value, PdfString):
            runs.append(self.state.run(bytes(value.data)))


def _jsonable(value: object, depth: int = 0) -> Any:
    """A font for fonts.json: its public fields, with text keys, else its text."""
    if depth > 8:
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, depth + 1) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    else:
        fields = getattr(value, "__dict__", None) or {}
    if not fields:
        return str(value)
    return {
        key: _jsonable(item, depth + 1) for key, item in fields.items() if not key.startswith("_")
    }


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def text_show_operations(
    operations: Iterator[Operation],
    fonts: Mapping[str, Any],
    strip_rotated: bool,
    debug_path: Path | None = None,
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
            state.set_font(walker.resolve_font(operands[0] if operands else ""), operands)
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
    shifted = [
        {**group, "tx": group["tx"] - left, "displaced_tx": group["displaced_tx"] - left}
        for group in ordered
    ]
    if debug_path:
        debug_path.joinpath("bts.json").write_text(
            json.dumps(shifted, indent=2, default=str), "utf-8"
        )
        debug_path.joinpath("tjs.json").write_text(
            json.dumps(runs, indent=2, default=lambda x: getattr(x, "to_dict", lambda: str(x))()),
            "utf-8",
        )
    return shifted


def y_coordinate_groups(
    groups: list[dict[str, Any]], debug_path: Path | None = None
) -> dict[int, list[dict[str, Any]]]:
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
    if debug_path:
        debug_path.joinpath("bt_groups.json").write_text(
            json.dumps(by_line, indent=2, default=str), "utf-8"
        )
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
            # Text drawn with no width has no column to align to.
            needed = (int(tx // char_width) if char_width else length) - length
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
    debug_path: Path | None = None,
) -> str:
    """pypdf's _layout_mode_text. With debug_path, it writes pypdf's four debug files there;
    fonts.json and tjs.json hold the facade's view of the fonts and text runs.
    """
    if debug_path:
        debug_path.joinpath("fonts.json").write_text(
            json.dumps(_jsonable(dict(fonts)), indent=2, default=str), "utf-8"
        )
    groups = text_show_operations(operations, fonts, strip_rotated, debug_path)
    if not groups:
        return ""
    by_line = y_coordinate_groups(groups, debug_path)
    char_width = fixed_char_width(groups, scale_weight)
    return fixed_width_page(by_line, char_width, space_vertically, font_height_weight)
