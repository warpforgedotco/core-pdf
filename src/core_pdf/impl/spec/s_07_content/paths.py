# SPDX-License-Identifier: AGPL-3.0-only
"""Unflattened PDF path construction commands."""

from __future__ import annotations

from dataclasses import dataclass, field

from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix


@dataclass(frozen=True, slots=True)
class PathCommand:
    operator: str
    operands: tuple[float, ...]
    ctm: Matrix = IDENTITY_MATRIX
    flatness: float = 0.0


@dataclass(slots=True)
class PdfPath:
    commands: list[PathCommand] = field(default_factory=list)

    def clear(self) -> None:
        self.commands.clear()

    def move_to(self, x: float, y: float) -> None:
        self.commands.append(PathCommand("m", (x, y)))

    def line_to(self, x: float, y: float) -> None:
        self.commands.append(PathCommand("l", (x, y)))

    def close(self) -> None:
        self.commands.append(PathCommand("h", ()))

    def rect(self, x: float, y: float, w: float, h: float) -> None:
        self.commands.append(PathCommand("re", (x, y, w, h)))

    def cubic_to(self, points: tuple[float, ...], ctm: Matrix, flatness: float) -> None:
        self.commands.append(PathCommand("c", points, ctm, flatness))
