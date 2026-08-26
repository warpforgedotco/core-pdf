# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable results of line-text reconstruction.

These are records with no behavior. They live in the model layer rather than with
the heuristics that build them because ``TextRun`` memoizes reconstruction results
on itself, and the record layer must not name types from ``layout/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True, slots=True)
class LayoutLineTextSegment:
    text: str
    separator_before: str
    advance_bbox: tuple[float, float, float, float]
    rotation_angle: int


@dataclass(frozen=True, slots=True)
class LayoutLineText:
    text: str
    segments: tuple[LayoutLineTextSegment, ...]


EMPTY_LAYOUT_LINE_TEXT = LayoutLineText("", ())


class LayoutWordSnapshot(NamedTuple):
    text: str
    bbox: tuple[float, float, float, float]
