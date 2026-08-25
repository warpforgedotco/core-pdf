# SPDX-License-Identifier: AGPL-3.0-only
"""Columnar page glyph storage behind a sequence-compatible boundary."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from core_pdf.impl.engine.layout.glyphs import GlyphObservation
from core_pdf.impl.exceptions import PdfContractError


class GlyphTable:
    """The page's glyph observations behind a tuple-like protocol.

    Consumers see a sequence of ``GlyphObservation`` rows: iteration, ``len``,
    truthiness, and integer indexing (including negative indexes, as the
    renderer's event payloads and the compat facades require). Slicing is
    deliberately unsupported — no consumer slices the page table.

    Row identity is stable for the lifetime of the table (materialize-once),
    so facades that hold rows and key maps by ``id()`` keep working.
    """

    __slots__ = ("internal_rows",)

    def __init__(self, rows: tuple[GlyphObservation, ...]) -> None:
        self.internal_rows = rows

    @classmethod
    def from_rows(cls, rows: Iterable[GlyphObservation], *, validate: bool = True) -> GlyphTable:
        materialized = tuple(rows)
        if validate and not all(
            isinstance(observation, GlyphObservation) for observation in materialized
        ):
            raise PdfContractError("page state emitted an invalid glyph product")
        return cls(materialized)

    def __iter__(self) -> Iterator[GlyphObservation]:
        return iter(self.internal_rows)

    def __len__(self) -> int:
        return len(self.internal_rows)

    def __bool__(self) -> bool:
        return bool(self.internal_rows)

    def __getitem__(self, index: int) -> GlyphObservation:
        if isinstance(index, slice):
            raise TypeError("GlyphTable does not support slicing")
        return self.internal_rows[index]


__all__ = ("GlyphTable",)
