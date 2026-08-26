# SPDX-License-Identifier: AGPL-3.0-only
"""Columnar page glyph storage behind a sequence-compatible boundary.

The capture hot loop appends one compact row tuple per glyph plus one
:class:`GlyphSegment` per text-showing operation carrying the op-constant
fields, instead of constructing a full :class:`GlyphObservation` per glyph.
Row-level consumers (the renderer, compat facades, tests) still receive real
observations, materialized once per table on first row access; engine
extraction reads the per-field iterators and never materializes rows.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from typing import Any

from core_pdf.impl.engine.model.glyphs import (
    GlyphObservation,
    GlyphSegment,
    internal_GlyphEntry,
    internal_materialize,
)
from core_pdf.impl.exceptions import PdfContractError


class GlyphTableBuilder:
    """Mutable capture-time glyph storage owned by one ``TextState``.

    The hot loop appends row tuples directly through ``rows.append``; the
    slower construction paths append prebuilt observations. The ActualText
    branch uses ``__len__`` as a mark with ``extract_rows``/``truncate``, and
    nested capture states (tiling patterns, annotation appearances) iterate
    materialized rows.
    """

    __slots__ = ("rows",)

    def __init__(self) -> None:
        self.rows: list[internal_GlyphEntry] = []

    def append_row(self, observation: GlyphObservation) -> None:
        self.rows.append(observation)

    def __len__(self) -> int:
        return len(self.rows)

    def __bool__(self) -> bool:
        return bool(self.rows)

    def __iter__(self) -> Iterator[GlyphObservation]:
        return (internal_materialize(entry) for entry in self.rows)

    def extract_rows(self, start: int) -> list[GlyphObservation]:
        return [internal_materialize(entry) for entry in self.rows[start:]]

    def truncate(self, start: int) -> None:
        del self.rows[start:]

    def build(self) -> GlyphTable:
        return GlyphTable(tuple(self.rows))


class GlyphTable:
    """The page's glyph observations behind a tuple-like protocol.

    Consumers see a sequence of ``GlyphObservation`` rows: iteration, ``len``,
    truthiness, and integer indexing (including negative indexes, as the
    renderer's event payloads and the compat facades require). Slicing is
    deliberately unsupported — no consumer slices the page table.

    Row identity is stable for the lifetime of the table (materialize-once),
    so facades that hold rows and key maps by ``id()`` keep working. Page
    programs are cached per page and pages are interpreted on pool workers,
    so the first materialization is guarded: without it two workers could
    each build a row tuple and the surviving identity would be whichever
    stored last. The engine extraction pipeline reads the ``iter_*`` field
    iterators instead and never triggers materialization.
    """

    __slots__ = ("internal_entries", "internal_materialized", "internal_lock")

    def __init__(self, entries: tuple[internal_GlyphEntry, ...]) -> None:
        self.internal_entries = entries
        self.internal_materialized: tuple[GlyphObservation, ...] | None = None
        self.internal_lock = threading.Lock()

    @classmethod
    def from_rows(cls, rows: Iterable[GlyphObservation], *, validate: bool = True) -> GlyphTable:
        materialized = tuple(rows)
        if validate and not all(
            isinstance(observation, GlyphObservation) for observation in materialized
        ):
            raise PdfContractError("page state emitted an invalid glyph product")
        table = cls(materialized)
        table.internal_materialized = materialized
        return table

    def internal_rows(self) -> tuple[GlyphObservation, ...]:
        # Double-checked: once materialized every later access is a single
        # attribute read, so the renderer's per-glyph indexing pays no lock.
        materialized = self.internal_materialized
        if materialized is None:
            with self.internal_lock:
                materialized = self.internal_materialized
                if materialized is None:
                    materialized = tuple(
                        internal_materialize(entry) for entry in self.internal_entries
                    )
                    self.internal_materialized = materialized
        return materialized

    def __iter__(self) -> Iterator[GlyphObservation]:
        return iter(self.internal_rows())

    def __len__(self) -> int:
        return len(self.internal_entries)

    def __bool__(self) -> bool:
        return bool(self.internal_entries)

    def __getitem__(self, index: int) -> GlyphObservation:
        if isinstance(index, slice):
            raise TypeError("GlyphTable does not support slicing")
        return self.internal_rows()[index]

    def iter_event_rows(self) -> Iterator[tuple[int, int, Any, bool, bool]]:
        """Yield ``(index, seqno, ink_bbox, visible, has_paint)`` per glyph.

        ``has_paint`` replicates ``GlyphObservation.has_paint`` over columns;
        capture-time rows never carry an eager bitmap, so that property term
        is statically false for them.
        """
        for index, entry in enumerate(self.internal_entries):
            if isinstance(entry, GlyphObservation):
                yield index, entry.seqno, entry.ink_bbox, entry.visible, entry.has_paint
                continue
            segment: GlyphSegment = entry[0]
            decoder = segment.font_decoder
            has_paint = decoder is not None and (
                entry[16] is not None or (entry[15] is not None and entry[13] > 0 and entry[14] > 0)
            )
            yield index, segment.seqno, entry[2], entry[9], has_paint

    def iter_font_names(self) -> Iterator[tuple[int, str | None]]:
        """Yield ``(seqno, font_name)`` per glyph without materializing rows."""
        for entry in self.internal_entries:
            if isinstance(entry, GlyphObservation):
                yield entry.seqno, entry.font_name
            else:
                segment: GlyphSegment = entry[0]
                yield segment.seqno, segment.font_name

    def iter_evidence_rows(
        self,
    ) -> Iterator[tuple[str, bool, object, bytes, str, float | None]]:
        """Yield ``(text, visible, font_decoder, code_bytes, unicode_source, confidence)``."""
        for entry in self.internal_entries:
            if isinstance(entry, GlyphObservation):
                yield (
                    entry.text,
                    entry.visible,
                    entry.font_decoder,
                    entry.code_bytes,
                    entry.unicode_source,
                    entry.confidence,
                )
            else:
                yield (
                    entry[1],
                    entry[9],
                    entry[0].font_decoder,
                    entry[5],
                    entry[11],
                    entry[10],
                )


__all__ = (
    "GlyphTable",
    "GlyphTableBuilder",
)
