# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from copy import replace

from core_pdf.impl.extract_contracts import ObservationBatch, PageAnalysis
from core_pdf.impl.extract_grids import (
    axis_segments,
    grid_components,
    merge_collinear_segments,
    split_grid_component,
    table_from_component,
)
from core_pdf.impl.extract_table_cleanup import (
    annotate_table_associations,
    merge_adjacent_tables,
    split_semantic_table,
    table_character_spaced_prose,
    table_is_single_column_prose,
    table_with_bands,
)
from core_pdf.impl.extract_table_core import (
    TableAnalysis,
    TableCandidate,
    TableContext,
    TableRule,
    TableSource,
    TableTransform,
)
from core_pdf.impl.extract_table_stream import StreamTableSource
from core_pdf.impl.output_model import Table
from core_pdf.impl.types import GeneratedRecord

TableGate = Callable[[PageAnalysis], bool]


def table_vertical_sort_key(table: Table) -> float:
    return -(table.bbox or (0.0, 0.0, 0.0, 0.0))[3]


class RuledTableSource:
    __slots__ = ()

    def detect(self, context: TableContext, start_order: int) -> tuple[TableCandidate, ...]:
        observations = context.analysis.observations
        horizontal, vertical = axis_segments(context.capture)
        horizontal = merge_collinear_segments(horizontal, coordinate=2, start=0, end=1)
        vertical = merge_collinear_segments(vertical, coordinate=0, start=1, end=2)
        found: list[TableCandidate] = []
        for component in grid_components(horizontal, vertical):
            for component_part in split_grid_component(*component):
                table = table_from_component(
                    start_order + len(found),
                    *component_part,
                    observations,
                )
                if table is not None:
                    found.append(TableCandidate(table))
        return tuple(found)

    def admit(
        self, accepted: list[TableCandidate], found: tuple[TableCandidate, ...]
    ) -> list[TableCandidate]:
        return [*accepted, *found]


class MergeAdjacentTables:
    __slots__ = ()

    def __call__(self, candidates: list[TableCandidate]) -> list[TableCandidate]:
        by_table = {id(candidate.table): candidate for candidate in candidates}
        return [
            by_table.get(id(table)) or TableCandidate(table)
            for table in merge_adjacent_tables([candidate.table for candidate in candidates])
        ]


class SplitSemanticSections:
    __slots__ = ()

    def __call__(self, candidates: list[TableCandidate]) -> list[TableCandidate]:
        return [segment for candidate in candidates for segment in split_semantic_table(candidate)]


CHARACTER_SPACED_PROSE = TableRule("character-spaced-prose", table_character_spaced_prose)
SINGLE_COLUMN_PROSE = TableRule("single-column-prose", table_is_single_column_prose)


class TableDetector(GeneratedRecord):
    gate: TableGate | None
    sources: tuple[TableSource, ...]
    transforms: tuple[TableTransform, ...] = ()
    filters: tuple[TableRule, ...] = ()
    supplements: tuple[TableSource, ...] = ()

    @classmethod
    def native(cls) -> TableDetector:
        return cls(
            None,
            (RuledTableSource(), StreamTableSource()),
            (MergeAdjacentTables(), SplitSemanticSections()),
            (CHARACTER_SPACED_PROSE, SINGLE_COLUMN_PROSE),
        )

    def detect(self, context: TableContext) -> tuple[Table, ...]:
        accepted: list[TableCandidate] = []
        for source in self.sources:
            accepted = source.admit(accepted, source.detect(context, len(accepted)))
        for transform in self.transforms:
            accepted = transform(accepted)
        filters = self.filters
        accepted = [
            candidate
            for candidate in accepted
            if not any(rule.rejects(candidate) for rule in filters)
        ]
        for supplement in self.supplements:
            accepted = supplement.admit(accepted, supplement.detect(context, len(accepted)))
        return tuple(candidate.table for candidate in accepted)

    def extract(self, capture: PageAnalysis, observations: ObservationBatch) -> tuple[Table, ...]:
        if self.gate is not None and not self.gate(capture):
            return ()
        analysis = TableAnalysis.build(observations, capture.width)
        return finalize_tables(self.detect(TableContext(capture, analysis)), analysis)


NATIVE_TABLES = TableDetector.native()


def extract_tables(capture: PageAnalysis, observations: ObservationBatch) -> tuple[Table, ...]:
    return NATIVE_TABLES.extract(capture, observations)


def finalize_tables(
    tables: tuple[Table, ...],
    analysis: TableAnalysis,
) -> tuple[Table, ...]:
    observations = analysis.observations
    tables = tuple(sorted(tables, key=table_vertical_sort_key))
    return tuple(
        table_with_bands(
            annotate_table_associations(
                replace(table, order=order) if table.order != order else table,
                observations,
                analysis.text_rows,
            )
        )
        for order, table in enumerate(tables)
    )
