# SPDX-License-Identifier: AGPL-3.0-only
"""Single coordinator for table detection, annotation, and projection inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace

from core_pdf.impl.extract.contracts import CapturedPage, ObservationBatch
from core_pdf.impl.extract.table_cleanup import (
    internal_annotate_table_associations,
    internal_table_with_bands,
)
from core_pdf.impl.extract.table_detection import (
    extract_chart_table,
    internal_detect_tables,
    internal_text_rows,
)
from core_pdf.impl.output import Table


@dataclass(frozen=True, slots=True)
class internal_TableExtractor:
    """Derive every table representation from one captured observation batch."""

    capture: CapturedPage
    observations: ObservationBatch

    def extract(self) -> tuple[Table, ...]:
        evidence = self.capture.evidence
        if evidence.vector_text_trusted or evidence.stroked_vector_text.trusted:
            return ()
        text_rows = internal_text_rows(self.observations)
        tables = internal_detect_tables(
            self.capture,
            self.observations,
            text_rows=text_rows,
        )
        chart_table = extract_chart_table(self.capture, self.observations)
        if chart_table is not None:
            tables = (*tables, chart_table)
        if not tables:
            return ()
        return tuple(
            internal_table_with_bands(
                internal_annotate_table_associations(
                    replace(table, order=order) if table.order != order else table,
                    self.observations,
                    text_rows,
                )
            )
            for order, table in enumerate(tables)
        )


def extract_tables(capture: CapturedPage, observations: ObservationBatch) -> tuple[Table, ...]:
    return internal_TableExtractor(capture, observations).extract()
