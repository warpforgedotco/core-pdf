# SPDX-License-Identifier: AGPL-3.0-only
"""Single coordinator for table detection, annotation, and projection inputs."""

from __future__ import annotations

from core_pdf.impl.extract.contracts import ObservationBatch, PageAnalysis
from core_pdf.impl.extract.table_detection import internal_TableAnalysis
from core_pdf.impl.output import Table


def extract_tables(capture: PageAnalysis, observations: ObservationBatch) -> tuple[Table, ...]:
    return internal_TableAnalysis.build(observations, capture.width).extract(capture)
