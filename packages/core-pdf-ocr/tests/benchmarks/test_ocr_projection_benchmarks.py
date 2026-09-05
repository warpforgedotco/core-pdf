# SPDX-License-Identifier: AGPL-3.0-only
"""Recognition table policy over real native layout and table candidates."""

import pytest

from core_pdf.impl._impl.output.model import Block, Table
from core_pdf_ocr.impl.extract.table_reconcile import internal_project_text_and_tables
from tests.helpers.benchmark_pages import projection_inputs


@pytest.fixture(scope="module")
def projection() -> tuple[list[Block], tuple[Table, ...]]:
    return projection_inputs()


def test_ocr_table_projection_benchmark(benchmark, projection) -> None:
    blocks, tables = benchmark(internal_project_text_and_tables, *projection)

    assert blocks
    assert tables
