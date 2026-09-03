# SPDX-License-Identifier: AGPL-3.0-only
"""Stage-by-stage throughput of the extraction pipeline.

``ExtractionCache`` memoizes every stage on the page, so a benchmark that drove
the pipeline through the cache would time a dictionary read. Each benchmark here
instead calls one stage function directly on inputs the fixture built once. The
stages are the seams ``extract.pipeline`` itself uses, in pipeline order:

    capture_page -> plan_page -> extract_tables -> layout_blocks_with_evidence

``test_extract_document_benchmark`` closes the file over the whole path,
including the parse the stage benchmarks deliberately exclude, so a regression
that moves between stages still shows up somewhere.

Every fixture is native-text only. ``assert_native_only`` enforces that, because
an OCR pass would put Tesseract inside the measurement -- see
``tests.helpers.benchmark_pages``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, NamedTuple

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.extract.block_layout import layout_blocks_with_evidence
from core_pdf.impl.extract.capture import capture_page
from core_pdf.impl.extract.observations import plan_page
from core_pdf.impl.extract.tables import extract_tables
from tests.helpers.benchmark_pages import (
    DENSE_PDF,
    MIXED_PDF,
    assert_native_only,
    opened_page,
)
from tests.helpers.paths import require_fixture


class Staged(NamedTuple):
    """One page carried through the pipeline, with each stage's input kept."""

    page: Any
    capture: Any
    plan: Any
    observations: Any
    obstacles: tuple[tuple[float, float, float, float], ...]


def internal_image_obstacles(capture: Any) -> tuple[tuple[float, float, float, float], ...]:
    """Mid-sized images only, matching ``ExtractionCache.internal_image_obstacles``."""
    evidence = capture.evidence
    return tuple(
        box
        for box in evidence.image_boxes
        if 0.01 <= ((box[2] - box[0]) * (box[3] - box[1])) / evidence.page_area < 0.65
    )


def internal_stage(page: Any) -> Staged:
    capture = capture_page(page)
    plan = plan_page(capture)
    assert_native_only(plan)
    # With no OCR passes ``fuse_observations`` returns the native batch
    # unchanged, so the capture's own observations are what the later stages
    # would receive from the cache.
    observations = capture.observations
    tables = extract_tables(capture, observations)
    obstacles = (
        *(table.bbox for table in tables if table.bbox is not None),
        *internal_image_obstacles(capture),
    )
    return Staged(page, capture, plan, observations, obstacles)


@pytest.fixture(scope="module")
def dense() -> Iterator[Staged]:
    with opened_page(DENSE_PDF) as page:
        yield internal_stage(page)


def test_capture_page_benchmark(benchmark, dense: Staged) -> None:
    """Deriving routing evidence and observations from an interpreted program."""
    capture = benchmark(capture_page, dense.page)

    assert len(capture.observations) > 0
    assert capture.evidence.page_area > 0.0


def test_plan_page_benchmark(benchmark, dense: Staged) -> None:
    """The routing decision. Cheap, but it gates every page."""
    plan = benchmark(plan_page, dense.capture)

    assert not plan.ocr_passes


def test_extract_tables_benchmark(benchmark, dense: Staged) -> None:
    """Ruled-grid detection and cell text assignment on a line-rich page."""
    tables = benchmark(extract_tables, dense.capture, dense.observations)

    assert isinstance(tables, tuple)


def test_layout_blocks_benchmark(benchmark, dense: Staged) -> None:
    """Reading-order reconstruction: the XY-cut and its block assembly."""
    blocks, evidence = benchmark(
        layout_blocks_with_evidence,
        dense.observations,
        obstacles=dense.obstacles,
        use_xy_cut=True,
        rotation=0,
        page_width=float(dense.capture.page.width),
        page_height=float(dense.capture.page.height),
    )

    assert blocks
    assert evidence is not None


def internal_extract_document(path: Any) -> Any:
    with PdfDocument.open(path) as document:
        return document.extract()


def test_extract_document_benchmark(benchmark) -> None:
    """Open and extract a whole document, the path a caller actually takes.

    A fresh document per round on purpose: ``PdfDocument.extract`` caches its
    result per page selection, so reusing one would measure the cache.
    """
    path = require_fixture(MIXED_PDF)

    document = benchmark(internal_extract_document, path)

    assert len(document.pages) == 1
    assert document.pages[0].blocks
