from __future__ import annotations

import pytest
from ocr_test_helpers.extract_fakes import capture as make_capture
from ocr_test_helpers.extract_fakes import page_evidence, text_run

from core_pdf.impl._impl.extract import table_detection
from core_pdf.impl._impl.model.runs import TextRun
from core_pdf.impl.spec.s_07_content.capture import CapturedLine
from core_pdf_ocr.impl.extract.contracts import (
    ObservationBatch,
    ObservationSource,
)
from core_pdf_ocr.impl.extract.table_detection import (
    extract_tables,
)

EIGHT_COLUMNS = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0)


def line(x0: float, y0: float, x1: float, y1: float) -> CapturedLine:
    return CapturedLine(x0, y0, x1, y1, 0.5)


def text(value: str, x: float, y: float, sequence: int) -> TextRun:
    return text_run(value, x, y, x + 5.0, y + 5.0, seqno=sequence)


RULED_GRID = (
    line(10.0, 90.0, 90.0, 90.0),
    line(10.0, 50.0, 90.0, 50.0),
    line(10.0, 10.0, 90.0, 10.0),
    line(10.0, 10.0, 10.0, 90.0),
    line(50.0, 10.0, 50.0, 90.0),
    line(90.0, 10.0, 90.0, 90.0),
)


def observations(runs: tuple[TextRun, ...]) -> ObservationBatch:
    return ObservationBatch.from_columns(
        (run.text for run in runs),
        ((run.x0, run.y0, run.x1, run.y1) for run in runs),
        source=ObservationSource.NATIVE,
        sequence=(run.seqno for run in runs),
        visible=(run.visible for run in runs),
        references=runs,
    )


def test_extract_tables_orders_a_synthetic_chart_above_a_ruled_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = (
        text("top-left", 20.0, 70.0, 0),
        text("top-right", 60.0, 70.0, 1),
        text("bottom-left", 20.0, 30.0, 2),
        text("bottom-right", 60.0, 30.0, 3),
    )
    ocr = ObservationBatch.from_columns(
        ("chart-a", "chart-b", "chart-c"),
        (
            (10.0, 300.0, 25.0, 310.0),
            (35.0, 300.0, 50.0, 310.0),
            (60.0, 300.0, 75.0, 310.0),
        ),
        source=ObservationSource.OCR,
    )
    batch = ObservationBatch.concatenate(observations(runs), ocr)
    capture = make_capture(
        page_evidence(page_area=40_000.0, uncovered_vector_area=25_000.0),
        runs=runs,
        grid_lines=RULED_GRID,
        width=100.0,
        height=400.0,
        batch=batch,
    )
    # Keep this focused on the ruled-grid and chart paths; otherwise the same
    # lower observations also form a redundant whitespace-inferred table.
    monkeypatch.setattr(
        table_detection,
        "internal_stream_tables",
        lambda capture, start_order, analysis: (),
    )

    tables = extract_tables(capture, batch)

    assert [table.bbox for table in tables] == [
        (10.0, 300.0, 75.0, 310.0),
        (10.0, 10.0, 90.0, 90.0),
    ]
    assert [table.order for table in tables] == [0, 1]
    assert tables[0].metadata["source"] == "chart-ocr"
    assert [[cell.text for cell in row] for row in tables[1].rows] == [
        ["top-left", "top-right"],
        ["bottom-left", "bottom-right"],
    ]
