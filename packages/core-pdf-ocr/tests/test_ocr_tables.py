from __future__ import annotations

import pytest
from ocr_test_helpers.extract_fakes import capture as make_capture
from ocr_test_helpers.extract_fakes import page_evidence, text_run

from core_pdf.impl._impl.extract import table_detection
from core_pdf.impl._impl.model.runs import TextRun
from core_pdf.impl._impl.runtime.execution import ExtractionScope
from core_pdf.impl.spec.s_07_content.capture import CapturedLine
from core_pdf_ocr import PdfDocument
from core_pdf_ocr.impl.extract.contracts import (
    ObservationBatch,
    ObservationSource,
    PageAnalysis,
    RecognitionResult,
    WorkPlan,
)
from core_pdf_ocr.impl.extract.ocr import pipeline as recognition_pipeline
from core_pdf_ocr.impl.extract.table_detection import (
    extract_chart_table,
    extract_tables,
)
from tests.helpers.pdf_bytes import one_page_pdf

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


def test_chart_preserves_repeated_values_and_labels_at_distinct_positions() -> None:
    batch = ObservationBatch.from_columns(
        ("Year", "10", "10", "20", "Year"),
        (
            (0.0, 10.0, 30.0, 20.0),
            (40.0, 10.0, 60.0, 20.0),
            (70.0, 10.0, 90.0, 20.0),
            (100.0, 10.0, 120.0, 20.0),
            (0.0, 40.0, 30.0, 50.0),
        ),
        source=ObservationSource.OCR,
    )
    capture = make_capture(page_evidence(uncovered_vector_area=25_000.0), batch=batch)

    table = extract_chart_table(capture, batch)

    assert table is not None
    assert [[cell.text for cell in row] for row in table.rows] == [
        ["Year"],
        ["Year", "10", "10", "20"],
    ]
    assert [cell.bbox for cell in table.rows[1][1:3]] == [
        (40.0, 10.0, 60.0, 20.0),
        (70.0, 10.0, 90.0, 20.0),
    ]


@pytest.mark.parametrize("offset", [0.0, 1.0, 5.0])
def test_chart_deduplicates_matching_overlapping_observations(offset: float) -> None:
    batch = ObservationBatch.from_columns(
        ("Year", "YEAR", "10", "20"),
        (
            (0.0, 10.0, 10.0, 20.0),
            (offset, 10.0, 10.0 + offset, 20.0),
            (20.0, 10.0, 30.0, 20.0),
            (40.0, 10.0, 50.0, 20.0),
        ),
        source=ObservationSource.OCR,
    )
    capture = make_capture(page_evidence(uncovered_vector_area=25_000.0), batch=batch)

    table = extract_chart_table(capture, batch)

    assert table is not None
    assert [[cell.text for cell in row] for row in table.rows] == [["Year", "10", "20"]]


def test_chart_preserves_repeated_values_with_only_incidental_overlap() -> None:
    batch = ObservationBatch.from_columns(
        ("Year", "10", "10", "20"),
        (
            (0.0, 10.0, 10.0, 20.0),
            (20.0, 10.0, 30.0, 20.0),
            (29.0, 10.0, 39.0, 20.0),
            (40.0, 10.0, 50.0, 20.0),
        ),
        source=ObservationSource.OCR,
    )
    capture = make_capture(page_evidence(uncovered_vector_area=25_000.0), batch=batch)

    table = extract_chart_table(capture, batch)

    assert table is not None
    assert [[cell.text for cell in row] for row in table.rows] == [["Year", "10", "10", "20"]]


@pytest.mark.parametrize(
    "invalid",
    [
        (0.0, 10.0, 0.0, 20.0),
        (0.0, 20.0, 20.0, 10.0),
        (0.0, 10.0, float("nan"), 20.0),
        (float("nan"), 10.0, 20.0, 20.0),
        (0.0, 10.0, float("inf"), 20.0),
    ],
    ids=["zero-width", "reversed", "nan-end", "nan-start", "infinite"],
)
def test_invalid_chart_box_does_not_suppress_a_later_valid_value(
    invalid: tuple[float, float, float, float],
) -> None:
    batch = ObservationBatch.from_columns(
        ("10", "Year", "10", "20"),
        (invalid, (10.0, 10.0, 20.0, 20.0), (30.0, 10.0, 40.0, 20.0), (50.0, 10.0, 60.0, 20.0)),
        source=ObservationSource.OCR,
    )
    capture = make_capture(page_evidence(uncovered_vector_area=25_000.0), batch=batch)

    table = extract_chart_table(capture, batch)

    assert table is not None
    assert [[cell.text for cell in row] for row in table.rows] == [["Year", "10", "20"]]
    assert table.bbox == (10.0, 10.0, 60.0, 20.0)


def test_chart_dense_numeric_line_preserves_repeated_tokens() -> None:
    batch = ObservationBatch.from_columns(
        ("Year 10 10 20", "Series", "Units"),
        ((0.0, 10.0, 80.0, 20.0), (90.0, 10.0, 120.0, 20.0), (130.0, 10.0, 160.0, 20.0)),
        source=ObservationSource.OCR,
    )
    capture = make_capture(page_evidence(uncovered_vector_area=25_000.0), batch=batch)

    table = extract_chart_table(capture, batch)

    assert table is not None
    assert [[cell.text for cell in row] for row in table.rows] == [
        ["Year", "10", "10", "20", "Series", "Units"]
    ]


def test_document_extraction_preserves_repeated_recognized_chart_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artwork = b"\n".join(
        f"{10 + column * 15} {20 + row * 15} 11 11 re f".encode()
        for row in range(9)
        for column in range(20)
    )
    pdf = one_page_pdf(
        b"BT /F1 10 Tf 10 380 Td (Chart) Tj ET\n" + artwork,
        media_box=(0, 0, 400, 400),
    )
    recognized = ObservationBatch.from_columns(
        ("Year", "10", "10", "20", "Month", "30", "40", "50"),
        (
            (10.0, 300.0, 40.0, 310.0),
            (70.0, 300.0, 90.0, 310.0),
            (130.0, 300.0, 150.0, 310.0),
            (190.0, 300.0, 210.0, 310.0),
            (10.0, 270.0, 40.0, 280.0),
            (70.0, 270.0, 90.0, 280.0),
            (130.0, 270.0, 150.0, 280.0),
            (190.0, 270.0, 210.0, 280.0),
        ),
        source=ObservationSource.OCR,
        confidence=(99.0,) * 8,
    )
    calls = 0

    def recognize(
        capture: PageAnalysis, plan: WorkPlan, context: ExtractionScope, **kwargs: object
    ) -> RecognitionResult:
        nonlocal calls
        calls += 1
        assert (capture.evidence.uncovered_vector_area or 0.0) >= 20_000.0
        context.raise_if_cancelled()
        return RecognitionResult(recognized)

    monkeypatch.setattr(recognition_pipeline, "recognize_page", recognize)
    with PdfDocument(pdf) as document:
        result = document.extract()

    # Poppler 26.07.0 verifies both 10s at distinct x=70 and x=130 positions
    # in /private/tmp/core-pdf-recognized-chart-verification/reference.pdf.
    # A real stream table may replace the equivalent synthetic chart. Its
    # cells describe column regions containing the recognized text rectangles.
    tables = result.pages[0].tables
    assert calls == 1
    assert len(tables) == 1
    assert [[cell.text for cell in row] for row in tables[0].rows] == [
        ["Year", "10", "10", "20"],
        ["Month", "30", "40", "50"],
    ]
    assert tables[0].bbox == (10.0, 270.0, 210.0, 310.0)
    first_box, second_box = (cell.bbox for cell in tables[0].rows[0][1:3])
    assert first_box is not None
    assert second_box is not None
    assert first_box[0] <= 70.0 < 90.0 <= first_box[2]
    assert second_box[0] <= 130.0 < 150.0 <= second_box[2]
    assert first_box[2] <= second_box[0]
    assert first_box[1::2] == second_box[1::2] == (300.0, 310.0)
