from copy import replace

import pytest

from core_pdf.impl.extract_contracts import ObservationBatch
from core_pdf_ocr.impl.extract import table_detection
from core_pdf_ocr.impl.extract.contracts import ObservationSource


def make_observations(texts, boxes=None, source=ObservationSource.OCR):
    return ObservationBatch.from_columns(
        texts,
        boxes
        if boxes is not None
        else tuple((i * 50, 10, i * 50 + 40, 20) for i in range(len(texts))),
        source=source,
        confidence=(90,) * len(texts),
    )


def chart_capture(capture, area=20000):
    return replace(capture, evidence=replace(capture.evidence, uncovered_vector_area=area))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Year 2020 2021 2022", ("Year", "2020", "2021", "2022")),
        (" -1 +2 3.5 4% ", ("-1", "+2", "3.5", "4%")),
        ("1,000 1/2 3-4 label", ("1,000", "1/2", "3-4", "label")),
        ("2020 2021 2022", ("2020 2021 2022",)),
        ("A report for 2020", ("A report for 2020",)),
        ("", ("",)),
    ],
)
def test_dense_numeric_lines_split_but_prose_and_short_lines_stay_intact(text, expected) -> None:
    assert table_detection.chart_cell_texts(text) == expected


@pytest.mark.parametrize(
    ("area", "count", "source"),
    [
        (None, 3, ObservationSource.OCR),
        (19999, 3, ObservationSource.OCR),
        (20000, 2, ObservationSource.OCR),
        (20000, 3, ObservationSource.NATIVE),
    ],
)
def test_chart_requires_sufficient_uncovered_artwork_and_ocr_observations(
    ocr_capture, area, count, source
) -> None:
    capture = chart_capture(ocr_capture, area)
    observations = make_observations(("label",) * count, source=source)
    assert table_detection.extract_chart_table(capture, observations) is None


def test_chart_splits_numeric_line_into_equal_boxes_and_orders_rows_by_center(ocr_capture) -> None:
    observations = make_observations(
        ("Year 2020 2021 2022", "lower", "upper"),
        ((0, 40, 160, 50), (200, 0, 240, 10), (200, 40, 240, 50)),
    )
    result = table_detection.extract_chart_table(chart_capture(ocr_capture), observations)
    assert result is not None
    assert [[cell.text for cell in row] for row in result.rows] == [
        ["Year", "2020", "2021", "2022", "upper"],
        ["lower"],
    ]
    assert [cell.bbox for cell in result.rows[0][:4]] == [
        (0, 40, 40, 50),
        (40, 40, 80, 50),
        (80, 40, 120, 50),
        (120, 40, 160, 50),
    ]
    assert [[cell.column for cell in row] for row in result.rows] == [[0, 1, 2, 3, 5], [4]]
    assert all(cell.row == i for i, row in enumerate(result.rows) for cell in row)
    assert result.bbox == (0, 0, 240, 50)
    assert result.confidence == 0.35
    assert dict(result.metadata) == {"source": "chart-ocr", "synthetic": True}
    assert observations.text[0] == "Year 2020 2021 2022"


@pytest.mark.parametrize(
    ("overlap_x", "expected"),
    [(20, ("Value", "other", "VALUE")), (21, ("Value", "value", "other", "VALUE"))],
)
def test_chart_deduplicates_only_same_text_overlapping_at_least_half(
    ocr_capture, overlap_x, expected
) -> None:
    observations = make_observations(
        ("Value", "value", "other", "VALUE"),
        ((0, 0, 40, 10), (overlap_x, 0, overlap_x + 40, 10), (100, 0, 140, 10), (200, 0, 240, 10)),
    )
    result = table_detection.extract_chart_table(chart_capture(ocr_capture), observations)
    assert result is not None
    assert tuple(cell.text for cell in result.rows[0]) == expected


@pytest.mark.parametrize("texts", [("", " ", "valid"), ("same", "SAME", "Same")])
def test_chart_declines_when_filtering_leaves_fewer_than_three_cells(ocr_capture, texts) -> None:
    observations = make_observations(texts, ((0, 0, 40, 10),) * 3)
    assert table_detection.extract_chart_table(chart_capture(ocr_capture), observations) is None


def test_chart_skips_nonfinite_boxes_before_deduplication(ocr_capture) -> None:
    observations = make_observations(
        ("bad", "a", "b", "c"),
        ((float("nan"), 0, 10, 10), (0, 0, 10, 10), (20, 0, 30, 10), (40, 0, 50, 10)),
    )
    result = table_detection.extract_chart_table(chart_capture(ocr_capture), observations)
    assert result is not None
    assert tuple(cell.text for cell in result.rows[0]) == ("a", "b", "c")


def test_chart_row_grouping_uses_center_instead_of_top_edge(ocr_capture) -> None:
    observations = make_observations(
        ("tall", "short", "lower"), ((0, 0, 20, 100), (30, 48, 50, 52), (60, 20, 80, 30))
    )
    result = table_detection.extract_chart_table(chart_capture(ocr_capture), observations)
    assert result is not None
    assert [[cell.text for cell in row] for row in result.rows] == [["tall", "short"], ["lower"]]


@pytest.mark.parametrize("stroked", [False, True])
def test_trusted_vector_text_bypasses_table_detection(ocr_capture, stroked) -> None:
    evidence = replace(
        ocr_capture.evidence,
        vector_text_trusted=not stroked,
        stroked_vector_text=replace(ocr_capture.evidence.stroked_vector_text, trusted=stroked),
    )
    assert (
        table_detection.extract_tables(
            replace(ocr_capture, evidence=evidence), make_observations(("a", "b", "c"))
        )
        == ()
    )


def test_chart_table_passes_through_shared_finalization(ocr_capture) -> None:
    observations = make_observations(("a", "b", "c"))
    result = table_detection.extract_tables(chart_capture(ocr_capture), observations)
    charts = [table for table in result if table.metadata.get("source") == "chart-ocr"]
    assert len(charts) == 1
    assert charts[0].order == 0
    assert charts[0].row_bands
    assert charts[0].column_bands
    assert tuple(cell.text for cell in charts[0].rows[0]) == ("a", "b", "c")
