import numpy
import pytest

from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.render.model import RasterImage
from core_pdf_ocr.impl.extract import grids
from core_pdf_ocr.impl.extract.ocr.types import OcrTask


@pytest.mark.parametrize("gap", [0, 1, 2, 3])
def test_row_gap_closing_fills_only_bounded_gaps_up_to_limit(gap: int) -> None:
    mask = numpy.asarray(
        [[False, True, False, True, False, False, True, False, False, False, True, False]]
    )
    expected = mask.copy()
    for left, right in ((1, 3), (3, 6), (6, 10)):
        if right - left - 1 <= gap:
            expected[0, left + 1 : right] = True
    numpy.testing.assert_array_equal(grids.close_row_gaps(mask, gap), expected)
    numpy.testing.assert_array_equal(
        mask, [[False, True, False, True, False, False, True, False, False, False, True, False]]
    )


def test_longest_runs_do_not_cross_row_boundaries() -> None:
    mask = numpy.asarray([[True, True, False, True], [True, True, True, True], [False] * 4])
    assert grids.longest_true_runs(mask).tolist() == [2, 4, 0]
    assert grids.longest_true_runs(numpy.zeros((3, 4), dtype=bool)).tolist() == [0, 0, 0]
    assert grids.cluster_line_positions(numpy.asarray([1, 2, 7, 20, 21])) == [3, 20]
    assert grids.cluster_line_positions(numpy.asarray([])) == []


def make_image(channels: int = 1) -> RasterImage:
    samples = numpy.full((300, 300, channels), 255, dtype=numpy.uint8)
    for position in (30, 90, 150, 210, 270):
        samples[position, 30:271] = 0
        samples[30:271, position] = 0
    return RasterImage(samples.tobytes(), 300, 300, channels)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_detected_grid_edges_map_back_from_pooling(channels: int) -> None:
    detected = grids.detect_ruling_grid(make_image(channels))
    assert detected is not None
    xs, ys, samples, slope = detected
    assert xs == ys == [31, 91, 151, 211, 271]
    assert samples.shape[:2] == (300, 300)
    assert slope == 0


@pytest.mark.parametrize(
    ("width", "height", "channels"), [(99, 300, 1), (300, 99, 1), (300, 300, 2), (300, 300, 1)]
)
def test_small_blank_and_unsupported_rasters_do_not_claim_a_grid(
    width: int, height: int, channels: int
) -> None:
    assert (
        grids.detect_ruling_grid(
            RasterImage(bytes([255]) * width * height * channels, width, height, channels)
        )
        is None
    )


def test_skew_estimation_and_shear_straighten_rulings() -> None:
    dark = numpy.zeros((100, 600), dtype=bool)
    for y in (20, 40, 60):
        dark[y, :200] = True
        dark[y + 4, -200:] = True
    assert grids.estimate_ruling_skew(dark) == pytest.approx(0.01)
    assert grids.estimate_ruling_skew(dark[:, :100]) == 0
    assert grids.estimate_ruling_skew(numpy.zeros_like(dark)) == 0
    source = numpy.arange(12).reshape(4, 3)
    numpy.testing.assert_array_equal(
        grids.vertical_shear(source, 1), [[0, 4, 8], [3, 7, 11], [6, 10, 11], [9, 10, 11]]
    )


def make_task() -> OcrTask:
    image = RasterImage(bytes([255]) * 10000, 100, 100, 1)
    return OcrTask(6, image, (0, 0, 100, 100), (10, 20, 210, 220), 80)


@pytest.mark.parametrize("channels", [1, 3])
def test_cell_tasks_inset_rules_skip_empty_cells_and_keep_page_transform(channels: int) -> None:
    task = make_task()
    shape = (100, 100) if channels == 1 else (100, 100, channels)
    samples = numpy.full(shape, 255, dtype=numpy.uint8)
    samples[5:10, 5:10] = 0
    result = grids.grid_cell_tasks(task, [0, 20, 40], [0, 20, 40], samples, 0)
    assert len(result) == 1
    cell = result[0]
    assert cell.mode == 7
    assert cell.rectangle == (2, 2, 16, 16)
    assert cell.image is task.image
    assert cell.page_box == task.page_box
    assert cell.minimum_confidence == 50
    assert grids.grid_region_page_box(task, [0, 20, 40], [0, 20, 40]) == (10, 140, 90, 220)


@pytest.mark.parametrize(
    ("xs", "ys"),
    [
        ([0, 5], [0, 20]),
        ([0, 20], [0, 5]),
        ([-40, -20], [0, 20]),
        (list(range(40)), list(range(40))),
    ],
)
def test_invalid_or_excessive_cells_do_not_schedule_ocr(xs: list[int], ys: list[int]) -> None:
    assert (
        grids.grid_cell_tasks(make_task(), xs, ys, numpy.zeros((100, 100), dtype=numpy.uint8), 0)
        == ()
    )


def test_row_observations_merge_in_reading_order_and_average_confidence() -> None:
    source = ObservationBatch.from_columns(
        (" right ", "lower", "left"),
        ((30, 80, 40, 90), (10, 20, 20, 30), (10, 80, 20, 90)),
        source=2,
        confidence=(80, 90, 100),
    )
    result = grids.grid_row_observations(source)
    assert result.text == ("left right", "lower")
    assert result.bbox.tolist() == [[10, 80, 40, 90], [10, 20, 20, 30]]
    assert result.confidence.tolist() == [90, 90]
    assert result.sequence.tolist() == [0, 1]
    empty = ObservationBatch.empty()
    assert grids.grid_row_observations(empty) is empty


def test_gap_closing_preserves_blank_full_and_zero_width_rows() -> None:
    for mask in (
        numpy.zeros((2, 10), dtype=bool),
        numpy.ones((2, 10), dtype=bool),
        numpy.zeros((2, 0), dtype=bool),
    ):
        numpy.testing.assert_array_equal(grids.close_row_gaps(mask, 20), mask)


@pytest.mark.parametrize(
    ("crossing", "count", "allowed"), [(0, 8, True), (2, 8, True), (3, 8, False), (7, 7, True)]
)
def test_regular_table_gate_limits_observations_crossing_column_rules(
    crossing: int, count: int, allowed: bool
) -> None:
    boxes = ((40, 150, 60, 155),) * crossing + ((12, 150, 20, 155),) * (count - crossing)
    prior = ObservationBatch.from_columns(
        ("word",) * (count + 1), (*boxes, (500, 500, 510, 510)), source=2
    )
    grid = (list(range(0, 101, 20)), list(range(0, 81, 10)), numpy.zeros((100, 100)), 0.0)
    assert grids.grid_is_regular_table(grid, prior, make_task()) is allowed
    assert grids.grid_is_regular_table(grid, ObservationBatch.empty(), make_task())


@pytest.mark.parametrize(
    ("xs", "ys"),
    [
        ([0, 20], list(range(9))),
        (list(range(6)), [0, 20]),
        (list(range(6)), [0] * 9),
        (list(range(6)), [0, 1, 2, 3, 4, 13, 22, 31, 40]),
    ],
)
def test_regular_table_gate_rejects_small_degenerate_and_irregular_grids(
    xs: list[int], ys: list[int]
) -> None:
    grid = (xs, ys, numpy.zeros((100, 100)), 0.0)
    assert not grids.grid_is_regular_table(grid, ObservationBatch.empty(), make_task())


def test_skew_requires_three_plausibly_paired_lines() -> None:
    dark = numpy.zeros((150, 600), dtype=bool)
    for y in (20, 60, 100):
        dark[y, :200] = True
        dark[y + 20, -200:] = True
    assert grids.estimate_ruling_skew(dark) == 0


def test_grid_detection_rejects_tightly_packed_rules() -> None:
    samples = numpy.full((300, 300), 255, dtype=numpy.uint8)
    for position in (30, 45, 60, 75):
        samples[position, 10:290] = 0
        samples[10:290, position] = 0
    assert grids.detect_ruling_grid(RasterImage(samples.tobytes(), 300, 300, 1)) is None
