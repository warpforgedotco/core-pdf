from copy import replace

import numpy
import pytest

from core_pdf.impl._impl.extract.contracts import ObservationBatch
from core_pdf.impl._impl.render.model import RasterImage
from core_pdf_ocr.impl.extract.contracts import OcrPass, OcrPassScope
from core_pdf_ocr.impl.extract.ocr import rescue
from core_pdf_ocr.impl.extract.ocr.types import internal_OcrTask
from core_pdf_ocr.impl.extract.quality import internal_candidate, internal_text_utility_stats


def internal_observations(text="abcdefgh", box=(0, 0, 100, 100)) -> ObservationBatch:
    return ObservationBatch.from_columns((text,), (box,), source=1, confidence=(100,))


def test_coverage_grid_distributes_utility_by_clipped_area() -> None:
    observations = internal_observations(box=(-100, -100, 100, 100))
    grid = rescue.internal_observation_coverage_grid(observations, (0, 0, 100, 100), 2, 2)
    assert grid.tolist() == [2, 2, 2, 2]
    assert grid.sum() == internal_text_utility_stats("abcdefgh", 100).utility
    upper_right = internal_observations(box=(50, 50, 100, 100))
    assert rescue.internal_observation_coverage_grid(
        upper_right, (0, 0, 100, 100), 2, 2
    ).tolist() == [0, 8, 0, 0]


@pytest.mark.parametrize(
    ("text", "box"),
    [("", (0, 0, 10, 10)), ("word", (200, 200, 300, 300)), ("word", (10, 0, 10, 10))],
)
def test_coverage_grid_ignores_empty_off_page_and_degenerate_observations(text, box) -> None:
    result = rescue.internal_observation_coverage_grid(
        internal_observations(text, box), (0, 0, 100, 100), 2, 2
    )
    assert result.tolist() == [0] * 4
    assert (
        rescue.internal_observation_coverage_grid(
            ObservationBatch.empty(), (0, 0, 100, 100), 2, 2
        ).tolist()
        == [0] * 4
    )


def test_rescue_coverage_deduplicates_same_raster_and_counts_only_unexplained_ink() -> None:
    samples = numpy.full((100, 100, 1), 255, dtype=numpy.uint8)
    samples[:50] = 0
    task = internal_OcrTask(
        6, RasterImage(samples.tobytes(), 100, 100, 1), (0, 0, 100, 100), (0, 0, 100, 100), 72
    )
    operation = OcrPass("rescue", OcrPassScope.WEAK_REGIONS, 1, (6, 11), tiles=2, region_columns=2)
    observations = internal_observations(box=(0, 50, 50, 100))
    coverage = rescue.internal_adaptive_rescue_coverage(
        (task, replace(task, mode=11)), operation, observations
    )
    assert coverage == rescue.internal_RescueCoverage(1, 4, 2, 1)
    assert coverage.mean_ink == 0.5
    assert coverage.weak_ink_ratio == 0.5
    empty = rescue.internal_adaptive_rescue_coverage((), operation, observations)
    assert empty.mean_ink == empty.weak_ink_ratio == 0


@pytest.mark.parametrize(
    ("characters", "confidence", "height", "ink", "weak", "rasters", "expected"),
    [
        (31, 95, 32, 0, 0, 0, False),
        (32, 95, 32, 0, 0, 0, True),
        (31, 94, 32, 0, 0, 0, True),
        (31, 95, 31, 0, 0, 0, True),
        (1000, 95, 10, 0.85, 0.85, 1, False),
        (999, 95, 10, 0.85, 0.85, 1, True),
        (2000, 92, 10, 0.85, 0.85, 1, False),
        (1999, 92, 10, 0.85, 0.85, 1, True),
        (2000, 91, 10, 0.85, 0.85, 1, True),
        (1000, 95, 10, 0.84, 0.84, 1, True),
        (300, 95, 10, 0.5, 0, 1, False),
        (299, 95, 10, 0.5, 0, 1, True),
        (300, 95, 10, 0.5, 0.01, 1, True),
        (600, 95, 10, 0.5, 0.01, 1, False),
        (600, 95, 10, 0.5, 0.015, 1, True),
        (600, 95, 10, 0.5, 0, 0, True),
    ],
)
def test_rescue_decision_boundaries(
    monkeypatch: pytest.MonkeyPatch, characters, confidence, height, ink, weak, rasters, expected
) -> None:
    candidate = internal_candidate(6, internal_observations())
    candidate = replace(
        candidate,
        metrics=replace(
            candidate.metrics,
            characters=characters,
            mean_confidence=confidence,
            median_text_height=height,
        ),
    )
    operation = OcrPass("primary", OcrPassScope.PAGE, 1, (6,))

    def coverage(tasks, coverage_pass, observations):
        assert tasks == ()
        assert observations is candidate.observations
        assert coverage_pass.scope is OcrPassScope.WEAK_REGIONS
        assert (coverage_pass.tiles, coverage_pass.region_columns, coverage_pass.max_regions) == (
            6,
            3,
            8,
        )
        return rescue.internal_RescueCoverage(rasters, 1, ink, weak)

    monkeypatch.setattr(rescue, "internal_adaptive_rescue_coverage", coverage)
    assert rescue.internal_adaptive_rescue_decision(candidate, (), operation) is expected
