from copy import replace
from types import SimpleNamespace
from typing import Any

import numpy
import pytest

from core_pdf.impl.execution import ExtractionScope
from core_pdf.impl.extract_contracts import ObservationBatch, PageFrame, PageState
from core_pdf.impl.extract_pipeline import (
    NATIVE_PIPELINE,
    DetectTables,
    LayoutBlocks,
    PagePipeline,
)


class Mark:
    def __init__(self, name: str, log: list[str] | None = None) -> None:
        self.name = name
        self.log = [] if log is None else log

    def __call__(self, state: PageState, *_args: Any) -> PageState:
        self.log.append(self.name)
        return replace(state, order_ambiguous=not state.order_ambiguous)


def test_the_native_pipeline_detects_tables_before_laying_out_blocks() -> None:
    assert [type(stage) for stage in NATIVE_PIPELINE.stages] == [DetectTables, LayoutBlocks]


def test_stages_are_replaced_and_inserted_around_anchors() -> None:
    first, second, extra = Mark("first"), Mark("second"), Mark("extra")
    pipeline = PagePipeline((first, second))
    assert pipeline.inserting_before(second, extra).stages == (first, extra, second)
    assert pipeline.inserting_after(Mark, extra).stages == (first, extra, second)
    assert pipeline.replacing(second, extra).stages == (first, extra)
    with pytest.raises(ValueError, match="no stage"):
        pipeline.replacing(extra, first)


def test_running_checks_cancellation_then_threads_state_through_stages() -> None:
    extraction: Any = SimpleNamespace(
        capture=SimpleNamespace(observations=ObservationBatch.empty())
    )
    log: list[str] = []
    pipeline = PagePipeline((Mark("first", log), Mark("second", log)))
    state = pipeline.run(extraction, ExtractionScope())
    assert log == ["first", "second"]
    assert state.observations is extraction.capture.observations
    assert state.order_ambiguous is False
    with pytest.raises(Exception, match="cancel"):
        pipeline.run(extraction, ExtractionScope(lambda: True))
    assert log == ["first", "second"]


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        (0, [[10.0, 20.0, 30.0, 60.0]]),
        (90, [[20.0, 70.0, 60.0, 90.0]]),
        (180, [[70.0, 140.0, 90.0, 180.0]]),
        (270, [[140.0, 10.0, 180.0, 30.0]]),
        (450, [[20.0, 70.0, 60.0, 90.0]]),
    ],
)
def test_frames_map_boxes_into_display_space(rotation: int, expected: list[list[float]]) -> None:
    frame = PageFrame(100.0, 200.0, rotation, 1)
    boxes = numpy.asarray([[10.0, 20.0, 30.0, 60.0]], dtype=numpy.float32)
    displayed = frame.display_boxes(boxes)
    assert displayed.tolist() == expected
    assert displayed.dtype == numpy.float32


@pytest.mark.parametrize(
    ("rotation", "expected"), [(0, 20.0), (90, 40.0), (180, -20.0), (270, -40.0), (-90, -40.0)]
)
def test_reading_axis_positions_follow_the_text_rotation(rotation: int, expected: float) -> None:
    boxes = numpy.asarray([[10.0, 20.0, 30.0, 60.0]])
    assert PageFrame.reading_axis_positions(boxes, rotation).tolist() == [expected]
