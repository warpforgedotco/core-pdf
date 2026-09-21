from dataclasses import replace

import numpy
import pytest

from core_pdf.impl._impl.extract.contracts import ObservationBatch
from core_pdf.impl._impl.render.model import RasterImage
from core_pdf_ocr.impl.extract.contracts import OcrPass, OcrPassScope
from core_pdf_ocr.impl.extract.ocr import region_tasks
from core_pdf_ocr.impl.extract.ocr.types import internal_Raster


def internal_raster(samples: numpy.ndarray) -> internal_Raster:
    height, width, channels = samples.shape
    return internal_Raster(RasterImage(samples.tobytes(), width, height, channels), 72)


def internal_pass() -> OcrPass:
    return OcrPass("test", OcrPassScope.TILES, 1, (6, 11), tiles=2)


def test_task_groups_preserve_order_identity_modes_and_batch_limits() -> None:
    source = internal_raster(numpy.zeros((10, 10, 1), dtype=numpy.uint8))
    tasks = region_tasks.internal_tile_tasks(source, (0, 0, 10, 10), internal_pass())
    groups = region_tasks.internal_ocr_task_groups(tasks)
    assert tuple(map(len, groups)) == (2, 2)
    assert tuple(task for group in groups for task in group) == tasks
    assert all(task.image is source.image for task in tasks)
    assert region_tasks.internal_ocr_task_groups(()) == ()
    repeated = (tasks[0],) * 17
    assert tuple(map(len, region_tasks.internal_ocr_task_groups(repeated))) == (16, 1)
    other = replace(tasks[0], image=RasterImage(bytes(100), 10, 10, 1))
    assert tuple(map(len, region_tasks.internal_ocr_task_groups((tasks[0], other)))) == (1, 1)
    large = replace(tasks[0], rectangle=(0, 0, 2000, 2000))
    assert tuple(map(len, region_tasks.internal_ocr_task_groups((large,) * 3))) == (2, 1)


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_text_band_height_tracks_visible_ink(channels: int) -> None:
    samples = numpy.full((120, 192, channels), 255, dtype=numpy.uint8)
    color_channels = 1 if channels == 2 else min(channels, 3)
    for y in (10, 30, 50, 70):
        samples[y : y + 6, :, :color_channels] = 0
        samples[y + 2, :, :color_channels] = 255
    assert region_tasks.internal_estimated_text_height(internal_raster(samples)) == 6
    if channels in (2, 4):
        samples[:, :, -1] = 0
        assert region_tasks.internal_estimated_text_height(internal_raster(samples)) == 0


@pytest.mark.parametrize(
    ("height", "width", "value"), [(20, 20, 255), (20, 3, 0), (2, 20, 0), (120, 192, 0)]
)
def test_text_band_estimator_rejects_blank_narrow_or_solid_images(
    height: int, width: int, value: int
) -> None:
    source = internal_raster(numpy.full((height, width, 1), value, dtype=numpy.uint8))
    expected = 2 if height == 2 else 0
    assert region_tasks.internal_estimated_text_height(source) == expected


def test_text_band_sampling_reports_original_pixel_height() -> None:
    samples = numpy.full((1200, 1200, 1), 255, dtype=numpy.uint8)
    for y in range(40, 1100, 80):
        samples[y : y + 12] = 0
    assert region_tasks.internal_estimated_text_height(internal_raster(samples)) == 12


def test_binary_tiles_share_preprocessed_image_and_recognition_options() -> None:
    source = internal_raster(numpy.full((100, 20, 4), 255, dtype=numpy.uint8))
    operation = replace(
        internal_pass(),
        preprocess="binary-clean",
        minimum_confidence=37,
        character_confidence_threshold=52,
        recognize_words=True,
        collect_symbols=True,
    )
    tasks = region_tasks.internal_tile_tasks(
        source, (10, 20, 30, 120), operation, compact_image=True
    )
    assert [task.rectangle for task in tasks] == [(0, 0, 20, 75), (0, 25, 20, 75)] * 2
    assert all(task.image is tasks[0].image for task in tasks)
    assert tasks[0].image.channels == 1
    assert tasks[0].page_box == (10, 20, 30, 120)
    assert tasks[0].minimum_confidence == 37
    assert tasks[0].character_confidence_threshold == 52
    assert tasks[0].recognize_words
    assert tasks[0].collect_symbols


def test_weak_regions_select_only_ink_and_expand_within_raster_bounds() -> None:
    samples = numpy.full((100, 100, 1), 255, dtype=numpy.uint8)
    samples[:50, :50] = 0
    source = internal_raster(samples)
    operation = replace(internal_pass(), max_regions=1)
    empty = ObservationBatch.empty()
    assert region_tasks.internal_weak_region_rectangles(
        source, (0, 0, 100, 100), operation, empty
    ) == ((0, 0, 74, 74),)
    tasks = region_tasks.internal_weak_region_tasks(source, (0, 0, 100, 100), operation, empty)
    assert len(tasks) == 2
    assert [task.mode for task in tasks] == [6, 11]
    blank = internal_raster(numpy.full((100, 100, 1), 255, dtype=numpy.uint8))
    assert region_tasks.internal_weak_region_tasks(blank, (0, 0, 100, 100), operation, empty) == ()


def test_utility_grid_maps_page_origin_and_edges_without_counting_outside_text() -> None:
    boxes = ((10, 100, 10, 100), (110, 0, 110, 0), (200, 0, 210, 10))
    observations = ObservationBatch.from_columns(
        ("hello", "hello", "outside"), boxes, source=1, confidence=(90, 90, 90)
    )
    grid = region_tasks.internal_observation_utility_grid(observations, (10, 0, 110, 100), 2, 2)
    assert grid[0] > 0
    assert grid[0] == grid[3]
    assert grid[1] == grid[2] == 0
    assert (
        region_tasks.internal_observation_utility_grid(
            observations, (300, 0, 400, 100), 2, 2
        ).tolist()
        == [0] * 4
    )


def test_dense_primary_text_reduces_rescue_grid_and_region_budget() -> None:
    primary = ObservationBatch.from_columns(
        ("recognized",) * 40, ((0, 0, 10, 10),) * 40, source=1, confidence=(90,) * 40
    )
    source = internal_raster(numpy.zeros((600, 300, 1), dtype=numpy.uint8))
    operation = replace(internal_pass(), tiles=12, region_columns=6, max_regions=30)
    assert region_tasks.internal_weak_region_grid_shape(source, operation, primary) == (6, 3)
    rectangles = region_tasks.internal_weak_region_rectangles(
        source, (0, 0, 300, 600), operation, primary
    )
    assert len(rectangles) == 8
    assert all(rectangle[:2] != (0, 452) for rectangle in rectangles)


@pytest.mark.parametrize("available", [False, True])
def test_high_resolution_rescue_deduplicates_source_modes_and_maps_crop(
    monkeypatch: pytest.MonkeyPatch, ocr_capture, available: bool
) -> None:
    samples = numpy.full((100, 100, 1), 255, dtype=numpy.uint8)
    samples[:50, :50] = 0
    source = internal_raster(samples)
    operation = replace(internal_pass(), max_regions=1)
    source_tasks = region_tasks.internal_tile_tasks(source, (10, 20, 210, 220), operation)
    crops = []

    def render(capture, scale, *, crop, rendered, max_pixels):
        assert capture is ocr_capture
        assert scale == operation.scale
        assert max_pixels == operation.pixel_budget
        crops.append(crop)
        return source if available else None

    monkeypatch.setattr(region_tasks, "internal_rendered_page_raster", render)
    result = region_tasks.internal_high_resolution_weak_region_tasks(
        ocr_capture,
        source_tasks,
        operation,
        ObservationBatch.empty(),
        rendered=None,
        compact_image=False,
    )
    assert crops == [(10, 72, 158, 220)]
    assert len(result) == (2 if available else 0)
    if result:
        assert all(task.page_box == crops[0] for task in result)
        assert all(task.rectangle == (0, 0, 100, 100) for task in result)
    assert (
        region_tasks.internal_high_resolution_weak_region_tasks(
            ocr_capture,
            (),
            operation,
            ObservationBatch.empty(),
            rendered=None,
            compact_image=False,
        )
        == ()
    )


@pytest.mark.parametrize("layered", [False, True])
def test_candidate_regions_use_direct_pixels_unless_overlaid_images_need_compositing(
    monkeypatch: pytest.MonkeyPatch, ocr_capture, layered: bool
) -> None:
    from core_pdf_ocr.impl.extract.ocr.types import internal_OcrRegion, internal_RasterRegion

    source = internal_raster(numpy.zeros((100, 100, 1), dtype=numpy.uint8))
    rendered_source = internal_raster(numpy.full((100, 100, 1), 255, dtype=numpy.uint8))
    box = (0, 0, 100, 100)
    direct = internal_RasterRegion(source, box)
    monkeypatch.setattr(
        region_tasks,
        "internal_page_image_regions",
        lambda *a, **k: (direct,) * (2 if layered else 1),
    )
    calls = []

    def render(*args, **kwargs):
        calls.append(kwargs["crop"])
        return rendered_source

    monkeypatch.setattr(region_tasks, "internal_rendered_page_raster", render)
    result = region_tasks.internal_candidate_region_tasks(
        ocr_capture,
        (internal_OcrRegion(box, 1, ("test",)),),
        internal_pass(),
        rendered=None,
        compact_image=False,
    )
    assert len(result) == 2
    assert all(
        task.image is (rendered_source.image if layered else source.image) for task in result
    )
    assert all(task.recognize_words is layered for task in result)
    assert calls == ([box] if layered else [])


@pytest.mark.parametrize(
    ("enabled", "characters", "images", "full_page", "expected"),
    [
        (False, 0, 1, True, False),
        (True, 10, 1, False, True),
        (True, 0, 0, False, True),
        (True, 0, 1, True, True),
        (True, 1, 1, True, False),
        (True, 0, 1, False, False),
    ],
)
def test_direct_scan_routing_respects_opt_out_and_native_page_evidence(
    ocr_capture, enabled: bool, characters: int, images: int, full_page: bool, expected: bool
) -> None:
    from core_pdf_ocr.impl.extract.contracts import PageRoute, WorkPlan

    capture = replace(
        ocr_capture,
        evidence=replace(
            ocr_capture.evidence,
            visible_native_characters=characters,
            image_count=images,
            full_page_image=full_page,
        ),
    )
    plan = WorkPlan(PageRoute.OCR, allow_direct_image_ocr=enabled)
    assert region_tasks.internal_direct_scan_allowed(capture, plan) is expected


@pytest.mark.parametrize(
    ("dominant", "render_available"), [(True, False), (False, True), (False, False)]
)
def test_candidate_region_falls_back_to_dominant_scan_or_renderer(
    monkeypatch: pytest.MonkeyPatch, ocr_capture, dominant: bool, render_available: bool
) -> None:
    from core_pdf_ocr.impl.extract.ocr.types import internal_OcrRegion, internal_RasterRegion

    source = internal_raster(numpy.zeros((100, 100, 1), dtype=numpy.uint8))
    box = (0, 0, 100, 100)
    calls = []

    def fallback(capture, *, max_pixels):
        assert capture is ocr_capture
        assert max_pixels == operation.pixel_budget
        calls.append("dominant")
        return internal_RasterRegion(source, box) if dominant else None

    def render(capture, scale, *, crop, rendered, max_pixels):
        assert crop == box
        assert max_pixels == operation.pixel_budget
        calls.append("render")
        return source if render_available else None

    operation = internal_pass()
    monkeypatch.setattr(region_tasks, "internal_page_image_regions", lambda *a, **k: ())
    monkeypatch.setattr(region_tasks, "internal_dominant_image_region", fallback)
    monkeypatch.setattr(region_tasks, "internal_rendered_page_raster", render)
    tasks = region_tasks.internal_candidate_region_tasks(
        ocr_capture,
        (internal_OcrRegion(box, 1, ("test",)),),
        operation,
        rendered=None,
        compact_image=False,
    )
    assert len(tasks) == (2 if dominant or render_available else 0)
    assert calls == (["dominant"] if dominant else ["dominant", "render"])
    if tasks:
        assert all(task.image is source.image for task in tasks)
