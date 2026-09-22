from copy import replace
from types import SimpleNamespace

import numpy
import pytest

from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.render.model import RasterImage
from core_pdf.impl.runtime.execution import ExtractionScope
from core_pdf_ocr.impl.extract.contracts import (
    MAX_OCR_PIXELS,
    PRIMARY_OCR_PIXELS,
    OcrPass,
    OcrPassScope,
    PageRoute,
    WorkPlan,
)
from core_pdf_ocr.impl.extract.ocr import session
from core_pdf_ocr.impl.extract.ocr.types import (
    OcrTask,
    Raster,
    RasterRegion,
)
from core_pdf_ocr.impl.extract.quality import make_candidate


def ocr_pass(scope=OcrPassScope.PAGE, **kwargs):
    return OcrPass("test", scope, 1, (6,), region_first=False, **kwargs)


@pytest.fixture
def owner(monkeypatch, ocr_capture):
    monkeypatch.setattr(session, "compose_page", lambda *a, **k: object())
    capture = replace(ocr_capture, page=SimpleNamespace(width=600, height=800, media_box=None))
    return session.OcrSession(
        capture,
        WorkPlan(PageRoute.OCR, ocr_passes=(ocr_pass(),)),
        False,
        ExtractionScope(),
        None,
    )


def make_raster(value=255):
    return Raster(RasterImage(bytes([value]) * 10000, 100, 100, 1), 72)


@pytest.mark.parametrize("media_box", [None, (10, 20, 610, 820)])
def test_projection_cache_composes_each_declared_mode_once_and_preserves_page_box(
    monkeypatch, ocr_capture, media_box
) -> None:
    projections = {False: object(), True: object()}
    calls = []
    capture = replace(ocr_capture, page=SimpleNamespace(width=600, height=800, media_box=media_box))

    def compose(page, options, *, page_program, fields, annotations):
        assert page is capture.page
        assert page_program is capture.program
        assert fields is capture.fields
        assert annotations is capture.annotations
        calls.append(options.include_text)
        return projections[options.include_text]

    monkeypatch.setattr(session, "compose_page", compose)
    operations = (ocr_pass(), ocr_pass(), ocr_pass(include_native_text=True))
    result = session.OcrSession(
        capture, WorkPlan(PageRoute.OCR, ocr_passes=operations), False, ExtractionScope(), None
    )
    assert calls == [False, True]
    assert result.rendered_page(False) is projections[False]
    assert result.rendered_page(True) is projections[True]
    assert result.page_box == (media_box or (0, 0, 600, 800))
    assert calls == [False, True]


def test_undeclared_projection_fails_before_rendering(owner) -> None:
    with pytest.raises(ValueError, match="undeclared raster projection"):
        owner.rendered_page(True)


@pytest.mark.parametrize(
    ("vector", "height", "available", "adapted"),
    [
        (False, 6, True, True),
        (False, 9, True, False),
        (True, 8, True, True),
        (True, 10, True, False),
        (False, 6, False, False),
        (True, 6, False, False),
    ],
)
def test_preflight_adapts_only_small_but_legible_projected_text(
    owner, monkeypatch, vector, height, available, adapted
) -> None:
    samples = numpy.full((1000, 1000, 1), 255, dtype=numpy.uint8)
    for y in range(20, 950, 40):
        samples[y : y + height] = 0
    raster = Raster(RasterImage(samples.tobytes(), 1000, 1000, 1), 72)
    owner.capture = replace(
        owner.capture,
        evidence=replace(
            owner.capture.evidence,
            full_page_image=not vector,
            vector_complexity=100000 if vector else 0,
        ),
    )
    operation = ocr_pass(adaptive_scale=True, pixel_budget=PRIMARY_OCR_PIXELS)
    calls = []

    def direct(*args, **kwargs):
        calls.append("direct")
        assert not kwargs["upscale"]
        assert kwargs["max_pixels"] == 1000000
        return RasterRegion(raster, owner.page_box) if available else None

    def render(*args, **kwargs):
        calls.append("render")
        assert kwargs["max_pixels"] == 1000000
        return raster if available else None

    monkeypatch.setattr(session, "dominant_image_region", direct)
    monkeypatch.setattr(session, "rendered_page_raster", render)
    result = owner.adapt_pass(operation)
    assert calls == (["render"] if vector else ["direct"])
    if adapted:
        assert result.pixel_budget == MAX_OCR_PIXELS
        assert result.scale == pytest.approx(max(1.5, 32 / (height * numpy.sqrt(6))))
    else:
        assert result is operation


@pytest.mark.parametrize(
    ("direct", "available", "adaptive"),
    [(True, True, False), (False, True, False), (False, False, False), (True, True, True)],
)
def test_page_materialization_preserves_direct_crop_or_uses_declared_renderer(
    owner, monkeypatch, direct, available, adaptive
) -> None:
    raster = make_raster(0)
    box = (10, 20, 110, 120)
    calls = []
    monkeypatch.setattr(
        session,
        "dominant_image_region",
        lambda *a, **k: RasterRegion(raster, box) if direct else None,
    )

    def render(capture, scale, *, rendered, crop, max_pixels):
        calls.append("render")
        assert rendered is owner.rendered_page(False)
        assert crop is None
        return raster if available else None

    monkeypatch.setattr(session, "rendered_page_raster", render)
    operation = replace(ocr_pass(), name="adaptive-page" if adaptive else "page")
    result = owner.materialize(operation, selected=None, selected_tasks=())
    assert result is not None
    assert len(result.tasks) == int(available)
    assert calls == ([] if direct else ["render"])
    if result.tasks:
        assert result.tasks[0].page_box == (box if direct else owner.page_box)
        assert result.tasks[0].image.channels == 1
        if not adaptive:
            assert result.tasks[0].image is raster.image


@pytest.mark.parametrize("available", [False, True])
def test_weak_region_reuses_selected_tasks_for_high_resolution_retry(
    owner, monkeypatch, available
) -> None:
    raster = make_raster()
    task = OcrTask(6, raster.image, (0, 0, 100, 100), owner.page_box, 72)
    selected = make_candidate(6, ObservationBatch.empty())
    operation = ocr_pass(OcrPassScope.WEAK_REGIONS)

    def high_resolution(capture, tasks, requested, observations, *, rendered, compact_image):
        assert tasks == (task,)
        assert requested is operation
        assert observations is selected.observations
        assert rendered is owner.rendered_page(False)
        return (task,) if available else ()

    monkeypatch.setattr(session, "high_resolution_weak_region_tasks", high_resolution)
    result = owner.materialize(operation, selected=selected, selected_tasks=(task,))
    assert result is not None
    assert result.tasks == ((task,) if available else ())


@pytest.mark.parametrize("recovered", [False, True])
def test_timeout_recovery_retries_smaller_raster_without_recursing(
    owner, monkeypatch, recovered
) -> None:
    image = RasterImage(bytes([255]) * 4500000, 3000, 1500, 1)
    task = OcrTask(3, image, (0, 0, 3000, 1500), owner.page_box, 300)
    empty = make_candidate(3, ObservationBatch.empty(), recognition_status="timeout")
    recovered_text = ObservationBatch.from_columns(
        ("recovered",), ((0, 0, 10, 10),), source=1, confidence=(90,)
    )
    calls = []

    def recognize(group, *, raise_if_cancelled):
        raise_if_cancelled()
        calls.append(group)
        if len(calls) == 1:
            assert group == (task,)
            return (empty,)
        assert group[0].image.width * group[0].image.height <= 4000000
        assert group[0].page_box == task.page_box
        return (make_candidate(group[0].mode, recovered_text),) if recovered else (empty,)

    monkeypatch.setattr(session, "recognize_group", recognize)
    result = owner.recognize_tasks((task,))
    assert len(calls) == 2
    if recovered:
        assert result[0].recognition_status == "timeout-recovered"
        assert result[0].observations is recovered_text
    else:
        assert result[0] is empty


@pytest.mark.parametrize("packed_available", [False, True])
def test_stroked_materialization_preserves_packed_mapping_and_symbol_options(
    owner, monkeypatch, packed_available
) -> None:
    from core_pdf_ocr.impl.extract.ocr.types import PackedStrokedTextRaster

    raster = make_raster()
    packed = PackedStrokedTextRaster(raster, owner.page_box, ())
    monkeypatch.setattr(
        session,
        "stroked_vector_text_raster",
        lambda *a, **k: packed if packed_available else None,
    )
    monkeypatch.setattr(
        session,
        "full_stroked_vector_text_raster",
        lambda *a, **k: RasterRegion(raster, owner.page_box),
    )
    result = owner.materialize(
        ocr_pass(OcrPassScope.STROKED_VECTOR_TEXT), selected=None, selected_tasks=()
    )
    assert result is not None
    assert result.packed_stroked is (packed if packed_available else None)
    assert len(result.tasks) == 1
    assert result.tasks[0].recognize_words is packed_available
    assert result.tasks[0].collect_symbols is packed_available


@pytest.mark.parametrize("direct", [False, True])
def test_weak_native_seed_uses_available_direct_or_rendered_raster(
    owner, monkeypatch, direct
) -> None:
    raster = make_raster(0)
    monkeypatch.setattr(
        session,
        "dominant_image_region",
        lambda *a, **k: RasterRegion(raster, owner.page_box) if direct else None,
    )
    monkeypatch.setattr(session, "rendered_page_raster", lambda *a, **k: raster if direct else None)
    operation = ocr_pass(OcrPassScope.WEAK_REGIONS, seed_with_native=True)
    result = owner.materialize(operation, selected=None, selected_tasks=())
    assert result is not None
    assert bool(result.tasks) is direct


@pytest.mark.parametrize("regions_available", [False, True])
def test_image_regions_filter_blank_images_and_fallback_to_cropped_render(
    owner, monkeypatch, regions_available
) -> None:
    samples = numpy.full((100, 100, 1), 255, dtype=numpy.uint8)
    samples[:, ::2] = 0
    text_raster = Raster(RasterImage(samples.tobytes(), 100, 100, 1), 72)
    box = (10, 20, 110, 120)
    text = RasterRegion(text_raster, box)
    blank = RasterRegion(make_raster(), (200, 20, 300, 120))
    calls = []

    def regions(*args, **kwargs):
        assert kwargs["maximum_axis_deviation"] == 0.01
        return (text, blank) if regions_available else ()

    def render(capture, scale, *, crop, rendered, max_pixels):
        calls.append("render")
        assert scale == 2
        assert crop == box
        return text_raster

    monkeypatch.setattr(session, "page_image_regions", regions)
    monkeypatch.setattr(session, "safe_image_crop", lambda *a: box)
    monkeypatch.setattr(session, "rendered_page_raster", render)
    result = owner.materialize(
        ocr_pass(OcrPassScope.IMAGE_REGIONS), selected=None, selected_tasks=()
    )
    assert result is not None
    assert len(result.tasks) == 1
    assert result.tasks[0].image is text_raster.image
    assert result.tasks[0].page_box == box
    assert calls == ([] if regions_available else ["render"])


@pytest.mark.parametrize("distributed", [False, True])
def test_initial_region_materialization_selects_full_page_for_distributed_outlines(
    owner, monkeypatch, distributed
) -> None:
    from core_pdf_ocr.impl.extract.ocr.types import OcrRegion

    proposed = OcrRegion((10, 20, 110, 120), 1, ("image",))
    monkeypatch.setattr(session, "candidate_ocr_regions", lambda *a: (proposed,))
    monkeypatch.setattr(session, "has_distributed_outline_text", lambda *a: distributed)
    seen = []

    def tasks(capture, regions, operation, *, rendered, compact_image):
        seen.extend(regions)
        assert rendered is owner.rendered_page(False)
        return ()

    monkeypatch.setattr(session, "candidate_region_tasks", tasks)
    operation = replace(ocr_pass(), region_first=True)
    result = owner.materialize(operation, selected=None, selected_tasks=())
    assert result is not None
    assert result.tasks == ()
    assert len(seen) == 1
    assert seen[0].page_box == (owner.page_box if distributed else proposed.page_box)
    assert seen[0].reasons == (("distributed-outline-text",) if distributed else ("image",))
