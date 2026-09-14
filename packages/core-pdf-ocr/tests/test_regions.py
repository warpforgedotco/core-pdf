"""Deterministic OCR crop selection from capture geometry."""

from dataclasses import replace

import pytest

from core_pdf.impl._impl.capture.program import CapturedProgram, PageProgram
from core_pdf.impl._impl.capture.records import CapturedDrawing, CapturedLine
from core_pdf.impl._impl.extract.contracts import ObservationBatch
from core_pdf_ocr.impl.extract.contracts import OcrPass, OcrPassScope, PageAnalysis
from core_pdf_ocr.impl.extract.ocr import regions
from core_pdf_ocr.impl.extract.ocr.types import internal_OcrRegion
from core_pdf_spec.types import PdfName


def internal_capture(capture: PageAnalysis, drawings: tuple[CapturedDrawing, ...]) -> PageAnalysis:
    return replace(capture, program=PageProgram(CapturedProgram(drawings=drawings)))


def internal_image(box: tuple[float, float, float, float], pixels: int = 64) -> CapturedDrawing:
    x0, y0, x1, y1 = box
    return CapturedDrawing(
        0,
        None,
        None,
        kind="image",
        bbox=box,
        items=(("quad", ((x0, y0), (x1, y0), (x0, y1), (x1, y1))),),
        raw_data=bytes([127]) * pixels * pixels,
        dictionary={
            "Width": pixels,
            "Height": pixels,
            "BitsPerComponent": 8,
            "ColorSpace": PdfName(b"DeviceGray"),
        },
    )


def test_image_region_decodes_original_raster_with_page_mapping(ocr_capture: PageAnalysis) -> None:
    capture = internal_capture(ocr_capture, (internal_image((0, 0, 600, 800)),))
    result = regions.internal_page_image_regions(capture, minimum_area_ratio=0.65, upscale=False)
    assert len(result) == 1
    assert result[0].page_box == (0, 0, 600, 800)
    assert result[0].raster.width == result[0].raster.height == 64
    assert bytes(result[0].raster.image.pixels) == bytes([127]) * 4096
    assert regions.internal_dominant_image_region(capture, upscale=False) is not None


@pytest.mark.parametrize(
    "drawing",
    [
        CapturedDrawing(0, None, None),
        replace(internal_image((0, 0, 600, 800)), items=()),
        replace(internal_image((0, 0, 600, 800)), bbox=None),
        internal_image((-20, 0, 600, 800)),
        internal_image((0, 0, 10, 10)),
        replace(internal_image((0, 0, 600, 800)), raw_data=None),
    ],
)
def test_unsuitable_direct_images_fall_back_to_compositing(
    ocr_capture: PageAnalysis,
    drawing: CapturedDrawing,
) -> None:
    capture = internal_capture(ocr_capture, (drawing,))
    assert (
        regions.internal_page_image_regions(capture, minimum_area_ratio=0.65, upscale=False) == ()
    )


def test_tiny_or_overlapping_dominant_images_are_not_selected(ocr_capture: PageAnalysis) -> None:
    tiny = internal_capture(ocr_capture, (internal_image((0, 0, 600, 800), pixels=32),))
    assert regions.internal_dominant_image_region(tiny, upscale=False) is None
    duplicate = internal_capture(ocr_capture, (internal_image((0, 0, 600, 800)),) * 2)
    assert regions.internal_dominant_image_region(duplicate, upscale=False) is None


def test_distinct_large_images_select_highest_pixel_resolution(ocr_capture: PageAnalysis) -> None:
    capture = internal_capture(
        ocr_capture,
        (
            internal_image((0, 0, 600, 600), pixels=64),
            internal_image((0, 200, 600, 800), pixels=80),
        ),
    )
    result = regions.internal_dominant_image_region(capture, upscale=False)
    assert result is not None
    assert result.page_box == (0, 200, 600, 800)
    assert result.raster.width == 80


def test_region_merge_preserves_reasons_and_is_order_independent() -> None:
    inputs = [
        internal_OcrRegion((0, 0, 10, 10), 5, ("image",)),
        internal_OcrRegion((5, 0, 15, 10), 3, ("image", "vector")),
        internal_OcrRegion((30, 30, 40, 40), 2, ("grid",)),
        internal_OcrRegion((0, 0, 0, 0), 1, ("empty",)),
    ]
    expected = (internal_OcrRegion((0, 0, 15, 10), 5.45, ("image", "vector")), inputs[2], inputs[3])
    assert regions.internal_merge_ocr_regions(inputs) == expected
    assert regions.internal_merge_ocr_regions(inputs[::-1]) == expected
    assert regions.internal_merge_ocr_regions([]) == ()


@pytest.mark.parametrize(("limit", "expected"), [(3, 8), (12, 12)])
def test_region_batch_honors_minimum_and_requested_count(limit: int, expected: int) -> None:
    candidates = tuple(internal_OcrRegion((i * 2, 0, i * 2 + 1, 1), 1, ()) for i in range(20))
    plan = OcrPass("regions", OcrPassScope.WEAK_REGIONS, 1, (6,), max_regions=limit)
    assert (
        regions.internal_ocr_region_batch(candidates, plan, page_area=1000) == candidates[:expected]
    )


def test_region_batch_keeps_first_large_region_but_skips_later_over_budget() -> None:
    plan = OcrPass("regions", OcrPassScope.WEAK_REGIONS, 1, (6,))
    large = internal_OcrRegion((0, 0, 100, 100), 5, ())
    small = internal_OcrRegion((0, 0, 1, 1), 1, ())
    assert regions.internal_ocr_region_batch((large, small), plan, page_area=100) == (large,)
    assert regions.internal_ocr_region_batch((), plan, page_area=0) == ()


def test_empty_page_has_explicit_page_fallback(ocr_capture: PageAnalysis) -> None:
    assert regions.internal_candidate_ocr_regions(ocr_capture) == (
        internal_OcrRegion((0, 0, 600, 800), 0, ("page-fallback",)),
    )


def test_image_candidates_are_padded_and_clipped(ocr_capture: PageAnalysis) -> None:
    capture = replace(
        ocr_capture, evidence=replace(ocr_capture.evidence, image_boxes=((0, 10, 100, 100),))
    )
    assert regions.internal_candidate_ocr_regions(capture) == (
        internal_OcrRegion((0, 4, 106, 106), 5, ("image",)),
    )


def test_native_text_suppresses_covered_vector_candidates(ocr_capture: PageAnalysis) -> None:
    drawing = CapturedDrawing(0, None, None, kind="fill", bbox=(210, 210, 240, 240))
    capture = internal_capture(ocr_capture, (drawing,))
    candidates = regions.internal_candidate_ocr_regions(capture)
    assert any("uncovered-vector" in c.reasons for c in candidates)
    native = ObservationBatch.from_columns(
        ("readable native text",), ((200, 200, 299, 299),), source=0
    )
    covered = replace(capture, observations=native)
    assert regions.internal_candidate_ocr_regions(covered)[0].reasons == ("page-fallback",)


@pytest.mark.parametrize(
    ("count", "distributed", "expected"),
    [(199, True, False), (200, False, False), (200, True, True)],
)
def test_outline_detection_requires_many_small_paths_distributed_across_page(
    ocr_capture: PageAnalysis,
    count: int,
    distributed: bool,
    expected: bool,
) -> None:
    drawings = tuple(
        CapturedDrawing(
            i,
            None,
            None,
            kind="fill",
            bbox=(
                (i % 20) * (25 if distributed else 1),
                (i // 20) * (65 if distributed else 1),
                (i % 20) * (25 if distributed else 1) + 5,
                (i // 20) * (65 if distributed else 1) + 5,
            ),
        )
        for i in range(count)
    )
    assert (
        regions.internal_has_distributed_outline_text(internal_capture(ocr_capture, drawings))
        is expected
    )


def test_grid_geometry_yields_grid_and_label_regions(ocr_capture: PageAnalysis) -> None:
    lines = tuple(
        [CapturedLine(20, y, 100, y) for y in (20, 60, 100)]
        + [CapturedLine(x, 20, x, 100) for x in (20, 60, 100)]
    )
    capture = replace(ocr_capture, program=PageProgram(CapturedProgram(lines=lines)))
    selected = regions.internal_candidate_ocr_regions(capture)
    reasons = {reason for region in selected for reason in region.reasons}
    assert {"grid", "grid-labels", "vector-density"} <= reasons
    assert all(0 <= r.page_box[0] < r.page_box[2] <= 600 for r in selected)


@pytest.mark.parametrize("large", [False, True])
def test_dense_vector_pages_propose_fine_label_regions(
    ocr_capture: PageAnalysis, large: bool
) -> None:
    box = (0, 0, 300, 300) if large else (210, 210, 220, 220)
    drawings = (
        CapturedDrawing(0, None, None, kind="stroke", bbox=box),
        CapturedDrawing(1, None, None, kind="image"),
        CapturedDrawing(2, None, None, kind="fill"),
    )
    capture = internal_capture(ocr_capture, drawings)
    capture = replace(capture, evidence=replace(capture.evidence, vector_complexity=180))
    selected = regions.internal_candidate_ocr_regions(capture)
    reasons = {reason for region in selected for reason in region.reasons}
    assert "vector-label-density" in reasons
    assert ("vector-label-neighborhood" in reasons) is not large
    assert all(0 <= r.page_box[1] < r.page_box[3] <= 800 for r in selected)


def test_header_density_keeps_moderate_text_but_body_density_does_not(
    ocr_capture: PageAnalysis,
) -> None:
    def select(y: float) -> tuple[internal_OcrRegion, ...]:
        capture = internal_capture(
            ocr_capture, (CapturedDrawing(0, None, None, kind="stroke", bbox=(10, y, 20, y + 10)),)
        )
        native = ObservationBatch.from_columns(("twelve chars",), ((10, y, 20, y + 10),), source=0)
        return regions.internal_candidate_ocr_regions(replace(capture, observations=native))

    assert "header-band" in select(10)[0].reasons
    assert "sparse-label" not in select(10)[0].reasons
    assert select(210)[0].reasons == ("page-fallback",)


def test_degenerate_large_and_offpage_shapes_do_not_create_uncovered_vector_regions(
    ocr_capture: PageAnalysis,
) -> None:
    drawings = tuple(
        CapturedDrawing(i, None, None, kind="fill", bbox=box)
        for i, box in enumerate(
            (
                (0, 0, 0, 10),
                (0, 0, 600, 800),
                (-500, -500, -450, -450),
            )
        )
    )
    capture = internal_capture(ocr_capture, drawings)
    capture = replace(
        capture, evidence=replace(capture.evidence, image_boxes=((-500, -500, -450, -450),))
    )
    selected = regions.internal_candidate_ocr_regions(capture)
    assert all("uncovered-vector" not in r.reasons for r in selected)
    assert all("image" not in r.reasons for r in selected)


def test_large_grid_is_not_mistaken_for_a_compact_label_region(ocr_capture: PageAnalysis) -> None:
    lines = tuple(
        [CapturedLine(0, y, 600, y) for y in (0, 400, 800)]
        + [CapturedLine(x, 0, x, 800) for x in (0, 300, 600)]
    )
    capture = replace(ocr_capture, program=PageProgram(CapturedProgram(lines=lines)))
    selected = regions.internal_candidate_ocr_regions(capture)
    assert all("grid-labels" not in r.reasons for r in selected)


def test_fine_regions_reject_components_fully_outside_page(ocr_capture: PageAnalysis) -> None:
    capture = internal_capture(
        ocr_capture, (CapturedDrawing(0, None, None, kind="stroke", bbox=(-100, -100, -90, -90)),)
    )
    capture = replace(capture, evidence=replace(capture.evidence, vector_complexity=180))
    assert all(
        "vector-label-density" not in r.reasons
        for r in regions.internal_candidate_ocr_regions(capture)
    )
