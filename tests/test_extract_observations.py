from __future__ import annotations

from core_pdf.impl.extract.contracts import (
    FusionPolicy,
    ObservationBatch,
    ObservationSource,
    OcrPass,
    OcrPassScope,
    PagePlanReason,
    PageRoute,
    WorkPlan,
)
from core_pdf.impl.extract.observations import fuse_observations
from core_pdf.impl.layout.spatial import maximum_candidate_coverage
from tests.helpers import extract_fakes


def observations(
    *items: tuple[str, float, float],
    source: ObservationSource,
) -> ObservationBatch:
    """Observations from ``(text, confidence, x0)`` triples on one 10pt-high line."""
    return extract_fakes.observations(
        ((text, (x0, 0.0, x0 + 10.0, 10.0)) for text, internal_confidence, x0 in items),
        source=source,
        confidence=tuple(confidence for internal_text, confidence, internal_x0 in items),
    )


def test_image_supplement_requires_high_confidence_informative_text() -> None:
    native = observations(("native", 100.0, 0.0), source=ObservationSource.NATIVE)
    ocr = observations(
        ("~", 99.0, 20.0),
        ("weak text", 70.0, 40.0),
        ("useful text", 90.0, 60.0),
        source=ObservationSource.OCR,
    )
    plan = WorkPlan(
        PageRoute.HYBRID,
        ocr_passes=(OcrPass("images", OcrPassScope.IMAGE_REGIONS, 1.0, (11,)),),
    )

    fused = fuse_observations(native, ocr, plan)

    assert list(fused.text) == ["native", "useful text"]


def test_primary_ocr_augmentation_keeps_moderate_confidence_text() -> None:
    native = observations(("native", 100.0, 0.0), source=ObservationSource.NATIVE)
    ocr = observations(("recovered", 50.0, 20.0), source=ObservationSource.OCR)

    fused = fuse_observations(native, ocr, WorkPlan(PageRoute.HYBRID))

    assert list(fused.text) == ["native", "recovered"]


def test_uncovered_vector_text_keeps_low_confidence_labeled_ocr() -> None:
    native = observations(("native", 100.0, 0.0), source=ObservationSource.NATIVE)
    ocr = observations(("R1", 35.0, 20.0), ("~", 35.0, 40.0), source=ObservationSource.OCR)

    fused = fuse_observations(
        native,
        ocr,
        WorkPlan(
            PageRoute.HYBRID,
            reason=PagePlanReason.UNCOVERED_VECTOR_TEXT,
            fusion_policy=FusionPolicy.UNCOVERED_VECTOR,
        ),
    )

    assert list(fused.text) == ["native", "R1"]


def test_uncovered_vector_text_drops_raster_duplicates() -> None:
    native = observations(
        ("The annual report revenue increased", 100.0, 0.0),
        source=ObservationSource.NATIVE,
    )
    ocr = observations(
        ("The annual report revenue increased", 90.0, 20.0),
        ("uncovered label", 90.0, 40.0),
        source=ObservationSource.OCR,
    )

    fused = fuse_observations(
        native,
        ocr,
        WorkPlan(
            PageRoute.HYBRID,
            reason=PagePlanReason.UNCOVERED_VECTOR_TEXT,
            fusion_policy=FusionPolicy.UNCOVERED_VECTOR,
        ),
    )

    assert list(fused.text) == ["The annual report revenue increased", "uncovered label"]


def test_hybrid_fusion_drops_short_native_label_duplicates() -> None:
    native = observations(
        ("R1", 100.0, 0.0),
        ("GAIN", 100.0, 20.0),
        source=ObservationSource.NATIVE,
    )
    ocr = observations(
        ("R1", 90.0, 40.0),
        ("new label", 90.0, 60.0),
        source=ObservationSource.OCR,
    )

    fused = fuse_observations(native, ocr, WorkPlan(PageRoute.HYBRID))

    assert list(fused.text) == ["R1", "GAIN", "new label"]


def test_dense_native_uncovered_vector_text_keeps_uncovered_ocr_supplement() -> None:
    native = observations(
        *((f"native-{index}", 100.0, float(index * 12)) for index in range(32)),
        source=ObservationSource.NATIVE,
    )
    ocr = observations(
        ("native-0", 90.0, 500.0),
        ("new label", 90.0, 520.0),
        source=ObservationSource.OCR,
    )

    fused = fuse_observations(
        native,
        ocr,
        WorkPlan(
            PageRoute.HYBRID,
            reason=PagePlanReason.UNCOVERED_VECTOR_TEXT,
            fusion_policy=FusionPolicy.UNCOVERED_VECTOR,
        ),
    )

    assert list(fused.text) == [*native.text, "new label"]


def test_sparse_native_augmentation_prefers_dense_ocr() -> None:
    native = observations(
        ("18", 100.0, 0.0),
        source=ObservationSource.NATIVE,
    )
    ocr = observations(
        *((f"ocr-{index}", 80.0, float(200 + index * 12)) for index in range(32)),
        source=ObservationSource.OCR,
    )

    fused = fuse_observations(
        native,
        ocr,
        WorkPlan(
            PageRoute.HYBRID,
            reason=PagePlanReason.NATIVE_TEXT_NEEDS_AUGMENTATION,
            fusion_policy=FusionPolicy.SPARSE_NATIVE,
        ),
    )

    assert list(fused.text) == list(ocr.text)


def test_noisy_native_text_prefers_better_ocr_candidate() -> None:
    native = observations(
        *((f"r • {index} llfo{index}", 100.0, float(index * 12)) for index in range(6)),
        source=ObservationSource.NATIVE,
    )
    ocr = observations(
        *(
            (f"secondary coordinate table value {index}", 95.0, float(index * 12))
            for index in range(6)
        ),
        source=ObservationSource.OCR,
    )

    fused = fuse_observations(
        native,
        ocr,
        WorkPlan(
            PageRoute.HYBRID,
            reason=PagePlanReason.NOISY_NATIVE_TEXT,
            fusion_policy=FusionPolicy.NOISY_NATIVE,
        ),
    )

    assert list(fused.text) == list(ocr.text)


def test_noisy_native_text_keeps_native_when_ocr_is_weaker() -> None:
    native = observations(
        *((f"native technical value {index}", 100.0, float(index * 12)) for index in range(6)),
        source=ObservationSource.NATIVE,
    )
    ocr = observations(
        ("extra", 89.0, 200.0),
        ("weak supplement", 89.0, 220.0),
        source=ObservationSource.OCR,
    )

    fused = fuse_observations(
        native,
        ocr,
        WorkPlan(
            PageRoute.HYBRID,
            reason=PagePlanReason.NOISY_NATIVE_TEXT,
            fusion_policy=FusionPolicy.NOISY_NATIVE,
        ),
    )

    assert list(fused.text) == list(native.text)


def test_large_candidate_coverage_matches_maximum_native_overlap() -> None:
    native = ObservationBatch.from_columns(
        (f"native-{index}" for index in range(300)),
        ((float(index * 20), 0.0, float(index * 20 + 10), 10.0) for index in range(300)),
        source=ObservationSource.NATIVE,
    )
    boxes = [
        (5.0, 0.0, 15.0, 10.0),
        (40.0, 0.0, 50.0, 10.0),
        (10_000.0, 0.0, 10_010.0, 10.0),
    ]
    boxes.extend(
        (10_100.0 + float(index), 0.0, 10_101.0 + float(index), 1.0) for index in range(297)
    )
    candidates = ObservationBatch.from_columns(
        (f"candidate-{index}" for index in range(300)),
        boxes,
        source=ObservationSource.OCR,
    )

    coverage = maximum_candidate_coverage(candidates.bbox, native.bbox)

    assert tuple(round(float(value), 2) for value in coverage[:3]) == (0.5, 1.0, 0.0)
    assert not coverage[3:].any()
