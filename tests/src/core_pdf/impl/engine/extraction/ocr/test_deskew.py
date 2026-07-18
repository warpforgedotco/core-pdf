from __future__ import annotations

import ctypes
from typing import Any, cast

import pytest
from core_ocr.impl import deskew
from core_ocr.impl.backend import (
    TesseractCtypesBackend,
    rgba_image_to_bmp,
)
from core_ocr.impl.types import (
    OcrImage,
    OcrObservation,
    OcrTextResult,
    ocr_int_value,
)


def test_rotation_transforms_round_trip_points() -> None:
    info = deskew.deskew_info(
        source="full_page_image",
        angle_degrees=2.5,
        confidence=8.0,
        applied=True,
        reason="applied",
        image_width=1200,
        image_height=1600,
    )

    deskewed = deskew.transform_point((235.5, 742.25), info.source_to_deskew)
    restored = deskew.transform_point(deskewed, info.deskew_to_source)

    assert restored == pytest.approx((235.5, 742.25))


def test_restore_result_geometry_inverse_maps_all_ocr_geometry() -> None:
    info = deskew.deskew_info(
        source="rendered_page_300dpi",
        angle_degrees=-2.0,
        confidence=11.25,
        applied=True,
        reason="applied",
        image_width=1000,
        image_height=1200,
    )
    original_bbox = (210, 330, 480, 390)
    deskewed_bbox = deskew.transform_pixel_bbox(
        original_bbox,
        info.source_to_deskew,
        width=info.image_width,
        height=info.image_height,
    )
    assert deskewed_bbox is not None
    left, top, right, bottom = deskewed_bbox
    result = OcrTextResult(
        "A line",
        91,
        line_rows=(
            {
                "text": "A line",
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
                "baseline": (left, bottom, right, bottom),
            },
        ),
        observations=(
            OcrObservation(
                "A line",
                2,
                91,
                deskewed_bbox,
                baseline=(left, bottom, right, bottom),
                image_width=1000,
                image_height=1200,
            ),
        ),
    )

    restored = deskew.restore_ocr_result_geometry(result, info)

    restored_row = restored.line_rows[0]
    restored_bbox = (
        ocr_int_value(restored_row["left"]),
        ocr_int_value(restored_row["top"]),
        ocr_int_value(restored_row["left"]) + ocr_int_value(restored_row["width"]),
        ocr_int_value(restored_row["top"]) + ocr_int_value(restored_row["height"]),
    )
    assert restored_bbox[0] <= original_bbox[0]
    assert restored_bbox[1] <= original_bbox[1]
    assert restored_bbox[2] >= original_bbox[2]
    assert restored_bbox[3] >= original_bbox[3]
    assert restored.observations[0].bbox == restored_bbox
    assert restored.observations[0].image_width == 1000
    assert restored.observations[0].image_height == 1200
    assert ("deskew_angle_degrees", -2.0) in restored.observations[0].provenance
    assert restored.deskew_info is info


def test_source_rectangle_is_scaled_then_forward_mapped() -> None:
    image = OcrImage(
        b"",
        100,
        200,
        4,
        400,
        source="full_page_image",
        target_width=200,
        target_height=400,
    )
    info = deskew.deskew_info(
        source=image.source,
        angle_degrees=1.5,
        confidence=6.0,
        applied=True,
        reason="applied",
        image_width=200,
        image_height=400,
    )

    mapped = deskew.source_rectangle_to_deskew_rectangle((10, 20, 50, 80), image, info)

    assert mapped == deskew.transform_pixel_bbox(
        (20, 40, 100, 160),
        info.source_to_deskew,
        width=200,
        height=400,
    )


def test_only_full_page_sources_are_eligible() -> None:
    def image(source: str) -> OcrImage:
        return OcrImage(b"\xff" * 4, 1, 1, 4, 4, source=source)

    assert deskew.full_page_ocr_image_should_be_deskewed(image("full_page_image"))
    assert deskew.full_page_ocr_image_should_be_deskewed(image("rendered_page_300dpi"))
    assert not deskew.full_page_ocr_image_should_be_deskewed(image("rendered_page_300dpi_tile_0_0"))
    assert not deskew.full_page_ocr_image_should_be_deskewed(image("figure_ocr"))


class _FakeLeptonica:
    def __init__(self, *, angle: float, confidence: float) -> None:
        self.angle = angle
        self.confidence = confidence
        self.convert_thresholds: list[int] = []
        self.rotate_angles: list[float] = []
        self.destroyed: list[int] = []

    def pixConvertTo1(self, _pix: ctypes.c_void_p, threshold: int) -> int:
        self.convert_thresholds.append(threshold)
        return 20

    def pixFindSkew(self, _pix: ctypes.c_void_p, angle: Any, confidence: Any) -> int:
        angle._obj.value = self.angle
        confidence._obj.value = self.confidence
        return 0

    def pixRotate(
        self,
        _pix: ctypes.c_void_p,
        angle: ctypes.c_float,
        _rotation_type: int,
        _incolor: int,
        _width: int,
        _height: int,
    ) -> int:
        self.rotate_angles.append(float(angle.value))
        return 30

    def pixDestroy(self, pix: Any) -> None:
        self.destroyed.append(int(pix._obj.value))


def deskew_backend(leptonica: _FakeLeptonica) -> TesseractCtypesBackend:
    backend = object.__new__(TesseractCtypesBackend)
    backend.leptonica = cast(Any, leptonica)
    backend.has_pix_convert_to_1 = True
    backend.has_pix_find_skew = True
    backend.has_pix_rotate = True
    return backend


def test_backend_applies_high_confidence_conservative_deskew() -> None:
    leptonica = _FakeLeptonica(angle=2.0, confidence=9.5)
    backend = deskew_backend(leptonica)

    pix, info = backend.deskew_pix(
        10,
        source="full_page_image",
        width=800,
        height=1000,
    )

    assert pix == 30
    assert info.applied
    assert info.angle_degrees == pytest.approx(2.0)
    assert info.confidence == pytest.approx(9.5)
    assert leptonica.convert_thresholds == [deskew.OCR_DESKEW_BINARY_THRESHOLD]
    assert leptonica.rotate_angles == pytest.approx([0.034906585])
    assert leptonica.destroyed == [20]


@pytest.mark.parametrize(
    ("angle", "confidence", "reason"),
    [
        (2.0, 3.99, "low_confidence"),
        (0.09, 8.0, "below_minimum_angle"),
        (5.01, 8.0, "angle_out_of_range"),
    ],
)
def test_backend_records_rejected_measurements_without_rotating(
    angle: float,
    confidence: float,
    reason: str,
) -> None:
    leptonica = _FakeLeptonica(angle=angle, confidence=confidence)
    backend = deskew_backend(leptonica)

    pix, info = backend.deskew_pix(
        10,
        source="full_page_image",
        width=800,
        height=1000,
    )

    assert pix == 10
    assert not info.applied
    assert info.reason == reason
    assert leptonica.rotate_angles == []
    assert leptonica.destroyed == [20]


def test_grayscale_image_can_be_wrapped_for_leptonica() -> None:
    image = OcrImage(bytes((16, 240)), 2, 1, 1, 2, source="full_page_image")

    bmp = rgba_image_to_bmp(image)

    assert bmp is not None
    assert bmp[54:60] == bytes((16, 16, 16, 240, 240, 240))
