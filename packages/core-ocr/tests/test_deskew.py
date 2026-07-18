from __future__ import annotations

import pytest
from core_ocr.impl import deskew
from core_ocr.impl.types import OcrImage


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
    point = deskew.transform_point((235.5, 742.25), info.source_to_deskew)
    assert deskew.transform_point(point, info.deskew_to_source) == pytest.approx((235.5, 742.25))


def test_only_full_page_sources_are_eligible() -> None:
    def image(source: str) -> OcrImage:
        return OcrImage(b"\xff" * 4, 1, 1, 4, 4, source=source)

    assert deskew.full_page_ocr_image_should_be_deskewed(image("full_page_image"))
    assert not deskew.full_page_ocr_image_should_be_deskewed(image("figure_ocr"))
