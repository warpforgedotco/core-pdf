from core_pdf.impl.engine.extraction.ocr.types import OcrComponentBox
from core_pdf.impl.engine.extraction.ocr.vertical_regions import (
    map_clockwise_rotated_regions_to_source,
    vertical_regions_from_component_boxes,
)


def test_groups_aligned_components_into_vertical_region() -> None:
    boxes = [
        OcrComponentBox(5, index, 100 + index * 3, 20 + index * 40, 12, 18) for index in range(6)
    ]

    assert vertical_regions_from_component_boxes(boxes, image_width=800, image_height=800) == (
        (92, 12, 135, 246),
    )


def test_rejects_horizontal_and_sparse_components() -> None:
    boxes = [OcrComponentBox(5, index, index * 50, 100, 40, 20) for index in range(6)]

    assert vertical_regions_from_component_boxes(boxes, image_width=800, image_height=800) == ()


def test_maps_clockwise_region_back_to_source_coordinates() -> None:
    assert map_clockwise_rotated_regions_to_source(
        ((10, 20, 40, 80),), source_width=200, source_height=100
    ) == ((20, 60, 80, 90),)
