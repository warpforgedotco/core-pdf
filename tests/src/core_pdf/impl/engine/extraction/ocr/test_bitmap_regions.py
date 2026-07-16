from core_pdf.impl.engine.extraction.ocr.bitmap_regions import bitmap_vertical_regions
from core_pdf.impl.engine.extraction.ocr.types import OcrImage


def test_bitmap_vertical_regions_returns_tall_component() -> None:
    width, height = 40, 80
    data = bytearray([255] * (width * height))
    for y in range(10, 70):
        for x in range(15, 22):
            data[y * width + x] = 0
    image = OcrImage(bytes(data), width, height, 1, width)

    assert bitmap_vertical_regions(image) == ((15, 10, 22, 70),)


def test_bitmap_vertical_regions_rejects_short_component() -> None:
    width, height = 40, 80
    data = bytearray([255] * (width * height))
    for y in range(10, 20):
        for x in range(15, 22):
            data[y * width + x] = 0
    image = OcrImage(bytes(data), width, height, 1, width)

    assert bitmap_vertical_regions(image) == ()
