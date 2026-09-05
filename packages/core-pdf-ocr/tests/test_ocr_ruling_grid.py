import numpy

from core_pdf.impl._impl.render.model import RasterImage
from core_pdf_ocr.impl.extract.grids import internal_detect_ruling_grid


def test_ruling_grid_detection_preserves_full_resolution_rgb_mask() -> None:
    height = 180
    width = 240
    samples = numpy.full((height, width, 3), 255, dtype=numpy.uint8)
    for x in (18, 72, 126, 180, 234):
        samples[:, x : x + 2] = (0, 220, 220)
    for y in (12, 51, 90, 129, 168):
        samples[y : y + 2, :] = (220, 0, 220)

    result = internal_detect_ruling_grid(RasterImage(samples.tobytes(), width, height, 3))

    assert result is not None
    x_lines, y_lines, source_samples, slope = result
    assert len(x_lines) == 5
    assert len(y_lines) == 5
    assert source_samples.shape == (height, width, 3)
    assert source_samples[12, 100].min() < 160
    assert source_samples[100, 18].min() < 160
    assert source_samples[100, 100].min() >= 160
    assert slope == 0.0


def test_ruling_grid_detection_rejects_plain_rgb_page() -> None:
    height = 180
    width = 240
    samples = numpy.full((height, width, 3), 255, dtype=numpy.uint8)

    assert internal_detect_ruling_grid(RasterImage(samples.tobytes(), width, height, 3)) is None
