from copy import replace

import numpy
import pytest

from core_pdf.impl.render.model import RasterImage
from core_pdf_ocr.impl.extract.contracts import PageAnalysis
from core_pdf_ocr.impl.extract.ocr import raster
from core_pdf_ocr.impl.extract.ocr.types import internal_Raster


def internal_image(samples: numpy.ndarray) -> RasterImage:
    height, width, channels = samples.shape
    return RasterImage(samples.astype(numpy.uint8).tobytes(), width, height, channels)


@pytest.mark.parametrize("channels", [2, 4])
def test_transparent_pixels_do_not_become_ink_or_adaptive_text(channels: int) -> None:
    samples = numpy.full((20, 20, channels), 255, dtype=numpy.uint8)
    samples[5:10, 5:10] = 0
    image = internal_image(samples)
    source = internal_Raster(image, 144)
    assert raster.internal_raster_ink_grid(source, 2, 2).tolist() == [0, 0, 0, 0]
    adapted = raster.internal_adaptive_ocr_raster(source)
    assert min(adapted.image.pixels) == 255
    assert adapted.resolution == 144
    assert bytes(image.pixels) == samples.tobytes()


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_visual_ink_grid_counts_visible_regions(channels: int) -> None:
    samples = numpy.full((4, 4, channels), 255, dtype=numpy.uint8)
    samples[:2, :2, : 1 if channels == 2 else min(3, channels)] = 0
    source = internal_Raster(internal_image(samples), 72)
    assert raster.internal_raster_ink_grid(source, 2, 2).tolist() == [1, 0, 0, 0]
    assert raster.internal_raster_ink_grid(source, 0, 2).size == 0
    dense = raster.internal_raster_ink_grid(source, 8, 8)
    assert numpy.isfinite(dense).all()


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_adaptive_threshold_recovers_faded_dark_patch(channels: int) -> None:
    samples = numpy.full((20, 20, channels), 220, dtype=numpy.uint8)
    if channels == 4:
        samples[:, :, 3] = 255
    samples[8:12, 8:12, : min(3, channels)] = 180
    result = raster.internal_adaptive_ocr_raster(internal_Raster(internal_image(samples), 300))
    actual = result.image.array()[:, :, 0]
    assert set(numpy.unique(actual)) == {0, 255}
    assert actual[9, 9] == 0
    assert actual[0, 0] == 255
    assert result.image.channels == 1


@pytest.mark.parametrize("channels", [2, 4])
@pytest.mark.parametrize(("alpha", "expected"), [(0, 255), (128, 127), (255, 0)])
def test_compaction_composites_alpha_onto_white(channels: int, alpha: int, expected: int) -> None:
    sample = numpy.zeros((1, 1, channels), dtype=numpy.uint8)
    sample[:, :, -1] = alpha
    image = internal_image(sample)
    result = raster.internal_compact_ocr_image(image)
    assert result.channels == 1
    assert bytes(result.pixels) == bytes([expected])
    assert bytes(image.pixels) == sample.tobytes()


def test_luma_primary_colors_and_achromatic_values_are_exact() -> None:
    samples = numpy.asarray(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [73, 73, 73]]], dtype=numpy.uint8
    )
    assert raster.internal_luma(samples).tolist() == [[77, 149, 29, 73]]
    image = internal_image(samples)
    assert raster.internal_compact_ocr_image(image) is image
    assert raster.internal_compact_ocr_image(image, grayscale=True) is image


def test_large_rgb_images_take_requested_grayscale_path() -> None:
    image = RasterImage(bytes([73]) * 5000000 * 3, 2500, 2000, 3)
    result = raster.internal_compact_ocr_image(image, grayscale=True)
    assert result.channels == 1
    assert len(result.pixels) == 5000000
    assert min(result.pixels) == max(result.pixels) == 73


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_text_signal_recognizes_column_transitions_and_ignores_transparent_ink(
    channels: int,
) -> None:
    samples = numpy.full((20, 20, channels), 255, dtype=numpy.uint8)
    samples[:, ::2, : 1 if channels == 2 else min(3, channels)] = 0
    signal = raster.internal_raster_text_signal(internal_image(samples))
    assert signal.likely_text
    assert signal.horizontal_edge_ratio == 1
    if channels in {2, 4}:
        samples[:, :, -1] = 0
        assert not raster.internal_raster_text_signal(internal_image(samples)).likely_text


@pytest.mark.parametrize(("height", "width"), [(1, 1), (20, 1), (1, 20)])
def test_blank_narrow_images_have_no_text_signal(height: int, width: int) -> None:
    image = RasterImage(bytes([255]) * width * height, width, height, 1)
    signal = raster.internal_raster_text_signal(image)
    assert not signal.likely_text
    assert signal.horizontal_edge_ratio == 0


@pytest.mark.parametrize(
    ("boxes", "ratio", "full", "expected"),
    [
        ((), 0, False, None),
        (((10, 20, 100, 200),), 0.1, False, None),
        (((10, 20, 100, 200),), 0.65, False, (10, 20, 100, 200)),
        (((-10, -20, 100, 200),), 0.1, True, (0, 0, 100, 200)),
        (((0, 0, 600, 800),), 1, True, None),
        (((-100, -100, -20, -20),), 0.65, False, None),
    ],
)
def test_safe_crop_requires_image_dominance_and_valid_partial_page_bounds(
    ocr_capture: PageAnalysis,
    boxes: tuple[tuple[float, float, float, float], ...],
    ratio: float,
    full: bool,
    expected: tuple[float, float, float, float] | None,
) -> None:
    capture = replace(
        ocr_capture,
        evidence=replace(
            ocr_capture.evidence, image_boxes=boxes, image_area_ratio=ratio, full_page_image=full
        ),
    )
    assert raster.internal_safe_image_crop(capture) == expected


def test_photo_like_tones_need_strong_horizontal_structure_to_pass() -> None:
    y = numpy.arange(96)[:, None]
    x = numpy.arange(100)[None, :]
    photo = ((y % 30) * 8 + (x // 20) * 64) % 240
    structured = ((y % 30) * 8 + (x % 2) * 64) % 240
    photo_signal = raster.internal_raster_text_signal(internal_image(photo[:, :, None]))
    text_signal = raster.internal_raster_text_signal(internal_image(structured[:, :, None]))
    assert 0.015 <= photo_signal.horizontal_edge_ratio < 0.09
    assert not photo_signal.likely_text
    assert text_signal.horizontal_edge_ratio >= 0.09
    assert text_signal.likely_text


def test_fractional_alpha_intensity_agrees_across_pixel_consumers() -> None:
    samples = numpy.asarray([[[0, 0, 0, 128], [255, 255, 255, 128]]], dtype=numpy.uint8)
    image = internal_image(samples)
    assert raster.internal_visible_intensity(samples).tolist() == [[127, 255]]
    assert raster.internal_raster_ink_grid(internal_Raster(image, 72), 1, 2).tolist() == [1, 0]
    assert raster.internal_raster_text_signal(image).horizontal_edge_ratio == 1


def test_large_image_sampling_retains_distributed_text_signal() -> None:
    samples = numpy.full((1000, 1000, 1), 255, dtype=numpy.uint8)
    for x in range(0, 1000, 20):
        samples[:, x : x + 10] = 0
    image = internal_image(samples)
    assert raster.internal_raster_text_signal(image).likely_text
    assert raster.internal_raster_ink_grid(internal_Raster(image, 72), 2, 2).tolist() == [0.5] * 4
