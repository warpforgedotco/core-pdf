import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from core_pdf.impl._impl.capture.records import CapturedDrawing
from core_pdf.impl._impl.render.model import RasterImage
from core_pdf_ocr.impl.extract.contracts import OcrPass, OcrPassScope, PageAnalysis
from core_pdf_ocr.impl.extract.ocr import raster
from core_pdf_ocr.impl.extract.ocr.region_tasks import internal_tile_tasks
from core_pdf_ocr.impl.extract.ocr.types import (
    internal_map_ocr_box,
    internal_ocr_region_box,
    internal_Raster,
    internal_raster_rectangle_page_box,
)


@pytest.mark.parametrize(
    ("orientation", "expected"),
    [
        (raster.DirectImageOrientation.IDENTITY, [[1, 2, 3], [4, 5, 6]]),
        (raster.DirectImageOrientation.FLIP_X, [[3, 2, 1], [6, 5, 4]]),
        (raster.DirectImageOrientation.FLIP_Y, [[4, 5, 6], [1, 2, 3]]),
        (raster.DirectImageOrientation.FLIP_XY, [[6, 5, 4], [3, 2, 1]]),
        (raster.DirectImageOrientation.TRANSPOSE, [[1, 4], [2, 5], [3, 6]]),
        (raster.DirectImageOrientation.TRANSPOSE_FLIP_X, [[4, 1], [5, 2], [6, 3]]),
        (raster.DirectImageOrientation.TRANSPOSE_FLIP_Y, [[3, 6], [2, 5], [1, 4]]),
        (raster.DirectImageOrientation.TRANSPOSE_FLIP_XY, [[6, 3], [5, 2], [4, 1]]),
    ],
)
def test_image_orientation_preserves_pixels_and_resolution(
    orientation: raster.DirectImageOrientation,
    expected: list[list[int]],
) -> None:
    source = internal_Raster(RasterImage(bytes(range(1, 7)), 3, 2, 1), 300)
    drawing = CapturedDrawing(0, None, None)
    result = raster.internal_orient_direct_image_raster(drawing, source, orientation=orientation)
    assert result.image.array()[:, :, 0].tolist() == expected
    assert result.resolution == 300
    assert source.image.array()[:, :, 0].tolist() == [[1, 2, 3], [4, 5, 6]]


@pytest.mark.parametrize(
    ("quad", "orientation"),
    [
        (((0, 0), (30, 0), (0, 20), (30, 20)), raster.DirectImageOrientation.IDENTITY),
        (((30, 0), (30, 20), (0, 0), (0, 20)), raster.DirectImageOrientation.TRANSPOSE_FLIP_X),
        (((0, 0), (30, 2), (0, 20), (30, 22)), None),
        (((0, 0), (0, 0), (0, 20), (30, 20)), None),
    ],
)
def test_direct_image_orientation_requires_axis_aligned_placement(
    quad: Any,
    orientation: raster.DirectImageOrientation | None,
) -> None:
    drawing = CapturedDrawing(0, None, None, items=(("quad", quad),))
    assert raster.internal_direct_image_orientation(drawing) is orientation


@pytest.mark.parametrize(
    ("width", "height", "budget"), [(100, 80, 100), (1, 1000, 4), (1000, 1, 4), (10, 10, 396)]
)
@pytest.mark.parametrize("upscale", [False, True])
def test_decoded_image_respects_pixel_budget(
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    height: int,
    budget: int,
    upscale: bool,
) -> None:
    decoded = SimpleNamespace(data=bytes(width * height), width=width, height=height, channels=1)
    monkeypatch.setattr(raster, "decode_pdf_image", lambda *args: decoded)
    drawing = CapturedDrawing(0, None, None, raw_data=b"source", dictionary={})
    result = raster.internal_decoded_image_raster(drawing, 100, max_pixels=budget, upscale=upscale)
    assert result is not None
    assert 0 < result.width * result.height <= budget
    assert result.resolution >= 70


@pytest.mark.parametrize("user_unit", [1, 2.5])
@pytest.mark.parametrize("crop", [None, (10, 20, 13, 27)])
def test_render_budget_accounts_for_user_units_crop_and_rounding(
    ocr_capture: PageAnalysis,
    user_unit: float,
    crop: tuple[int, int, int, int] | None,
) -> None:
    class Rendered:
        def unrotated_raster_size(self, scale: float, *, crop: Any) -> tuple[int, int]:
            w, h = (600, 800) if crop is None else (crop[2] - crop[0], crop[3] - crop[1])
            return max(1, math.ceil(w * scale * user_unit)), max(
                1, math.ceil(h * scale * user_unit)
            )

        def rasterize(
            self, *, scale: float, max_pixels: int, crop: Any, **kwargs: Any
        ) -> RasterImage:
            assert crop == expected_crop
            w, h = self.unrotated_raster_size(scale, crop=crop)
            assert w * h <= max_pixels == 20
            return RasterImage(bytes(w * h), w, h, 1)

    expected_crop = crop
    capture = replace(ocr_capture, page=SimpleNamespace(width=600, height=800, user_unit=user_unit))
    result = raster.internal_rendered_page_raster(
        capture, 8, rendered=Rendered(), crop=crop, max_pixels=20
    )
    assert result is not None
    assert result.width * result.height <= 20


def test_tile_rectangles_overlap_without_changing_page_coordinates() -> None:
    source = internal_Raster(RasterImage(bytes(100 * 1000), 100, 1000, 1), 72)
    page_box = (10.0, 20.0, 210.0, 2020.0)
    tasks = internal_tile_tasks(
        source, page_box, OcrPass("tiles", OcrPassScope.TILES, 1, (3,), tiles=4)
    )
    assert len(tasks) == 4
    for first, second in zip(tasks, tasks[1:], strict=False):
        assert second.rectangle[1] < first.rectangle[1] + first.rectangle[3]
    for task in tasks:
        assert task.image is source.image
        assert task.page_box == page_box
        x, y, w, h = task.rectangle
        assert 0 <= x < x + w <= 100
        assert 0 <= y < y + h <= 1000
        assert internal_map_ocr_box(task, (x, y, x + w, y + h)) == (
            internal_raster_rectangle_page_box(source, page_box, task.rectangle)
        )
    assert internal_map_ocr_box(tasks[2], (10, 100, 30, 200)) == (30, 1620, 70, 1820)


@pytest.mark.parametrize(
    ("ratio", "boxes", "expected"),
    [
        (0.64, ((0, 0, 500, 600),), None),
        (0.65, ((-10, 20, 500, 600),), (0, 20, 500, 600)),
        (0.9, ((0, 0, 600, 800),), None),
        (0.9, ((700, 0, 800, 600),), None),
    ],
)
def test_safe_crop_does_not_hide_text_outside_sparse_images(
    ocr_capture: PageAnalysis,
    ratio: float,
    boxes: Any,
    expected: Any,
) -> None:
    capture = replace(
        ocr_capture,
        evidence=replace(ocr_capture.evidence, image_area_ratio=ratio, image_boxes=boxes),
    )
    assert raster.internal_safe_image_crop(capture) == expected


def test_region_padding_clips_to_page_and_rejects_empty_regions() -> None:
    assert internal_ocr_region_box(
        (-2, 5, 90, 100), page_width=100, page_height=100, padding=10
    ) == (0, 0, 100, 100)
    assert (
        internal_ocr_region_box((200, 0, 300, 20), page_width=100, page_height=100, padding=10)
        is None
    )


@pytest.mark.parametrize("representation", ["array", "bytearray", "bytes"])
def test_decoded_raster_accepts_supported_sample_buffers(monkeypatch, representation) -> None:
    import numpy

    source = numpy.array([[[0], [80]], [[160], [240]]], dtype=numpy.uint8)
    data = (
        source
        if representation == "array"
        else bytearray(source.tobytes())
        if representation == "bytearray"
        else source.tobytes()
    )
    decoded = SimpleNamespace(data=data, width=2, height=2, channels=1)
    monkeypatch.setattr(raster, "decode_pdf_image", lambda *args: decoded)
    drawing = CapturedDrawing(0, None, None, raw_data=b"source", dictionary={})
    result = raster.internal_decoded_image_raster(drawing, 4, upscale=False)
    assert result is not None
    assert bytes(result.image.pixels) == source.tobytes()
    assert result.resolution == 72


def test_unavailable_image_decoder_returns_no_raster(monkeypatch) -> None:
    monkeypatch.setattr(raster, "decode_pdf_image", lambda *args: None)
    drawing = CapturedDrawing(0, None, None, raw_data=b"bad", dictionary={})
    assert raster.internal_decoded_image_raster(drawing, 100) is None


def test_fractional_upscale_blends_pixels_within_budget(monkeypatch) -> None:
    decoded = SimpleNamespace(data=bytes([0, 100, 0, 100]), width=2, height=2, channels=1)
    monkeypatch.setattr(raster, "decode_pdf_image", lambda *args: decoded)
    drawing = CapturedDrawing(0, None, None, raw_data=b"source", dictionary={})
    result = raster.internal_decoded_image_raster(drawing, 4, max_pixels=30)
    assert result is not None
    assert (result.width, result.height) == (5, 5)
    assert result.image.array()[0, :, 0].tolist() == [0, 10, 50, 90, 100]
    assert result.resolution == 180


@pytest.mark.parametrize(
    "quad",
    [
        ((0,), (1, 0), (0, 1), (1, 1)),
        ((None, 0), (1, 0), (0, 1), (1, 1)),
        ((float("nan"), 0), (1, 0), (0, 1), (1, 1)),
        ((float("inf"), 0), (1, 0), (0, 1), (1, 1)),
        ((0, float("-inf")), (1, 0), (0, 1), (1, 1)),
        ((0, 0),) * 4,
    ],
)
def test_direct_orientation_rejects_malformed_nonfinite_or_degenerate_quads(quad) -> None:
    drawing = CapturedDrawing(0, None, None, items=(("quad", quad),))
    assert raster.internal_direct_image_orientation(drawing) is None


def test_malformed_compositor_image_does_not_prevent_native_extraction(ocr_capture) -> None:
    class Rendered:
        def unrotated_raster_size(self, scale, *, crop):
            return (10, 10)

        def rasterize(self, **kwargs):
            raise IndexError("source sample outside decoded image")

    capture = replace(ocr_capture, page=SimpleNamespace(width=100, height=100))
    assert raster.internal_rendered_page_raster(capture, 1, rendered=Rendered()) is None


def test_whole_factor_upscale_replicates_pixels_exactly(monkeypatch) -> None:
    import numpy

    source = numpy.arange(100, dtype=numpy.uint8).reshape(10, 10, 1)
    decoded = SimpleNamespace(data=source.tobytes(), width=10, height=10, channels=1)
    monkeypatch.setattr(raster, "decode_pdf_image", lambda *args: decoded)
    drawing = CapturedDrawing(0, None, None, raw_data=b"source", dictionary={})
    result = raster.internal_decoded_image_raster(drawing, 51.84, max_pixels=1600)
    assert result is not None
    assert (result.width, result.height, result.resolution) == (40, 40, 400)
    numpy.testing.assert_array_equal(
        result.image.array(), source.repeat(4, axis=0).repeat(4, axis=1)
    )


def test_decoded_array_reduction_preserves_source_buffer(monkeypatch) -> None:
    import numpy

    source = numpy.arange(100, dtype=numpy.uint8).reshape(10, 10, 1)
    original = source.copy()
    decoded = SimpleNamespace(data=source, width=10, height=10, channels=1)
    monkeypatch.setattr(raster, "decode_pdf_image", lambda *args: decoded)
    drawing = CapturedDrawing(0, None, None, raw_data=b"source", dictionary={})
    result = raster.internal_decoded_image_raster(drawing, 100, max_pixels=25, upscale=False)
    assert result is not None
    assert result.width * result.height <= 25
    assert result.image.array()[:, :, 0].tolist() == [
        [0, 2, 5, 7],
        [20, 22, 25, 27],
        [50, 52, 55, 57],
        [70, 72, 75, 77],
    ]
    numpy.testing.assert_array_equal(source, original)


@pytest.mark.parametrize(
    ("quad", "expected"),
    [
        (((0, 0), (30, 0), (0, 20), (30, 20)), [[1, 2, 3], [4, 5, 6]]),
        (((30, 0), (0, 0), (30, 20), (0, 20)), [[3, 2, 1], [6, 5, 4]]),
        (((0, 20), (30, 20), (0, 0), (30, 0)), [[4, 5, 6], [1, 2, 3]]),
        (((30, 20), (0, 20), (30, 0), (0, 0)), [[6, 5, 4], [3, 2, 1]]),
        (((0, 0), (0, 30), (20, 0), (20, 30)), [[1, 4], [2, 5], [3, 6]]),
        (((20, 0), (20, 30), (0, 0), (0, 30)), [[4, 1], [5, 2], [6, 3]]),
        (((0, 30), (0, 0), (20, 30), (20, 0)), [[3, 6], [2, 5], [1, 4]]),
        (((20, 30), (20, 0), (0, 30), (0, 0)), [[6, 3], [5, 2], [4, 1]]),
    ],
)
@pytest.mark.parametrize("channels", [1, 3])
def test_direct_image_pixels_follow_placed_source_corners(quad, expected, channels):
    drawing = CapturedDrawing(0, None, None, items=(("quad", quad),))
    source = internal_Raster(
        RasterImage(bytes(value for value in range(1, 7) for _ in range(channels)), 3, 2, channels),
        300,
    )
    result = raster.internal_orient_direct_image_raster(drawing, source)
    assert result.image.array().tolist() == [
        [[value] * channels for value in row] for row in expected
    ]
    assert result.resolution == source.resolution
    assert source.image.pixels == bytes(value for value in range(1, 7) for _ in range(channels))
