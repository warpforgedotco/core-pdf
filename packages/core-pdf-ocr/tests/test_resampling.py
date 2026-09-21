import numpy
import pytest

from core_pdf_ocr.impl.extract.ocr import resampling


@pytest.mark.parametrize(
    "resize",
    [
        resampling.resample_nearest,
        resampling.resample_box,
        resampling.resample_bilinear,
        resampling.resample_smooth,
    ],
)
@pytest.mark.parametrize(
    ("shape", "height", "width", "message"),
    [
        ((2, 2), 0, 2, "dimensions must be positive"),
        ((2, 2), 2, -1, "dimensions must be positive"),
        ((2,), 2, 2, "2D or 3D"),
        ((1, 1, 1, 1), 2, 2, "2D or 3D"),
        ((0, 2), 2, 2, "positive spatial"),
        ((2, 0), 2, 2, "positive spatial"),
    ],
)
def test_resampling_rejects_invalid_dimensions_consistently(
    resize, shape, height, width, message
) -> None:
    with pytest.raises(ValueError, match=message):
        resize(numpy.zeros(shape, dtype=numpy.uint8), height, width)


@pytest.mark.parametrize(
    "resize",
    [
        resampling.resample_nearest,
        resampling.resample_box,
        resampling.resample_bilinear,
        resampling.resample_smooth,
    ],
)
def test_unchanged_contiguous_image_preserves_identity(resize) -> None:
    source = numpy.arange(12, dtype=numpy.uint8).reshape(2, 2, 3)
    assert resize(source, 2, 2) is source


def test_nearest_neighbour_handles_noncontiguous_views_without_changing_channels() -> None:
    source = numpy.arange(24, dtype=numpy.uint8).reshape(2, 4, 3)[:, ::2]
    copy = resampling.resample_nearest(source, 2, 2)
    numpy.testing.assert_array_equal(copy, source)
    assert copy.flags.c_contiguous
    result = resampling.resample_nearest(source, 4, 4)
    numpy.testing.assert_array_equal(result, source.repeat(2, axis=0).repeat(2, axis=1))


def test_box_reduction_retains_thin_stroke_energy() -> None:
    source = numpy.array([[0, 255, 255, 255], [0, 255, 255, 255]], dtype=numpy.uint8)
    assert resampling.resample_box(source, 1, 2).tolist() == [[128, 255]]
    assert source.tolist() == [[0, 255, 255, 255]] * 2
    with pytest.raises(ValueError, match="only reduces"):
        resampling.resample_box(source, 4, 2)


def test_bilinear_interpolation_uses_pixel_centers_and_clamps_edges() -> None:
    source = numpy.array([[0, 100]], dtype=numpy.uint8)
    assert resampling.resample_bilinear(source, 3, 4).tolist() == [[0, 25, 75, 100]] * 3
    assert resampling.resample_smooth(source, 3, 4).tolist() == [[0, 25, 75, 100]] * 3


@pytest.mark.parametrize("transpose", [False, True])
def test_mixed_axis_resize_averages_before_enlarging(transpose) -> None:
    source = numpy.array([[0, 100], [20, 120], [40, 140], [60, 160]], dtype=numpy.uint8)
    expected = numpy.array([[10, 35, 85, 110], [50, 75, 125, 150]], dtype=numpy.uint8)
    if transpose:
        source, expected = source.T, expected.T
    result = resampling.resample_smooth(source, *expected.shape)
    numpy.testing.assert_array_equal(result, expected)
    assert result.dtype == source.dtype
    assert result.flags.c_contiguous


def test_single_pixel_bilinear_expansion_preserves_each_channel() -> None:
    source = numpy.array([[[3, 73, 255, 127]]], dtype=numpy.uint8)
    result = resampling.resample_bilinear(source, 3, 5)
    numpy.testing.assert_array_equal(result, numpy.broadcast_to(source, (3, 5, 4)))
