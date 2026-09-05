import numpy
import pytest

from core_pdf_ocr.impl.extract.ocr.resampling import (
    resample_bilinear,
    resample_box,
    resample_nearest,
    resample_smooth,
)


def test_resample_nearest_handles_channel_arrays() -> None:
    source = numpy.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=numpy.uint8)

    result = resample_nearest(source, 4, 4)

    assert result.shape == (4, 4, 2)
    assert result[:, :, 0].tolist() == [
        [1, 1, 3, 3],
        [1, 1, 3, 3],
        [5, 5, 7, 7],
        [5, 5, 7, 7],
    ]
    assert result.flags.c_contiguous


def test_resample_box_averages_the_covered_source_area() -> None:
    source = numpy.asarray([[0, 100], [200, 255]], dtype=numpy.uint8)

    result = resample_box(source, 1, 1)

    assert result.shape == (1, 1)
    # Point sampling would return one corner; the average keeps every stroke.
    assert result[0, 0] == round((0 + 100 + 200 + 255) / 4)


def test_resample_box_preserves_a_flat_field_at_awkward_ratios() -> None:
    source = numpy.full((97, 53, 3), 200, dtype=numpy.uint8)

    result = resample_box(source, 31, 17)

    assert result.shape == (31, 17, 3)
    assert result.min() == 200
    assert result.max() == 200
    assert result.dtype == numpy.uint8


def test_resample_box_refuses_to_enlarge() -> None:
    source = numpy.zeros((4, 4), dtype=numpy.uint8)

    with pytest.raises(ValueError, match="only reduces"):
        resample_box(source, 8, 8)


def test_resample_bilinear_interpolates_between_neighbours() -> None:
    source = numpy.asarray([[0, 200]], dtype=numpy.uint8)

    result = resample_bilinear(source, 1, 4)

    # Replication would give [0, 0, 200, 200]; interpolation ramps between them.
    numpy.testing.assert_array_equal(result, ((0, 50, 150, 200),))


def test_resample_bilinear_preserves_a_flat_field() -> None:
    source = numpy.full((11, 7), 200, dtype=numpy.uint8)

    result = resample_bilinear(source, 40, 26)

    assert result.shape == (40, 26)
    assert result.min() == 200
    assert result.max() == 200
    assert result.flags.c_contiguous


def test_resample_smooth_dispatches_by_resize_direction() -> None:
    source = numpy.arange(24, dtype=numpy.uint8).reshape(4, 2, 3)

    reduced = resample_smooth(source, 2, 1)
    enlarged = resample_smooth(source, 8, 4)
    mixed = resample_smooth(source, 2, 4)

    numpy.testing.assert_array_equal(reduced, resample_box(source, 2, 1))
    numpy.testing.assert_array_equal(enlarged, resample_bilinear(source, 8, 4))
    numpy.testing.assert_array_equal(
        mixed,
        resample_bilinear(resample_box(source, 2, 2), 2, 4),
    )
