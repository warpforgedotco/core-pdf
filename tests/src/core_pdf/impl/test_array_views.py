import numpy
import pytest

from core_pdf.impl.runtime.array_views import (
    contiguous_bytes,
    finite_median,
    nearest_indices,
    uint8_image_view,
    uint8_view,
)


def test_finite_median_matches_numpy_without_mutating_input() -> None:
    for values in (
        numpy.asarray([3.0], dtype=numpy.float32),
        numpy.asarray([4.0, 1.0], dtype=numpy.float32),
        numpy.asarray([9.0, 2.0, 5.0], dtype=numpy.float32),
        numpy.asarray([8.0, 2.0, 6.0, 4.0], dtype=numpy.float32),
    ):
        original = values.copy()
        assert finite_median(values) == float(numpy.median(values))
        numpy.testing.assert_array_equal(values, original)

    with pytest.raises(ValueError, match="at least one"):
        finite_median(numpy.asarray([], dtype=numpy.float32))


def test_uint8_view_borrows_bytes() -> None:
    source = b"abcd"

    view = uint8_view(source)

    assert view.base is source
    assert view.tolist() == [97, 98, 99, 100]


def test_uint8_view_borrows_contiguous_uint8_array() -> None:
    source = numpy.arange(8, dtype=numpy.uint8).reshape(2, 4)

    view = uint8_view(source)
    view[0] = 99

    assert source[0, 0] == 99


def test_uint8_view_copies_non_contiguous_array() -> None:
    source = numpy.arange(8, dtype=numpy.uint8)[::2]

    view = uint8_view(source)
    view[0] = 99

    assert source[0] != 99


def test_uint8_image_view_validates_shape() -> None:
    with pytest.raises(ValueError, match="smaller"):
        uint8_image_view(b"abc", (2, 2))

    with pytest.raises(ValueError, match="larger"):
        uint8_image_view(b"abcde", (2, 2))

    assert uint8_image_view(b"abcde", (2, 2), allow_trailing=True).shape == (2, 2)


def test_nearest_indices_are_bounded_and_immutable() -> None:
    indexes = nearest_indices(5, 2)

    assert indexes.tolist() == [0, 0, 0, 1, 1]
    assert not indexes.flags.writeable


def test_contiguous_bytes_borrows_contiguous_and_copies_non_contiguous_arrays() -> None:
    contiguous = numpy.arange(4, dtype=numpy.uint8).reshape(2, 2)
    borrowed = contiguous_bytes(contiguous)
    borrowed[0] = 99

    assert contiguous[0, 0] == 99

    source = numpy.arange(8, dtype=numpy.uint8).reshape(2, 4)[:, ::2]

    encoded = contiguous_bytes(source)
    encoded[0] = 99

    assert bytes(encoded) == b"\x63\x02\x04\x06"
    assert source[0, 0] == 0
