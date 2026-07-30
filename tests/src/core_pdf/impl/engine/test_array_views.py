import numpy
import pytest

from core_pdf.impl.engine.array_views import (
    contiguous_bytes,
    nearest_indices,
    resample_nearest,
    typed_view,
    uint8_image_view,
    uint8_view,
)


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


def test_typed_view_supports_byte_offsets_and_array_inputs() -> None:
    source = b"\x00\x01\x00\x02\x00\x03"

    view = typed_view(source, ">u2", count=2, offset=2)
    assert view.tolist() == [2, 3]

    array = numpy.asarray([1, 2, 3], dtype=numpy.uint16)
    array_view = typed_view(array, numpy.dtype(numpy.uint16), count=2, offset=1)
    assert array_view.tolist() == [2, 3]


def test_nearest_indices_are_bounded_and_cached() -> None:
    indexes = nearest_indices(5, 2)

    assert indexes.tolist() == [0, 0, 0, 1, 1]
    assert indexes is nearest_indices(5, 2)
    assert not indexes.flags.writeable


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


def test_contiguous_bytes_copies_only_non_contiguous_arrays() -> None:
    source = numpy.arange(8, dtype=numpy.uint8).reshape(2, 4)[:, ::2]

    encoded = contiguous_bytes(source)

    assert bytes(encoded) == b"\x00\x02\x04\x06"
