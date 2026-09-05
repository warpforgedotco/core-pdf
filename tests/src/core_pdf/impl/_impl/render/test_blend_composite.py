# SPDX-License-Identifier: AGPL-3.0-only
"""`internal_composite_blended_group_numpy` must match the per-pixel math it replaced.

The kernel vectorizes what used to be one `internal_RasterTarget.blend_px` call per
pixel of a non-Normal transparency group. Its docstring promises the two agree bit
for bit -- the golden-raster digests depend on that -- so the scalar loop is kept
in ``tests.helpers.raster_reference`` as an independent oracle and the two are
compared byte for byte.
"""

from __future__ import annotations

import numpy
import pytest

from core_pdf.impl._impl.render.blend import internal_composite_blended_group_numpy
from core_pdf.impl._impl.runtime.array_views import uint8_image_view
from tests.helpers.raster_reference import random_rgba, reference_composite

BLEND_MODES = (None, "Normal", "Multiply", "MULTIPLY", "Screen", "Darken")
ALPHA_SCALES = (None, 0.0, 0.35, 1.0)


@pytest.mark.parametrize("blend_mode", BLEND_MODES)
@pytest.mark.parametrize("source_alpha_scale", ALPHA_SCALES)
@pytest.mark.parametrize("target_alpha_scale", ALPHA_SCALES)
def test_blended_group_matches_scalar_blend(
    blend_mode: str | None,
    source_alpha_scale: float | None,
    target_alpha_scale: float | None,
) -> None:
    width, height = 11, 9
    seed = hash((blend_mode, source_alpha_scale, target_alpha_scale)) & 0xFFFF
    destination = random_rgba(width * height, seed, (0, 1, 77, 128, 254, 255))
    source = bytes(random_rgba(width * height, seed + 1, (0, 1, 3, 128, 254, 255)))

    expected = bytearray(destination)
    reference_composite(expected, source, source_alpha_scale, target_alpha_scale, blend_mode)

    internal_composite_blended_group_numpy(
        uint8_image_view(destination, (height, width, 4)),
        uint8_image_view(bytearray(source), (height, width, 4)),
        source_alpha_scale,
        target_alpha_scale,
        blend_mode,
    )
    assert bytes(destination) == bytes(expected)


def test_blended_group_covers_every_byte_value() -> None:
    """Sweep all 256 source and destination alphas against every channel value."""
    values = bytearray()
    for source_alpha in range(256):
        for destination_alpha in range(256):
            values.extend((source_alpha, destination_alpha))
    count = 256 * 256
    destination = bytearray(count * 4)
    source = bytearray(count * 4)
    for index in range(count):
        offset = index * 4
        source_alpha = values[index * 2]
        destination_alpha = values[index * 2 + 1]
        destination[offset : offset + 4] = bytes(
            (index % 256, (index * 7) % 256, (index * 13) % 256, destination_alpha)
        )
        source[offset : offset + 4] = bytes(
            ((index * 3) % 256, (index * 11) % 256, (index * 5) % 256, source_alpha)
        )
    expected = bytearray(destination)
    reference_composite(expected, bytes(source), 0.35, 0.5, "Multiply")
    internal_composite_blended_group_numpy(
        uint8_image_view(destination, (count, 1, 4)),
        uint8_image_view(source, (count, 1, 4)),
        0.35,
        0.5,
        "Multiply",
    )
    assert bytes(destination) == bytes(expected)


def test_blended_group_leaves_fully_transparent_source_untouched() -> None:
    destination = random_rgba(64, 7, (0, 90, 255))
    original = bytes(destination)
    source = bytearray(64 * 4)
    for i in range(0, len(source), 4):
        source[i] = 200
        source[i + 1] = 100
        source[i + 2] = 50
    internal_composite_blended_group_numpy(
        uint8_image_view(destination, (8, 8, 4)),
        uint8_image_view(source, (8, 8, 4)),
        None,
        None,
        "Multiply",
    )
    assert bytes(destination) == original


def test_blended_group_accepts_an_empty_view() -> None:
    empty = numpy.empty((0, 0, 4), dtype=numpy.uint8)
    internal_composite_blended_group_numpy(empty, empty, 0.5, 0.5, "Screen")


def test_blended_group_writes_through_a_non_contiguous_destination() -> None:
    """A box slice narrower than the row stride must still receive every write."""
    width, height = 16, 6
    destination = random_rgba(width * height, 21, (0, 128, 255))
    view = uint8_image_view(destination, (height, width, 4))
    box = view[1:5, 2:9]
    assert not box.flags["C_CONTIGUOUS"]
    source = numpy.full((4, 7, 4), 200, dtype=numpy.uint8)
    before = bytes(destination)
    internal_composite_blended_group_numpy(box, source, None, None, "Screen")
    assert bytes(destination) != before
    # Only the boxed pixels may change.
    changed = numpy.asarray(bytearray(destination), dtype=numpy.uint8).reshape(height, width, 4)
    untouched = numpy.frombuffer(before, dtype=numpy.uint8).reshape(height, width, 4)
    mask = numpy.ones((height, width), dtype=bool)
    mask[1:5, 2:9] = False
    assert numpy.array_equal(changed[mask], untouched[mask])
