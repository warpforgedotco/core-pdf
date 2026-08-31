# SPDX-License-Identifier: AGPL-3.0-only
"""`internal_composite_blended_group_numpy` must match the per-pixel math it replaced.

The kernel vectorizes what used to be one `internal_RasterTarget.blend_px` call per
pixel of a non-Normal transparency group. Its docstring promises the two agree bit
for bit -- the golden-raster digests depend on that -- so the scalar loop is kept
here as an independent oracle and the two are compared byte for byte.
"""

from __future__ import annotations

import random

import numpy
import pytest

from core_pdf.impl.render.kernels import internal_composite_blended_group_numpy
from core_pdf.impl.runtime.array_views import uint8_image_view

BLEND_MODES = (None, "Normal", "Multiply", "MULTIPLY", "Screen", "Darken")
ALPHA_SCALES = (None, 0.0, 0.35, 1.0)


def internal_reference_composite(
    destination: bytearray,
    source: bytes,
    source_alpha_scale: float | None,
    target_alpha_scale: float | None,
    blend_mode: str | None,
) -> None:
    """The scalar `blend_px` loop the kernel replaced, verbatim in shape."""
    for idx in range(0, len(source), 4):
        sa = source[idx + 3]
        if sa <= 0:
            continue
        if source_alpha_scale is not None:
            sa = max(0, min(255, int(round(sa * source_alpha_scale))))
            if sa <= 0:
                continue
        sr, sg, sb = source[idx], source[idx + 1], source[idx + 2]
        if target_alpha_scale is not None:
            sa = max(0, min(255, int(round(sa * target_alpha_scale))))
            if sa <= 0:
                continue
        dr, dg, db, da = (
            destination[idx],
            destination[idx + 1],
            destination[idx + 2],
            destination[idx + 3],
        )
        src_a = sa / 255.0
        dst_a = da / 255.0
        src_r, src_g, src_b = sr / 255.0, sg / 255.0, sb / 255.0
        dst_r, dst_g, dst_b = dr / 255.0, dg / 255.0, db / 255.0
        mode = blend_mode.lower() if isinstance(blend_mode, str) else None
        if mode == "multiply":
            src_r *= dst_r
            src_g *= dst_g
            src_b *= dst_b
        elif mode == "screen":
            src_r = 1.0 - (1.0 - src_r) * (1.0 - dst_r)
            src_g = 1.0 - (1.0 - src_g) * (1.0 - dst_g)
            src_b = 1.0 - (1.0 - src_b) * (1.0 - dst_b)
        out_a = src_a + dst_a * (1.0 - src_a)
        if out_a <= 0:
            destination[idx : idx + 4] = b"\x00\x00\x00\x00"
            continue
        out_r = int(round(((src_r * 255.0) * src_a + dr * dst_a * (1.0 - src_a)) / out_a))
        out_g = int(round(((src_g * 255.0) * src_a + dg * dst_a * (1.0 - src_a)) / out_a))
        out_b = int(round(((src_b * 255.0) * src_a + db * dst_a * (1.0 - src_a)) / out_a))
        destination[idx] = max(0, min(255, out_r))
        destination[idx + 1] = max(0, min(255, out_g))
        destination[idx + 2] = max(0, min(255, out_b))
        destination[idx + 3] = max(0, min(255, int(round(out_a * 255.0))))


def internal_random_rgba(count: int, alpha_choices: tuple[int, ...], seed: int) -> bytearray:
    rng = random.Random(seed)
    buffer = bytearray(count * 4)
    for i in range(0, len(buffer), 4):
        buffer[i] = rng.randrange(256)
        buffer[i + 1] = rng.randrange(256)
        buffer[i + 2] = rng.randrange(256)
        buffer[i + 3] = rng.choice(alpha_choices)
    return buffer


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
    destination = internal_random_rgba(width * height, (0, 1, 77, 128, 254, 255), seed)
    source = bytes(internal_random_rgba(width * height, (0, 1, 3, 128, 254, 255), seed + 1))

    expected = bytearray(destination)
    internal_reference_composite(
        expected, source, source_alpha_scale, target_alpha_scale, blend_mode
    )

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
    internal_reference_composite(expected, bytes(source), 0.35, 0.5, "Multiply")
    internal_composite_blended_group_numpy(
        uint8_image_view(destination, (count, 1, 4)),
        uint8_image_view(source, (count, 1, 4)),
        0.35,
        0.5,
        "Multiply",
    )
    assert bytes(destination) == bytes(expected)


def test_blended_group_leaves_fully_transparent_source_untouched() -> None:
    destination = internal_random_rgba(64, (0, 90, 255), seed=7)
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
    destination = internal_random_rgba(width * height, (0, 128, 255), seed=21)
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
