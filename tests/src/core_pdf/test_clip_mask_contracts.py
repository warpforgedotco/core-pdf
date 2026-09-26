"""A clip's pixel mask marks exactly pixel_in_clip's pixels, and its row caches are released."""

import math
import random

import pytest

from core_pdf.impl.capture_records import CapturedPath
from tests.src.core_pdf.raster_support import make_target


def polygon(rng: random.Random, width: int, height: int) -> CapturedPath:
    path = CapturedPath()
    centre_x = rng.uniform(0, width)
    centre_y = rng.uniform(0, height)
    count = rng.randint(3, 12)
    points = [
        (
            centre_x + rng.uniform(2, width) * math.cos(2 * math.pi * index / count),
            centre_y + rng.uniform(2, height) * math.sin(2 * math.pi * index / count),
        )
        for index in range(count)
    ]
    rng.shuffle(points)
    path.move_to(*points[0])
    for point in points[1:]:
        path.line_to(*point)
    path.close()
    return path


@pytest.mark.parametrize("seed", range(20))
def test_mask_is_pixel_in_clip(seed: int) -> None:
    rng = random.Random(seed)
    target = make_target(40, 30)
    for _ in range(rng.randint(1, 3)):
        target.clip.push(polygon(rng, 40, 30), rng.choice(["nonzero", "evenodd"]))
    for _ in range(10):
        ix0, ix1 = sorted(rng.sample(range(41), 2))
        iy0, iy1 = sorted(rng.sample(range(31), 2))
        expected = bytearray(
            target.clip.pixel_in_clip(px, py) for py in range(iy0, iy1) for px in range(ix0, ix1)
        )
        assert target.clip_pixel_mask(ix0, iy0, ix1, iy1) == expected


def test_row_arrays_go_with_their_region() -> None:
    target = make_target(40, 30)
    clip = target.clip
    rng = random.Random(1)
    clip.push(polygon(rng, 40, 30), "nonzero")
    outer = clip.current_region()
    assert outer is not None
    clip.row_span_arrays(outer)
    depth = clip.depth
    for _ in range(3):
        clip.push(polygon(rng, 40, 30), "nonzero")
        region = clip.current_region()
        assert region is not None
        clip.row_span_arrays(region)
    assert len(clip.span_arrays) == 4
    clip.pop()
    assert len(clip.span_arrays) == 3
    clip.restore(depth)
    assert list(clip.span_arrays) == [id(outer)]
    clip.restore(0)
    assert not clip.span_arrays
