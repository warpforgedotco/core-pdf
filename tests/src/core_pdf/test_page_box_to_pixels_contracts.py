"""A page box maps to pixels as max(0, min(size, floor/ceil(edge))) for each edge."""

import math
import random

import pytest

from core_pdf.impl.render_clipping import ClipState


def by_formula(clip: ClipState, x0: float, y0: float, x1: float, y1: float) -> object:
    ix0 = max(0, min(clip.width, math.floor((x0 - clip.crop_x0) * clip.scale)))
    ix1 = max(0, min(clip.width, math.ceil((x1 - clip.crop_x0) * clip.scale)))
    iy0 = max(0, min(clip.height, math.floor((clip.crop_y1 - y1) * clip.scale)))
    iy1 = max(0, min(clip.height, math.ceil((clip.crop_y1 - y0) * clip.scale)))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return ix0, iy0, ix1, iy1


@pytest.mark.parametrize("seed", range(20))
def test_clamped_edges_match_the_formula(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(2000):
        clip = ClipState(
            crop_x0=rng.uniform(-50, 50),
            crop_y1=rng.uniform(0, 900),
            scale=rng.choice([0.5, 1.0, 1.5, 2.0, 2.7]),
            width=rng.randint(0, 400),
            height=rng.randint(0, 400),
        )
        box = tuple(rng.uniform(-100, 1000) for _ in range(4))
        assert clip.page_box_to_pixels(*box) == by_formula(clip, *box)


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_non_finite_edge_still_raises(value: float) -> None:
    clip = ClipState(crop_x0=0.0, crop_y1=10.0, scale=1.0, width=10, height=10)
    with pytest.raises((ValueError, OverflowError)):
        clip.page_box_to_pixels(value, 0.0, 5.0, 5.0)
