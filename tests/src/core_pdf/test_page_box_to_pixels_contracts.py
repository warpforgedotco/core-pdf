import math
import random

import pytest

from core_pdf.impl.render_grid import DeviceGrid


def by_formula(grid: DeviceGrid, x0: float, y0: float, x1: float, y1: float) -> object:
    ix0 = max(0, min(grid.width, math.floor((x0 - grid.crop_x0) * grid.scale)))
    ix1 = max(0, min(grid.width, math.ceil((x1 - grid.crop_x0) * grid.scale)))
    iy0 = max(0, min(grid.height, math.floor((grid.crop_y1 - y1) * grid.scale)))
    iy1 = max(0, min(grid.height, math.ceil((grid.crop_y1 - y0) * grid.scale)))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return ix0, iy0, ix1, iy1


@pytest.mark.parametrize("seed", range(20))
def test_clamped_edges_match_the_formula(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(2000):
        grid = DeviceGrid(
            crop_x0=rng.uniform(-50, 50),
            crop_y0=0.0,
            crop_y1=rng.uniform(0, 900),
            scale=rng.choice([0.5, 1.0, 1.5, 2.0, 2.7]),
            width=rng.randint(0, 400),
            height=rng.randint(0, 400),
        )
        box = tuple(rng.uniform(-100, 1000) for _ in range(4))
        assert grid.page_box_to_pixels(*box) == by_formula(grid, *box)


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_non_finite_edge_still_raises(value: float) -> None:
    grid = DeviceGrid(crop_x0=0.0, crop_y0=0.0, crop_y1=10.0, scale=1.0, width=10, height=10)
    with pytest.raises((ValueError, OverflowError)):
        grid.page_box_to_pixels(value, 0.0, 5.0, 5.0)
