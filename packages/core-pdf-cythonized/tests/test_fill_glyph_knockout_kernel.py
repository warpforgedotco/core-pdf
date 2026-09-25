# SPDX-License-Identifier: AGPL-3.0-only

"""The fused glyph knockout: what a glyph's scratch group made, in one pass.

core's knockout_glyph_fill copied the parent's backdrop window, zeroed two
float32 planes, filled the glyph into them with fill_glyph_coverage and
carried the result into the parent with composite_elementary_knockout. Both
kernels stay in this package, each pinned by its own vectors, so the
reference here is that composition, run over the glyph coverage golden cases
at opaque, translucent and invisible paints, into windows of larger parent
buffers whose surroundings must come through untouched, with and without a
parent shape plane.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import (
    composite_elementary_knockout,
    fill_glyph_coverage,
    fill_glyph_knockout,
)

GOLDEN_PATH = Path(__file__).parent / "glyph_coverage_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))
PAINTS = ((0, 0, 0, 255), (200, 30, 90, 255), (10, 250, 40, 128), (255, 255, 255, 1), (3, 4, 5, 0))
MARGIN = 2


def parent(rng, height, width, *, opaque_backdrop):
    shape = (height + 2 * MARGIN, width + 2 * MARGIN)
    backdrop = rng.integers(0, 256, (*shape, 4), dtype=numpy.uint8)
    if opaque_backdrop:
        backdrop[..., 3] = 255
    view = rng.integers(0, 256, (*shape, 4), dtype=numpy.uint8)
    group_alpha = rng.random(shape, dtype=numpy.float32)
    group_alpha[rng.random(shape) < 0.3] = 0.0
    parent_shape = rng.random(shape, dtype=numpy.float32)
    return backdrop, view, group_alpha, parent_shape


def window(array):
    return array[MARGIN:-MARGIN, MARGIN:-MARGIN]


def composed(case, rgba, backdrop, view, group_alpha, parent_shape, shape_scale):
    rendered = window(backdrop).copy()
    height, width = case["h"], case["w"]
    source_alpha = numpy.zeros((height, width), dtype=numpy.float32)
    source_shape = numpy.zeros((height, width), dtype=numpy.float32)
    drawn = fill_glyph_coverage(
        case["src"],
        case["crop_x0"],
        case["crop_y1"],
        case["scale"],
        int(case["ix0"]),
        int(case["iy0"]),
        width,
        height,
        rgba,
        rendered,
        source_alpha,
        source_shape,
        shape_scale,
    )
    if drawn is None:
        return None
    composite_elementary_knockout(
        window(view),
        window(backdrop),
        rendered,
        source_alpha,
        window(group_alpha),
        source_shape,
        None if parent_shape is None else window(parent_shape),
    )
    return drawn


def fused(case, rgba, backdrop, view, group_alpha, parent_shape, shape_scale):
    return fill_glyph_knockout(
        case["src"],
        case["crop_x0"],
        case["crop_y1"],
        case["scale"],
        int(case["ix0"]),
        int(case["iy0"]),
        case["w"],
        case["h"],
        rgba,
        window(view),
        window(backdrop),
        window(group_alpha),
        None if parent_shape is None else window(parent_shape),
        shape_scale,
    )


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_one_pass_matches_the_scratch_group(index):
    case = GOLDEN[index]
    rng = numpy.random.default_rng(index)
    for rgba in PAINTS:
        for with_shape in (False, True):
            for shape_scale in (1.0, 0.5):
                inputs = parent(rng, case["h"], case["w"], opaque_backdrop=bool(index % 2))
                reference = [array.copy() for array in inputs]
                candidate = [array.copy() for array in inputs]
                if not with_shape:
                    reference[3] = candidate[3] = None
                expected = composed(
                    case, rgba, reference[0], reference[1], reference[2], reference[3], shape_scale
                )
                got = fused(
                    case, rgba, candidate[0], candidate[1], candidate[2], candidate[3], shape_scale
                )
                assert got == expected
                for want, have in zip(reference, candidate, strict=True):
                    if want is None:
                        assert have is None
                    else:
                        assert want.tobytes() == have.tobytes()


def test_the_backdrop_is_only_read():
    case = next(case for case in GOLDEN if case["w"] and case["h"])
    rng = numpy.random.default_rng(7)
    backdrop, view, group_alpha, parent_shape = parent(
        rng, case["h"], case["w"], opaque_backdrop=False
    )
    before = backdrop.copy()
    fused(case, (40, 50, 60, 200), backdrop, view, group_alpha, parent_shape, 1.0)
    assert backdrop.tobytes() == before.tobytes()
