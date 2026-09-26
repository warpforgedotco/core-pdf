import itertools

import numpy
import pytest

from core_pdf.impl.render_blend import FALLBACK_BLEND_CONTEXT, blend_channels_f64
from tests.src.core_pdf.raster_support import make_target

SOURCES = [(204, 85, 34, 1), (204, 85, 34, 64), (0, 255, 128, 128), (255, 255, 255, 254)]
DESTINATIONS = [(51, 102, 153, 0), (51, 102, 153, 128), (250, 5, 0, 255), (0, 0, 0, 37)]


def array_blend(source, destination, mode):
    sr, sg, sb, sa = (value / 255.0 for value in source)
    dr, dg, db, da = (numpy.array([float(value)]) for value in destination)
    channels = blend_channels_f64(
        sr, sg, sb, sa, dr, dg, db, da, mode, semantic_context=FALLBACK_BLEND_CONTEXT
    )
    return tuple(int(numpy.clip(channel[0], 0, 255)) for channel in channels)


@pytest.mark.parametrize("mode", [None, "multiply", "screen", "colordodge", "colorburn"])
def test_scalar_blend_matches_the_array_formulas(mode: str | None) -> None:
    for source, destination in itertools.product(SOURCES, DESTINATIONS):
        target = make_target(1, 1, pixel=bytes(destination))
        target.semantic_context = FALLBACK_BLEND_CONTEXT
        target.blend_px(0, source, mode)
        assert tuple(target.pixels[:4]) == array_blend(source, destination, mode), (
            source,
            destination,
        )
