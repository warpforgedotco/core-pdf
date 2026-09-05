# SPDX-License-Identifier: AGPL-3.0-only
"""`blend_px` takes its span-invariant arguments pre-resolved; the pixels must not move.

`internal_resolved_blend` lifts the enclosing group's alpha and the lowercased blend
mode out of the per-pixel loop that every paint method runs. This pins that pair
against the semantics `blend_px` had when it re-derived both on each call: the oracle in
``tests.helpers.raster_reference`` takes the *raw* `buffer_stack` entry and the *raw*
blend mode, exactly as the method used to.
"""

from __future__ import annotations

import random

import pytest

from core_pdf.impl._impl.render.target import internal_RasterTarget
from tests.helpers.raster_reference import random_rgba, reference_blend_px

BLEND_MODES = (None, "Normal", "Multiply", "MULTIPLY", "Screen", "Darken")
GROUP_ALPHAS = (None, 0.0, 0.4, 1, 1.0)


def internal_bare_target(pixels: bytearray, group_alpha: float | None) -> internal_RasterTarget:
    """A target carrying only the two slots the blend path reads."""
    target = object.__new__(internal_RasterTarget)
    target.pixels = pixels
    target.buffer_stack = [(pixels, group_alpha, None)]
    return target


@pytest.mark.parametrize("blend_mode", BLEND_MODES)
@pytest.mark.parametrize("group_alpha", GROUP_ALPHAS)
def test_blend_px_matches_unresolved_semantics(
    blend_mode: str | None, group_alpha: float | None
) -> None:
    count = 256
    seed = hash((blend_mode, group_alpha)) & 0xFFFF
    rng = random.Random(seed)
    sources = [
        (rng.randrange(256), rng.randrange(256), rng.randrange(256), source_alpha)
        for source_alpha in (0, 1, 3, 90, 128, 254, 255) * 37
    ][:count]

    expected = random_rgba(count, seed)
    for index, rgba in enumerate(sources):
        reference_blend_px(expected, index * 4, rgba, group_alpha, blend_mode)

    actual = random_rgba(count, seed)
    target = internal_bare_target(actual, group_alpha)
    alpha_scale, mode = target.internal_resolved_blend(blend_mode)
    for index, rgba in enumerate(sources):
        target.blend_px(index * 4, rgba, alpha_scale, mode)

    assert bytes(actual) == bytes(expected)


@pytest.mark.parametrize("blend_mode", BLEND_MODES)
@pytest.mark.parametrize("group_alpha", GROUP_ALPHAS)
def test_resolved_blend_reports_the_hoisted_pair(
    blend_mode: str | None, group_alpha: float | None
) -> None:
    target = internal_bare_target(bytearray(4), group_alpha)
    alpha_scale, mode = target.internal_resolved_blend(blend_mode)

    assert mode == (blend_mode.lower() if isinstance(blend_mode, str) else None)
    if group_alpha is None:
        assert alpha_scale is None
    else:
        assert alpha_scale == float(group_alpha)
        assert isinstance(alpha_scale, float)


def test_resolved_blend_tolerates_an_empty_buffer_stack() -> None:
    target = object.__new__(internal_RasterTarget)
    target.pixels = bytearray(4)
    target.buffer_stack = []
    assert target.internal_resolved_blend("Multiply") == (None, "multiply")


def test_blend_px_opaque_source_matches_the_general_path() -> None:
    """The `sa >= 255` shortcut must agree with the arithmetic it skips."""
    pixels = random_rgba(64, seed=3)
    shortcut = internal_bare_target(bytearray(pixels), None)
    general = internal_bare_target(bytearray(pixels), None)
    for index in range(64):
        rgba = (index * 3 % 256, index * 5 % 256, index * 7 % 256, 255)
        shortcut.blend_px(index * 4, rgba, None, None)
        # "darken" is not a mode blend_px premultiplies, so it takes the full
        # compositing path while computing a plain source-over blend.
        general.blend_px(index * 4, rgba, None, "darken")
    assert bytes(shortcut.pixels) == bytes(general.pixels)
