# SPDX-License-Identifier: AGPL-3.0-only
"""Object blending is independent of its enclosing group's composite opacity."""

from __future__ import annotations

import random

import pytest

from core_pdf.impl._impl.render.model import internal_RasterGroup
from core_pdf.impl._impl.render.target import internal_RasterTarget
from tests.helpers.raster_reference import random_rgba, reference_blend_px

BLEND_MODES = (None, "Normal", "Multiply", "MULTIPLY", "Screen", "Darken")
GROUP_ALPHAS = (None, 0.0, 0.4, 1, 1.0)


def internal_bare_target(pixels: bytearray, group_alpha: float | None) -> internal_RasterTarget:
    """A target carrying only the two slots the blend path reads."""
    target = object.__new__(internal_RasterTarget)
    target.pixels = pixels
    target.buffer_stack = [internal_RasterGroup(pixels, group_alpha)]
    return target


@pytest.mark.parametrize("blend_mode", BLEND_MODES)
@pytest.mark.parametrize("group_alpha", GROUP_ALPHAS)
def test_blend_px_does_not_apply_enclosing_group_opacity(
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
        reference_blend_px(expected, index * 4, rgba, None, blend_mode)

    actual = random_rgba(count, seed)
    target = internal_bare_target(actual, group_alpha)
    mode = target.internal_resolved_blend(blend_mode)
    for index, rgba in enumerate(sources):
        target.blend_px(index * 4, rgba, mode)

    assert bytes(actual) == bytes(expected)


@pytest.mark.parametrize("blend_mode", BLEND_MODES)
@pytest.mark.parametrize("group_alpha", GROUP_ALPHAS)
def test_resolved_blend_only_normalizes_the_object_blend_mode(
    blend_mode: str | None, group_alpha: float | None
) -> None:
    target = internal_bare_target(bytearray(4), group_alpha)
    mode = target.internal_resolved_blend(blend_mode)

    assert mode == (blend_mode.lower() if isinstance(blend_mode, str) else None)


def test_resolved_blend_tolerates_an_empty_buffer_stack() -> None:
    target = object.__new__(internal_RasterTarget)
    target.pixels = bytearray(4)
    target.buffer_stack = []
    assert target.internal_resolved_blend("Multiply") == "multiply"


def test_blend_px_opaque_source_matches_the_general_path() -> None:
    """The `sa >= 255` shortcut must agree with the arithmetic it skips."""
    pixels = random_rgba(64, seed=3)
    shortcut = internal_bare_target(bytearray(pixels), None)
    general = internal_bare_target(bytearray(pixels), None)
    for index in range(64):
        rgba = (index * 3 % 256, index * 5 % 256, index * 7 % 256, 255)
        shortcut.blend_px(index * 4, rgba, None)
        # "darken" is not a mode blend_px premultiplies, so it takes the full
        # compositing path while computing a plain source-over blend.
        general.blend_px(index * 4, rgba, "darken")
    assert bytes(shortcut.pixels) == bytes(general.pixels)
