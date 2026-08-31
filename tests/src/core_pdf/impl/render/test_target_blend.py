# SPDX-License-Identifier: AGPL-3.0-only
"""`blend_px` takes its span-invariant arguments pre-resolved; the pixels must not move.

`internal_resolved_blend` lifts the enclosing group's alpha and the lowercased blend
mode out of the per-pixel loop that every paint method runs. This pins that pair
against the semantics `blend_px` had when it re-derived both on each call: the oracle
below takes the *raw* `buffer_stack` entry and the *raw* blend mode, exactly as the
method used to.
"""

from __future__ import annotations

import random

import pytest

from core_pdf.impl.render.target import internal_RasterTarget

BLEND_MODES = (None, "Normal", "Multiply", "MULTIPLY", "Screen", "Darken")
GROUP_ALPHAS = (None, 0.0, 0.4, 1, 1.0)


def internal_reference_blend_px(
    pixels: bytearray,
    idx: int,
    rgba: tuple[int, int, int, int],
    target_alpha: float | None,
    blend_mode: str | None,
) -> None:
    """`blend_px` as it was before the resolution moved out to the caller."""
    sr, sg, sb, sa = rgba
    if sa <= 0:
        return
    if sa >= 255 and target_alpha is None and blend_mode is None:
        pixels[idx : idx + 4] = bytes((sr, sg, sb, 255))
        return
    if type(target_alpha) is int or type(target_alpha) is float:
        sa = max(0, min(255, int(round(sa * float(target_alpha)))))
        if sa <= 0:
            return
    dr, dg, db, da = pixels[idx], pixels[idx + 1], pixels[idx + 2], pixels[idx + 3]
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
        pixels[idx : idx + 4] = b"\x00\x00\x00\x00"
        return
    out_r = int(round(((src_r * 255.0) * src_a + dr * dst_a * (1.0 - src_a)) / out_a))
    out_g = int(round(((src_g * 255.0) * src_a + dg * dst_a * (1.0 - src_a)) / out_a))
    out_b = int(round(((src_b * 255.0) * src_a + db * dst_a * (1.0 - src_a)) / out_a))
    pixels[idx] = max(0, min(255, out_r))
    pixels[idx + 1] = max(0, min(255, out_g))
    pixels[idx + 2] = max(0, min(255, out_b))
    pixels[idx + 3] = max(0, min(255, int(round(out_a * 255.0))))


def internal_bare_target(pixels: bytearray, group_alpha: float | None) -> internal_RasterTarget:
    """A target carrying only the two slots the blend path reads."""
    target = object.__new__(internal_RasterTarget)
    target.pixels = pixels
    target.buffer_stack = [(pixels, group_alpha, None)]
    return target


def internal_sample_pixels(count: int, seed: int) -> bytearray:
    rng = random.Random(seed)
    buffer = bytearray(count * 4)
    for i in range(0, len(buffer), 4):
        buffer[i] = rng.randrange(256)
        buffer[i + 1] = rng.randrange(256)
        buffer[i + 2] = rng.randrange(256)
        buffer[i + 3] = rng.choice((0, 1, 64, 128, 254, 255))
    return buffer


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

    expected = internal_sample_pixels(count, seed)
    for index, rgba in enumerate(sources):
        internal_reference_blend_px(expected, index * 4, rgba, group_alpha, blend_mode)

    actual = internal_sample_pixels(count, seed)
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
    pixels = internal_sample_pixels(64, seed=3)
    shortcut = internal_bare_target(bytearray(pixels), None)
    general = internal_bare_target(bytearray(pixels), None)
    for index in range(64):
        rgba = (index * 3 % 256, index * 5 % 256, index * 7 % 256, 255)
        shortcut.blend_px(index * 4, rgba, None, None)
        # "darken" is not a mode blend_px premultiplies, so it takes the full
        # compositing path while computing a plain source-over blend.
        general.blend_px(index * 4, rgba, None, "darken")
    assert bytes(shortcut.pixels) == bytes(general.pixels)
