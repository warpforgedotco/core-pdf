# SPDX-License-Identifier: AGPL-3.0-only
"""Scalar reference implementations the vectorised raster kernels are checked against.

``reference_blend_px`` is ``internal_RasterTarget.blend_px`` as it was before the
per-pixel loop was lifted out: it takes the raw group alpha and the raw blend mode.
The golden-raster digests depend on the kernels agreeing with it bit for bit.
"""

from __future__ import annotations

import random


def reference_blend_px(
    pixels: bytearray,
    idx: int,
    rgba: tuple[int, int, int, int],
    target_alpha: float | None,
    blend_mode: str | None,
) -> None:
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


def reference_composite(
    destination: bytearray,
    source: bytes,
    source_alpha_scale: float | None,
    target_alpha_scale: float | None,
    blend_mode: str | None,
) -> None:
    """One ``reference_blend_px`` per source pixel, after the source alpha scale."""
    for idx in range(0, len(source), 4):
        sa = source[idx + 3]
        if sa <= 0:
            continue
        if source_alpha_scale is not None:
            sa = max(0, min(255, int(round(sa * source_alpha_scale))))
            if sa <= 0:
                continue
        rgba = (source[idx], source[idx + 1], source[idx + 2], sa)
        reference_blend_px(destination, idx, rgba, target_alpha_scale, blend_mode)


def random_rgba(
    count: int,
    seed: int,
    alpha_choices: tuple[int, ...] = (0, 1, 64, 128, 254, 255),
) -> bytearray:
    """``count`` RGBA pixels with uniform colour channels and alphas drawn from a set."""
    rng = random.Random(seed)
    buffer = bytearray(count * 4)
    for i in range(0, len(buffer), 4):
        buffer[i] = rng.randrange(256)
        buffer[i + 1] = rng.randrange(256)
        buffer[i + 2] = rng.randrange(256)
        buffer[i + 3] = rng.choice(alpha_choices)
    return buffer


__all__ = ("random_rgba", "reference_blend_px", "reference_composite")
