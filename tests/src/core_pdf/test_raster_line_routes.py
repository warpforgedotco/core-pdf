"""Equivalent line raster routes must preserve coverage and group attribution."""

import numpy as np
import pytest

from core_pdf.impl._impl.capture.records import CapturedPath
from core_pdf.impl._impl.render import path_stroke_target
from tests.src.core_pdf.test_pattern_rendering import internal_target


def internal_expected_coverage(cap: int, clipped: bool) -> np.ndarray:
    """Sample a width-three rectangle or capsule in line-local coordinates."""
    length = np.hypot(9, 7)
    tangent = np.array([9, 7]) / length
    normal = np.array([-7, 9]) / length
    counts = np.zeros((16, 16))
    for row in range(16):
        for column in range(16):
            if clipped and not (4 <= row < 12 and 5 <= column < 12):
                continue
            for sy in (0.125, 0.375, 0.625, 0.875):
                for sx in (0.125, 0.375, 0.625, 0.875):
                    relative = np.array([column + sx - 3, 16 - row - sy - 4])
                    along = float(relative @ tangent)
                    across = float(relative @ normal)
                    if cap == 1:
                        beyond = max(-along, along - length, 0)
                        inside = np.hypot(beyond, across) <= 1.5
                    else:
                        extension = 1.5 if cap == 2 else 0
                        # Normalizing the tangent can move exact endpoint samples by an ULP.
                        inside = (
                            -extension - 1e-12 <= along <= length + extension + 1e-12
                            and abs(across) <= 1.5
                        )
                    counts[row, column] += inside
    return counts / 16


@pytest.mark.parametrize("cap", [0, 1, 2])
@pytest.mark.parametrize("alpha", [0, 128, 255])
@pytest.mark.parametrize("clipped", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_line_routes_preserve_pixels_alpha_and_shape(
    monkeypatch: pytest.MonkeyPatch, cap: int, alpha: int, clipped: bool, reverse: bool
) -> None:
    results = []
    coverage = internal_expected_coverage(cap, clipped)
    for route in ("scalar", "vector", "blend"):
        monkeypatch.setattr(
            path_stroke_target, "RASTER_KERNEL_MIN_PIXEL_AREA", 10_000 if route == "scalar" else 0
        )
        target = internal_target(16, 16)
        target.push_group(bytearray(16 * 16 * 4), None, None, isolated=False, track_shape=True)
        if clipped:
            clip = CapturedPath()
            clip.rect(5, 4, 7, 8)
            target.clip.push(clip, "nonzero")
        endpoints = (12, 11, 3, 4) if reverse else (3, 4, 12, 11)
        target.fill_line(
            *endpoints,
            3,
            (200, 50, 10, alpha),
            blend_mode="Normal" if route == "blend" else None,
            line_cap=cap,
        )
        pixels = np.frombuffer(target.pixels, dtype=np.uint8).reshape(16, 16, 4).copy()
        group = target.pop_group()
        assert group.source_alpha is not None
        assert group.source_shape is not None
        np.testing.assert_allclose(group.source_shape, np.rint(255 * coverage) / 255, atol=1e-7)
        np.testing.assert_allclose(group.source_alpha, np.rint(alpha * coverage) / 255, atol=1e-7)
        assert np.count_nonzero(group.source_shape) > 0
        if alpha == 0:
            assert not np.any(pixels)
            assert not np.any(group.source_alpha)
        else:
            assert np.count_nonzero(pixels[..., 3]) > 0
        if clipped:
            outside = np.ones((16, 16), dtype=bool)
            outside[4:12, 5:12] = False
            assert not np.any(pixels[outside])
            assert not np.any(group.source_shape[outside])
        results.append((pixels, group.source_alpha, group.source_shape))
    for actual in results[1:]:
        for expected_plane, actual_plane in zip(results[0], actual, strict=True):
            np.testing.assert_allclose(actual_plane, expected_plane, atol=1e-7, rtol=0)
