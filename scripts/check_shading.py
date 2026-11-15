# SPDX-License-Identifier: AGPL-3.0-only

"""Check the compiled shading_values against a Python reference, bit for bit.

shading_values gives each pixel of a box the function input of an axial (type 2) or radial
(type 3) shading, and whether the shading paints it (ISO 32000-2 8.7.4.5.3 and 8.7.4.5.4).
Without an inverse, the pixel centre's point in default user space is the shading-space
point; with one, the inverse of the shading's transform (the CTM at sh, or a shading
pattern's matrix) takes that point back to shading space, and a BBox clips there. This
computes the same values in Python, with the same operations in the same order, for
thousands of random shadings, transforms, boxes and Extend and Domain settings, and edge
cases (degenerate axes, concentric and touching circles, a zero radius), and compares both
arrays exactly.

uv run --all-packages python scripts/check_shading.py [random cases]
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy

from core_pdf.impl.spec.s_08_graphics.matrix import invert_affine
from core_pdf_cythonized import shading_values


def axial_t(c: list[float], px: float, py: float) -> float | None:
    dx = c[2] - c[0]
    dy = c[3] - c[1]
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return None
    return ((px - c[0]) * dx + (py - c[1]) * dy) / denom


def radial_t(c: list[float], px: float, py: float, extend0: bool, extend1: bool) -> float | None:
    """The largest s whose circle passes through the point, with a radius of at least 0 and
    inside the extended range (8.7.4.5.4)."""
    dx = c[3] - c[0]
    dy = c[4] - c[1]
    dr = c[5] - c[2]
    qx = px - c[0]
    qy = py - c[1]
    a = dx * dx + dy * dy - dr * dr
    b = -2.0 * (qx * dx + qy * dy + c[2] * dr)
    cc = qx * qx + qy * qy - c[2] * c[2]

    def usable(s: float) -> bool:
        return (
            math.isfinite(s)
            and c[2] + s * dr >= 0.0
            and (s >= 0.0 or extend0)
            and (s <= 1.0 or extend1)
        )

    if abs(a) <= 1e-12:
        if abs(b) <= 1e-12:
            return None
        s = -cc / b
        return s if usable(s) else None
    disc = b * b - 4.0 * a * cc
    if disc < 0.0:
        return None
    root = disc**0.5
    s0 = (-b - root) / (2.0 * a)
    s1 = (-b + root) / (2.0 * a)
    high, low = (s1, s0) if s1 > s0 else (s0, s1)
    if usable(high):
        return high
    if usable(low):
        return low
    return None


def reference(
    kind: int,
    coords: list[float],
    crop_x0: float,
    crop_y1: float,
    scale: float,
    box: tuple[int, int, int, int],
    allowed: Any,
    extend0: bool,
    extend1: bool,
    domain0: float,
    domain_span: float,
    inverse: tuple[float, ...] | None,
    bbox: tuple[float, ...] | None,
) -> tuple[Any, Any]:
    ix0, iy0, ix1, iy1 = box
    values = numpy.zeros((iy1 - iy0, ix1 - ix0), dtype=numpy.float64)
    painted = numpy.zeros((iy1 - iy0, ix1 - ix0), dtype=numpy.uint8)
    for r in range(iy1 - iy0):
        page_y = crop_y1 - (float(iy0 + r) + 0.5) / scale
        for col in range(ix1 - ix0):
            if not allowed[r, col]:
                continue
            page_x = crop_x0 + (float(ix0 + col) + 0.5) / scale
            if inverse is not None:
                m = inverse
                x = m[0] * page_x + m[2] * page_y + m[4]
                y = m[1] * page_x + m[3] * page_y + m[5]
                if bbox is not None and not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
                    continue
            else:
                x, y = page_x, page_y
            t = axial_t(coords, x, y) if kind == 2 else radial_t(coords, x, y, extend0, extend1)
            if t is None:
                continue
            if t < 0.0:
                if not extend0:
                    continue
                t = 0.0
            elif t > 1.0:
                if not extend1:
                    continue
                t = 1.0
            values[r, col] = domain0 + t * domain_span
            painted[r, col] = 1
    return values, painted


def random_case(random: numpy.random.Generator, index: int) -> dict[str, Any]:
    kind = 2 if index % 2 == 0 else 3
    unit = index % 3 == 0
    spread = 2.0 if unit else 120.0
    if kind == 2:
        coords = [float(v) for v in random.uniform(-spread, spread, 4)]
    else:
        centres = random.uniform(-spread, spread, 4)
        radii = random.uniform(0.0, spread, 2)
        coords = [float(centres[0]), float(centres[1]), float(radii[0])]
        coords += [float(centres[2]), float(centres[3]), float(radii[1])]
    edge = index % 17
    if edge == 1:
        coords[2:4] = coords[0:2] if kind == 2 else coords[2:4]
    elif edge == 2 and kind == 3:
        coords[3:5] = coords[0:2]
    elif edge == 3 and kind == 3:
        coords[2] = 0.0
    elif edge == 4 and kind == 3:
        coords[5] = coords[2] + math.hypot(coords[3] - coords[0], coords[4] - coords[1])
    inverse = None
    bbox = None
    if index % 5:
        angle = float(random.uniform(0, 2 * math.pi))
        sx, sy = (float(v) for v in random.uniform(0.2, 3.0, 2) * (60.0 if unit else 1.0))
        matrix = (
            sx * math.cos(angle),
            sx * math.sin(angle),
            -sy * math.sin(angle),
            sy * math.cos(angle),
            float(random.uniform(0, 200)),
            float(random.uniform(0, 200)),
        )
        if index % 7 == 0:
            matrix = (matrix[0], 0.0, 0.0, matrix[3], matrix[4], matrix[5])
        inverse = invert_affine(matrix)
        if index % 3 == 1:
            low = random.uniform(-spread, 0, 2)
            bbox = (float(low[0]), float(low[1]), float(low[0] + spread), float(low[1] + spread))
    domain = random.uniform(-2, 2, 2)
    width, height = int(random.integers(1, 40)), int(random.integers(1, 40))
    ix0, iy0 = int(random.integers(0, 160)), int(random.integers(0, 160))
    allowed = (random.random((height, width)) > 0.1).astype(numpy.uint8)
    return {
        "kind": kind,
        "coords": coords,
        "crop_x0": float(random.uniform(-10, 10)),
        "crop_y1": float(random.uniform(150, 250)),
        "scale": float(random.choice([1.0, 2.0, 0.5, 1.3])),
        "box": (ix0, iy0, ix0 + width, iy0 + height),
        "allowed": allowed,
        "extend0": bool(random.integers(0, 2)),
        "extend1": bool(random.integers(0, 2)),
        "domain0": float(domain[0]),
        "domain_span": float(domain[1] - domain[0]),
        "inverse": inverse,
        "bbox": bbox,
    }


def main(argv: list[str]) -> int:
    count = int(argv[0]) if argv else 3000
    random = numpy.random.default_rng(20261114)
    failures: list[str] = []
    pixels = 0
    for index in range(count):
        case = random_case(random, index)
        ix0, iy0, ix1, iy1 = case["box"]
        got_values, got_painted = shading_values(
            case["kind"],
            case["coords"],
            case["crop_x0"],
            case["crop_y1"],
            case["scale"],
            ix0,
            iy0,
            ix1,
            iy1,
            case["allowed"],
            case["extend0"],
            case["extend1"],
            case["domain0"],
            case["domain_span"],
            0.5,
            case["inverse"],
            case["bbox"],
        )
        want_values, want_painted = reference(
            case["kind"],
            case["coords"],
            case["crop_x0"],
            case["crop_y1"],
            case["scale"],
            case["box"],
            case["allowed"],
            case["extend0"],
            case["extend1"],
            case["domain0"],
            case["domain_span"],
            case["inverse"],
            case["bbox"],
        )
        pixels += got_values.size
        if not numpy.array_equal(got_painted, want_painted):
            failures.append(f"case {index}: painted differs ({case['kind']}, {case['coords']})")
        elif got_values.tobytes() != want_values.tobytes():
            failures.append(f"case {index}: values differ ({case['kind']}, {case['coords']})")
    print(f"{count} shadings, {pixels} pixels: {len(failures)} failures")
    for failure in failures[:20]:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
