# SPDX-License-Identifier: AGPL-3.0-only

"""Check the compiled function-based and mesh shading kernels against Python, bit for bit.

function_values gives each pixel the point of a type 1 shading's Domain it shows (ISO
32000-2 8.7.4.5.2); mesh_values rasterizes triangles and interpolates their vertices'
values by barycentric weights (Gouraud shading, 8.7.4.5.5); patch_triangles divides Coons
and tensor-product patches into such triangles (8.7.4.5.7, 8.7.4.5.8). This computes the
same arrays in Python, with the same operations in the same order, for thousands of random
cases under random transforms, BBoxes, clips and scales, and edge cases (degenerate,
sliver and huge triangles, shared edges, folded and flat patches, infinite points), and
compares them exactly.

uv run --all-packages python scripts/check_mesh_shading.py [random cases]
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy

from core_pdf.impl.spec.s_08_graphics.matrix import invert_affine
from core_pdf_cythonized import function_values, mesh_values, patch_triangles


def inside(box: tuple[float, ...], x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def function_reference(
    crop_x0: float,
    crop_y1: float,
    scale: float,
    pixel_box: tuple[int, int, int, int],
    allowed: Any,
    inverse: tuple[float, ...] | None,
    bbox: tuple[float, ...] | None,
    to_domain: tuple[float, ...],
    domain: tuple[float, ...],
) -> tuple[Any, Any]:
    ix0, iy0, ix1, iy1 = pixel_box
    values = numpy.zeros((iy1 - iy0, ix1 - ix0, 2), dtype=numpy.float64)
    painted = numpy.zeros((iy1 - iy0, ix1 - ix0), dtype=numpy.uint8)
    d = to_domain
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
            else:
                x, y = page_x, page_y
            if bbox is not None and not inside(bbox, x, y):
                continue
            dx = d[0] * x + d[2] * y + d[4]
            dy = d[1] * x + d[3] * y + d[5]
            if not (domain[0] <= dx <= domain[1] and domain[2] <= dy <= domain[3]):
                continue
            values[r, col, 0] = dx
            values[r, col, 1] = dy
            painted[r, col] = 1
    return values, painted


def mesh_reference(
    triangles: Any,
    crop_x0: float,
    crop_y1: float,
    scale: float,
    pixel_box: tuple[int, int, int, int],
    allowed: Any,
    forward: tuple[float, ...] | None,
    inverse: tuple[float, ...] | None,
    bbox: tuple[float, ...] | None,
) -> tuple[Any, Any]:
    ix0, iy0, ix1, iy1 = pixel_box
    k = triangles.shape[2] - 2
    values = numpy.zeros((iy1 - iy0, ix1 - ix0, k), dtype=numpy.float64)
    painted = numpy.zeros((iy1 - iy0, ix1 - ix0), dtype=numpy.uint8)
    for triangle in triangles.tolist():
        xs = []
        ys = []
        for sx, sy, *_ in triangle:
            if forward is not None:
                f = forward
                px = f[0] * sx + f[2] * sy + f[4]
                py = f[1] * sx + f[3] * sy + f[5]
            else:
                px, py = sx, sy
            xs.append((px - crop_x0) * scale)
            ys.append((crop_y1 - py) * scale)
        area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if area == 0.0 or not math.isfinite(area):
            continue
        positive = area > 0.0
        low = max(float(ix0), math.ceil(min(xs) - 0.5))
        high = min(float(ix1 - 1), math.floor(max(xs) - 0.5))
        if low > high:
            continue
        column_low, column_high = int(low), int(high)
        low = max(float(iy0), math.ceil(min(ys) - 0.5))
        high = min(float(iy1 - 1), math.floor(max(ys) - 0.5))
        if low > high:
            continue
        for row in range(int(low), int(high) + 1):
            cy = float(row) + 0.5
            for column in range(column_low, column_high + 1):
                cx = float(column) + 0.5
                w0 = (xs[1] - cx) * (ys[2] - cy) - (xs[2] - cx) * (ys[1] - cy)
                w1 = (xs[2] - cx) * (ys[0] - cy) - (xs[0] - cx) * (ys[2] - cy)
                w2 = (xs[0] - cx) * (ys[1] - cy) - (xs[1] - cx) * (ys[0] - cy)
                if positive:
                    if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                        continue
                elif w0 > 0.0 or w1 > 0.0 or w2 > 0.0:
                    continue
                r, col = row - iy0, column - ix0
                if not allowed[r, col]:
                    continue
                if bbox is not None:
                    page_x = crop_x0 + (float(column) + 0.5) / scale
                    page_y = crop_y1 - (float(row) + 0.5) / scale
                    if inverse is not None:
                        m = inverse
                        x = m[0] * page_x + m[2] * page_y + m[4]
                        y = m[1] * page_x + m[3] * page_y + m[5]
                    else:
                        x, y = page_x, page_y
                    if not inside(bbox, x, y):
                        continue
                l0, l1, l2 = w0 / area, w1 / area, w2 / area
                for j in range(k):
                    values[r, col, j] = (
                        l0 * triangle[0][2 + j] + l1 * triangle[1][2 + j] + l2 * triangle[2][2 + j]
                    )
                painted[r, col] = 1
    return values, painted


def device_length(linear: tuple[float, ...], dx: float, dy: float) -> float:
    ex = linear[0] * dx + linear[2] * dy
    ey = linear[1] * dx + linear[3] * dy
    return math.sqrt(ex * ex + ey * ey)


def larger(current: float, length: float) -> float:
    """length when it is larger than current or not a number, as `if not length <= current`;
    max() would keep current over a NaN."""
    if length <= current:
        return current
    return length


def divisions_reference(
    p: list[list[float]],
    c: list[list[float]],
    linear: tuple[float, ...],
    tolerance: float,
    color_tolerances: list[float],
    max_divisions: int,
) -> int:
    along_u = along_v = twist = 0.0
    for j in range(4):
        for i in range(2):
            a, b, e = p[i * 4 + j], p[(i + 1) * 4 + j], p[(i + 2) * 4 + j]
            dx = a[0] - 2.0 * b[0] + e[0]
            dy = a[1] - 2.0 * b[1] + e[1]
            along_u = larger(along_u, device_length(linear, dx, dy))
    for i in range(4):
        for j in range(2):
            a, b, e = p[i * 4 + j], p[i * 4 + j + 1], p[i * 4 + j + 2]
            dx = a[0] - 2.0 * b[0] + e[0]
            dy = a[1] - 2.0 * b[1] + e[1]
            along_v = larger(along_v, device_length(linear, dx, dy))
    for i in range(3):
        for j in range(3):
            a, b = p[i * 4 + j], p[(i + 1) * 4 + j]
            e, g = p[i * 4 + j + 1], p[(i + 1) * 4 + j + 1]
            dx = a[0] - b[0] - e[0] + g[0]
            dy = a[1] - b[1] - e[1] + g[1]
            twist = larger(twist, device_length(linear, dx, dy))
    need = 1.0
    bound = (6.0 * along_u + 18.0 * twist + 6.0 * along_v) / 8.0
    if not bound <= 0.0:
        need = math.sqrt(bound / tolerance) if bound == bound else math.nan
    for j, color_tolerance in enumerate(color_tolerances):
        if color_tolerance > 0.0:
            length = abs(c[0][j] - c[1][j] + c[2][j] - c[3][j])
            bound = math.sqrt(length / (4.0 * color_tolerance)) if length == length else math.nan
            need = larger(need, bound)
    if not need <= float(max_divisions):
        return max_divisions
    need = math.ceil(need)
    return 1 if need < 1.0 else int(need)


def patch_reference(
    patches: Any,
    corners: Any,
    linear: tuple[float, ...],
    tolerance: float,
    color_tolerances: list[float],
    max_divisions: int,
) -> Any:
    out: list[list[list[float]]] = []
    for p, c in zip(patches.tolist(), corners.tolist(), strict=True):
        n = divisions_reference(p, c, linear, tolerance, color_tolerances, max_divisions)
        grid: list[list[float]] = []
        for b in range(n + 1):
            v = float(b) / float(n)
            mv = 1.0 - v
            bv = (mv * mv * mv, 3.0 * v * mv * mv, 3.0 * v * v * mv, v * v * v)
            for a in range(n + 1):
                u = float(a) / float(n)
                mu = 1.0 - u
                bu = (mu * mu * mu, 3.0 * u * mu * mu, 3.0 * u * u * mu, u * u * u)
                x = 0.0
                y = 0.0
                for i in range(4):
                    for j in range(4):
                        weight = bu[i] * bv[j]
                        x = x + p[i * 4 + j][0] * weight
                        y = y + p[i * 4 + j][1] * weight
                values = [
                    mu * mv * c[0][j] + mu * v * c[1][j] + u * v * c[2][j] + u * mv * c[3][j]
                    for j in range(len(color_tolerances))
                ]
                grid.append([x, y, *values])
        for b in range(n):
            for a in range(n):
                g00 = b * (n + 1) + a
                g10 = g00 + 1
                g11 = (b + 1) * (n + 1) + a + 1
                g01 = (b + 1) * (n + 1) + a
                out.append([grid[g00], grid[g10], grid[g11]])
                out.append([grid[g00], grid[g11], grid[g01]])
    width = 2 + len(color_tolerances)
    return numpy.asarray(out, dtype=numpy.float64).reshape(-1, 3, width)


def random_matrix(random: numpy.random.Generator, index: int) -> tuple[float, ...] | None:
    if index % 4 == 0:
        return None
    angle = float(random.uniform(0, 2 * math.pi))
    sx, sy = (float(v) for v in random.uniform(0.3, 2.5, 2))
    matrix = (
        sx * math.cos(angle),
        sx * math.sin(angle),
        -sy * math.sin(angle) + (float(random.uniform(-0.5, 0.5)) if index % 3 == 0 else 0.0),
        sy * math.cos(angle),
        float(random.uniform(0, 120)),
        float(random.uniform(0, 120)),
    )
    if index % 7 == 1:
        matrix = (-matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5])
    return matrix


def random_frame(random: numpy.random.Generator) -> dict[str, Any]:
    width, height = int(random.integers(1, 48)), int(random.integers(1, 48))
    ix0, iy0 = int(random.integers(0, 120)), int(random.integers(0, 120))
    return {
        "crop_x0": float(random.uniform(-10, 10)),
        "crop_y1": float(random.uniform(150, 250)),
        "scale": float(random.choice([1.0, 2.0, 0.5, 1.3])),
        "pixel_box": (ix0, iy0, ix0 + width, iy0 + height),
        "allowed": (random.random((height, width)) > 0.1).astype(numpy.uint8),
    }


def random_triangles(random: numpy.random.Generator, index: int, k: int) -> Any:
    count = int(random.integers(1, 12))
    points = random.uniform(-20, 220, (count, 3, 2))
    edge = index % 11
    if edge == 1:
        points[:, 2] = points[:, 1]  # degenerate: two vertices at one point
    elif edge == 2:
        points[:, 2] = points[:, 0] + (points[:, 1] - points[:, 0]) * 0.5  # collinear
    elif edge == 3:
        points[:, 2] = points[:, 1] + random.uniform(-0.3, 0.3, (count, 2))  # slivers
    elif edge == 4 and count > 1:
        points[1:, 0] = points[:-1, 1]  # shared edges
        points[1:, 1] = points[:-1, 2]
    elif edge == 5:
        points = numpy.round(points)  # vertices on pixel edges
    elif edge == 6:
        points[0, 0, 0] = math.inf
    elif edge == 7:
        points *= 1e6  # far beyond the box
    values = random.uniform(-1, 2, (count, 3, k))
    return numpy.ascontiguousarray(numpy.concatenate((points, values), axis=2))


def random_patches(random: numpy.random.Generator, index: int, k: int) -> tuple[Any, Any]:
    count = int(random.integers(1, 4))
    spread = float(random.choice([2.0, 30.0, 150.0]))
    base = numpy.stack(
        numpy.meshgrid(numpy.linspace(0, 1, 4), numpy.linspace(0, 1, 4), indexing="ij"), axis=-1
    ).reshape(16, 2)
    patches = base[None] * spread * 1.5 + random.uniform(-spread, spread, (count, 16, 2)) * 0.4
    patches += random.uniform(0, 100, (count, 1, 2))
    edge = index % 9
    if edge == 1:
        patches[:] = patches[:, :1]  # a patch collapsed to a point
    elif edge == 2:
        patches[:, 5:7] = patches[:, 9:11][:, ::-1] * 3.0  # folds
    elif edge == 3:
        patches[0, 3, 0] = math.nan
    corners = random.uniform(-1, 1, (count, 4, k))
    return numpy.ascontiguousarray(patches), numpy.ascontiguousarray(corners)


def same(got: tuple[Any, Any], want: tuple[Any, Any]) -> bool:
    return bool(numpy.array_equal(got[1], want[1])) and got[0].tobytes() == want[0].tobytes()


def main(argv: list[str]) -> int:
    count = int(argv[0]) if argv else 1500
    random = numpy.random.default_rng(20261115)
    failures: list[str] = []
    pixels = triangles = 0
    for index in range(count):
        frame = random_frame(random)
        ix0, iy0, ix1, iy1 = frame["pixel_box"]
        common = (frame["crop_x0"], frame["crop_y1"], frame["scale"])
        matrix = random_matrix(random, index)
        inverse = invert_affine(matrix) if matrix is not None else None
        bbox = None
        if index % 3 == 1:
            low = random.uniform(-20, 120, 2)
            bbox = (float(low[0]), float(low[1]), float(low[0] + 90), float(low[1] + 90))

        # Type 1: Domain and its Matrix.
        domain_matrix = random_matrix(random, index + 1) or (60.0, 0.0, 0.0, 60.0, 30.0, 30.0)
        to_domain = invert_affine(domain_matrix)
        if to_domain is not None:
            domain = (-1.0, 1.0, -0.5, 2.0) if index % 2 else (0.0, 1.0, 0.0, 1.0)
            got = function_values(
                *common, ix0, iy0, ix1, iy1, frame["allowed"], inverse, bbox, to_domain, domain
            )
            want = function_reference(
                *common, frame["pixel_box"], frame["allowed"], inverse, bbox, to_domain, domain
            )
            if not same(got, want):
                failures.append(f"case {index}: function_values differs")
            pixels += got[1].size

        # Types 4 and 5: triangles.
        k = int(random.choice([1, 3, 4]))
        mesh = random_triangles(random, index, k)
        got = mesh_values(
            mesh, *common, ix0, iy0, ix1, iy1, frame["allowed"], matrix, inverse, bbox
        )
        want = mesh_reference(
            mesh, *common, frame["pixel_box"], frame["allowed"], matrix, inverse, bbox
        )
        if not same(got, want):
            failures.append(f"case {index}: mesh_values differs (edge {index % 11}, k {k})")
        triangles += len(mesh)

        # Types 6 and 7: patches.
        patches, corners = random_patches(random, index, k)
        scale = frame["scale"]
        outer = matrix if matrix is not None else (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        linear = (outer[0] * scale, outer[1] * scale, outer[2] * scale, outer[3] * scale)
        tolerances = [float(v) for v in random.choice([0.0, 1.0 / 512.0, 0.01], k)]
        tolerance = float(random.choice([0.25, 0.5, 2.0]))
        max_divisions = int(random.choice([1, 8, 64]))
        got_triangles = patch_triangles(
            patches,
            corners,
            linear,
            tolerance,
            numpy.asarray(tolerances, dtype=numpy.float64),
            max_divisions,
        )
        want_triangles = patch_reference(
            patches, corners, linear, tolerance, tolerances, max_divisions
        )
        if got_triangles.shape != want_triangles.shape or (
            got_triangles.tobytes() != want_triangles.tobytes()
        ):
            failures.append(f"case {index}: patch_triangles differs (edge {index % 9})")
        triangles += len(got_triangles)
    print(f"{count} cases, {pixels} type 1 pixels, {triangles} triangles: {len(failures)} failures")
    for failure in failures[:20]:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
