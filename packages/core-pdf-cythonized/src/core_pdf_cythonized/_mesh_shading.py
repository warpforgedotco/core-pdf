# SPDX-License-Identifier: AGPL-3.0-only

"""Function-based (type 1) and mesh (types 4 to 7) shadings: the values each pixel takes.

function_values gives each pixel the point of a type 1 shading's Domain it shows.
mesh_values rasterizes triangles, each vertex carrying k values, and interpolates the values
over each triangle by its barycentric coordinates (Gouraud shading); a pixel whose centre a
later triangle covers takes that triangle's values. patch_triangles divides each Coons or
tensor-product patch into triangles fine enough for mesh_values to paint.
"""

import cython
import numpy
from cython.cimports.libc.math import ceil, fabs, floor, isfinite, sqrt

__all__ = ("function_values", "mesh_values", "patch_triangles")


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def inside_box(box: cython.p_const_double, x: cython.double, y: cython.double) -> cython.bint:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


@cython.cfunc
@cython.exceptval(-1)
def read_numbers(
    values, count: cython.Py_ssize_t, name: str, target: cython.p_double
) -> cython.int:
    i: cython.Py_ssize_t
    if len(values) != count:
        raise ValueError(f"{name} must hold {count} numbers")
    for i in range(count):
        target[i] = values[i]
    return 0


def function_values(
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    ix1: cython.Py_ssize_t,
    iy1: cython.Py_ssize_t,
    allowed: cython.const[cython.uchar][:, ::1],
    inverse,
    bbox,
    to_domain,
    domain,
):
    """Each pixel's point (x, y) of a type 1 shading's Domain, (H, W, 2), and whether the
    shading paints it. inverse (or None for the identity) takes the pixel centre's default
    user space point to the shading's target space, where bbox (or None) clips; to_domain
    takes that point to Domain's space, and only points inside domain, (xmin, xmax, ymin,
    ymax), are painted."""
    height: cython.Py_ssize_t = iy1 - iy0
    width: cython.Py_ssize_t = ix1 - ix0
    if allowed.shape[0] != height or allowed.shape[1] != width:
        raise ValueError("allowed differs from the box in shape")
    m = cython.declare(cython.double[6])
    d = cython.declare(cython.double[6])
    box = cython.declare(cython.double[4])
    limits = cython.declare(cython.double[4])
    mapped: cython.bint = inverse is not None
    boxed: cython.bint = bbox is not None
    if mapped:
        read_numbers(inverse, 6, "inverse", m)
    if boxed:
        read_numbers(bbox, 4, "bbox", box)
    read_numbers(to_domain, 6, "to_domain", d)
    read_numbers(domain, 4, "domain", limits)
    values_array = numpy.zeros((height, width, 2), dtype=numpy.float64)
    painted_array = numpy.zeros((height, width), dtype=numpy.uint8)
    values: cython.double[:, :, ::1] = values_array
    painted: cython.uchar[:, ::1] = painted_array
    r: cython.Py_ssize_t
    col: cython.Py_ssize_t
    page_x: cython.double
    page_y: cython.double
    target_x: cython.double
    target_y: cython.double
    domain_x: cython.double
    domain_y: cython.double
    with cython.nogil:
        for r in range(height):
            page_y = crop_y1 - (cython.cast(cython.double, iy0 + r) + 0.5) / scale
            for col in range(width):
                if not allowed[r, col]:
                    continue
                page_x = crop_x0 + (cython.cast(cython.double, ix0 + col) + 0.5) / scale
                if mapped:
                    target_x = m[0] * page_x + m[2] * page_y + m[4]
                    target_y = m[1] * page_x + m[3] * page_y + m[5]
                else:
                    target_x = page_x
                    target_y = page_y
                if boxed and not inside_box(
                    cython.cast(cython.p_const_double, box), target_x, target_y
                ):
                    continue
                domain_x = d[0] * target_x + d[2] * target_y + d[4]
                domain_y = d[1] * target_x + d[3] * target_y + d[5]
                if not (limits[0] <= domain_x <= limits[1] and limits[2] <= domain_y <= limits[3]):
                    continue
                values[r, col, 0] = domain_x
                values[r, col, 1] = domain_y
                painted[r, col] = 1
    return values_array, painted_array


def mesh_values(
    triangles: cython.const[cython.double][:, :, ::1],
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    ix1: cython.Py_ssize_t,
    iy1: cython.Py_ssize_t,
    allowed: cython.const[cython.uchar][:, ::1],
    forward=None,
    inverse=None,
    bbox=None,
):
    """Each pixel's values, (H, W, k), and whether a triangle paints it. triangles is
    (T, 3, 2 + k): each vertex's point in the shading's target space and its k values.
    forward (or None for the identity) takes target space to default user space. A pixel
    is painted when its centre lies inside a triangle or on its edge, and takes the values
    of the last such triangle, weighted by the centre's barycentric coordinates. bbox, when
    given, keeps only centres inside it in target space, where inverse (the inverse of
    forward, or None) takes them."""
    height: cython.Py_ssize_t = iy1 - iy0
    width: cython.Py_ssize_t = ix1 - ix0
    if allowed.shape[0] != height or allowed.shape[1] != width:
        raise ValueError("allowed differs from the box in shape")
    if triangles.shape[1] != 3 or triangles.shape[2] < 2:
        raise ValueError("triangles must be (T, 3, 2 + k)")
    k: cython.Py_ssize_t = triangles.shape[2] - 2
    f = cython.declare(cython.double[6])
    m = cython.declare(cython.double[6])
    box = cython.declare(cython.double[4])
    mapped: cython.bint = forward is not None
    boxed: cython.bint = bbox is not None
    inverted: cython.bint = inverse is not None
    if mapped:
        read_numbers(forward, 6, "forward", f)
    if inverted:
        read_numbers(inverse, 6, "inverse", m)
    if boxed:
        if mapped and not inverted:
            raise ValueError("bbox under forward needs its inverse")
        read_numbers(bbox, 4, "bbox", box)
    values_array = numpy.zeros((height, width, k), dtype=numpy.float64)
    painted_array = numpy.zeros((height, width), dtype=numpy.uint8)
    values: cython.double[:, :, ::1] = values_array
    painted: cython.uchar[:, ::1] = painted_array
    xs = cython.declare(cython.double[3])
    ys = cython.declare(cython.double[3])
    t: cython.Py_ssize_t
    v: cython.Py_ssize_t
    j: cython.Py_ssize_t
    row: cython.Py_ssize_t
    column: cython.Py_ssize_t
    r: cython.Py_ssize_t
    col: cython.Py_ssize_t
    row_low: cython.Py_ssize_t
    row_high: cython.Py_ssize_t
    column_low: cython.Py_ssize_t
    column_high: cython.Py_ssize_t
    sx: cython.double
    sy: cython.double
    px: cython.double
    py: cython.double
    area: cython.double
    low: cython.double
    high: cython.double
    cx: cython.double
    cy: cython.double
    w0: cython.double
    w1: cython.double
    w2: cython.double
    l0: cython.double
    l1: cython.double
    l2: cython.double
    page_x: cython.double
    page_y: cython.double
    target_x: cython.double
    target_y: cython.double
    positive: cython.bint
    with cython.nogil:
        for t in range(triangles.shape[0]):
            for v in range(3):
                sx = triangles[t, v, 0]
                sy = triangles[t, v, 1]
                if mapped:
                    px = f[0] * sx + f[2] * sy + f[4]
                    py = f[1] * sx + f[3] * sy + f[5]
                else:
                    px = sx
                    py = sy
                xs[v] = (px - crop_x0) * scale
                ys[v] = (crop_y1 - py) * scale
            area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
            if area == 0.0 or not isfinite(area):
                continue
            positive = area > 0.0
            # The pixels whose centres (c + 0.5) can lie in the triangle's bounds.
            low = min(xs[0], min(xs[1], xs[2]))
            high = max(xs[0], max(xs[1], xs[2]))
            low = ceil(low - 0.5)
            high = floor(high - 0.5)
            if low < ix0:
                low = ix0
            if high > ix1 - 1:
                high = ix1 - 1
            if low > high:
                continue
            column_low = cython.cast(cython.Py_ssize_t, low)
            column_high = cython.cast(cython.Py_ssize_t, high)
            low = min(ys[0], min(ys[1], ys[2]))
            high = max(ys[0], max(ys[1], ys[2]))
            low = ceil(low - 0.5)
            high = floor(high - 0.5)
            if low < iy0:
                low = iy0
            if high > iy1 - 1:
                high = iy1 - 1
            if low > high:
                continue
            row_low = cython.cast(cython.Py_ssize_t, low)
            row_high = cython.cast(cython.Py_ssize_t, high)
            for row in range(row_low, row_high + 1):
                cy = cython.cast(cython.double, row) + 0.5
                r = row - iy0
                for column in range(column_low, column_high + 1):
                    cx = cython.cast(cython.double, column) + 0.5
                    w0 = (xs[1] - cx) * (ys[2] - cy) - (xs[2] - cx) * (ys[1] - cy)
                    w1 = (xs[2] - cx) * (ys[0] - cy) - (xs[0] - cx) * (ys[2] - cy)
                    w2 = (xs[0] - cx) * (ys[1] - cy) - (xs[1] - cx) * (ys[0] - cy)
                    if positive:
                        if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                            continue
                    elif w0 > 0.0 or w1 > 0.0 or w2 > 0.0:
                        continue
                    col = column - ix0
                    if not allowed[r, col]:
                        continue
                    if boxed:
                        page_x = crop_x0 + (cython.cast(cython.double, column) + 0.5) / scale
                        page_y = crop_y1 - (cython.cast(cython.double, row) + 0.5) / scale
                        if inverted:
                            target_x = m[0] * page_x + m[2] * page_y + m[4]
                            target_y = m[1] * page_x + m[3] * page_y + m[5]
                        else:
                            target_x = page_x
                            target_y = page_y
                        if not inside_box(
                            cython.cast(cython.p_const_double, box), target_x, target_y
                        ):
                            continue
                    l0 = w0 / area
                    l1 = w1 / area
                    l2 = w2 / area
                    for j in range(k):
                        values[r, col, j] = (
                            l0 * triangles[t, 0, 2 + j]
                            + l1 * triangles[t, 1, 2 + j]
                            + l2 * triangles[t, 2, 2 + j]
                        )
                    painted[r, col] = 1
    return values_array, painted_array


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def device_length(
    linear: cython.p_const_double, dx: cython.double, dy: cython.double
) -> cython.double:
    ex: cython.double = linear[0] * dx + linear[2] * dy
    ey: cython.double = linear[1] * dx + linear[3] * dy
    return sqrt(ex * ex + ey * ey)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def patch_divisions(
    patches: cython.const[cython.double][:, :, ::1],
    corners: cython.const[cython.double][:, :, ::1],
    p: cython.Py_ssize_t,
    linear: cython.p_const_double,
    tolerance: cython.double,
    color_tolerances: cython.const[cython.double][::1],
    max_divisions: cython.Py_ssize_t,
) -> cython.Py_ssize_t:
    """How many parts to divide each side of patch p into so that its triangles stay within
    tolerance device pixels of the surface, and within each value's colour tolerance of its
    bilinear interpolation."""
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    a: cython.Py_ssize_t
    b: cython.Py_ssize_t
    c: cython.Py_ssize_t
    e: cython.Py_ssize_t
    along_u: cython.double = 0.0
    along_v: cython.double = 0.0
    twist: cython.double = 0.0
    length: cython.double
    need: cython.double = 1.0
    bound: cython.double
    for j in range(4):
        for i in range(2):
            a = i * 4 + j
            b = (i + 1) * 4 + j
            c = (i + 2) * 4 + j
            length = device_length(
                linear,
                patches[p, a, 0] - 2.0 * patches[p, b, 0] + patches[p, c, 0],
                patches[p, a, 1] - 2.0 * patches[p, b, 1] + patches[p, c, 1],
            )
            if not length <= along_u:
                along_u = length
    for i in range(4):
        for j in range(2):
            a = i * 4 + j
            b = i * 4 + j + 1
            c = i * 4 + j + 2
            length = device_length(
                linear,
                patches[p, a, 0] - 2.0 * patches[p, b, 0] + patches[p, c, 0],
                patches[p, a, 1] - 2.0 * patches[p, b, 1] + patches[p, c, 1],
            )
            if not length <= along_v:
                along_v = length
    for i in range(3):
        for j in range(3):
            a = i * 4 + j
            b = (i + 1) * 4 + j
            c = i * 4 + j + 1
            e = (i + 1) * 4 + j + 1
            length = device_length(
                linear,
                patches[p, a, 0] - patches[p, b, 0] - patches[p, c, 0] + patches[p, e, 0],
                patches[p, a, 1] - patches[p, b, 1] - patches[p, c, 1] + patches[p, e, 1],
            )
            if not length <= twist:
                twist = length
    # Linear interpolation over a cell of side h strays at most about
    # h^2 / 8 * (|S_uu| + 2 |S_uv| + |S_vv|) from the surface S, where |S_uu| <= 6 * the
    # largest second difference along u, |S_vv| likewise and |S_uv| <= 9 * the largest twist.
    bound = (6.0 * along_u + 18.0 * twist + 6.0 * along_v) / 8.0
    if not bound <= 0.0:
        need = sqrt(bound / tolerance)
    for j in range(corners.shape[2]):
        if color_tolerances[j] > 0.0:
            # Gouraud over the two triangles of a cell strays at most |twist| / (4 n^2) from
            # the bilinear colour.
            length = fabs(corners[p, 0, j] - corners[p, 1, j] + corners[p, 2, j] - corners[p, 3, j])
            bound = sqrt(length / (4.0 * color_tolerances[j]))
            if not bound <= need:
                need = bound
    if not need <= cython.cast(cython.double, max_divisions):
        return max_divisions
    need = ceil(need)
    if need < 1.0:
        return 1
    return cython.cast(cython.Py_ssize_t, need)


def patch_triangles(
    patches: cython.const[cython.double][:, :, ::1],
    corners: cython.const[cython.double][:, :, ::1],
    linear,
    tolerance: cython.double,
    color_tolerances: cython.const[cython.double][::1],
    max_divisions: cython.Py_ssize_t,
):
    """The triangles (T, 3, 2 + k) that approximate each patch: patches is (P, 16, 2), the
    control points p_ij at i * 4 + j of a tensor-product patch (column i along u, row j
    along v), and corners (P, 4, k) the values at p00, p03, p33 and p30. linear (a, b, c, d)
    takes a difference in target space to device pixels. Each patch is divided into n by n
    cells (patch_divisions, at most max_divisions), the surface
    S(u, v) = sum p_ij B_i(u) B_j(v) taken at each cell corner with the values interpolated
    bilinearly, and each cell split into two triangles. The cells come in order of rising v,
    then rising u, so where a patch folds over itself the point with the larger v, then u,
    is painted last (8.7.4.5.7)."""
    count: cython.Py_ssize_t = patches.shape[0]
    if patches.shape[1] != 16 or patches.shape[2] != 2:
        raise ValueError("patches must be (P, 16, 2)")
    if corners.shape[0] != count or corners.shape[1] != 4:
        raise ValueError("corners must be (P, 4, k)")
    k: cython.Py_ssize_t = corners.shape[2]
    if color_tolerances.shape[0] != k:
        raise ValueError("color_tolerances must hold one number per value")
    if max_divisions < 1:
        raise ValueError("max_divisions must be at least 1")
    if not tolerance > 0.0:
        raise ValueError("tolerance must be positive")
    lin = cython.declare(cython.double[4])
    read_numbers(linear, 4, "linear", lin)
    divisions_array = numpy.zeros(count, dtype=numpy.intp)
    divisions: cython.Py_ssize_t[::1] = divisions_array
    p: cython.Py_ssize_t
    total: cython.Py_ssize_t = 0
    with cython.nogil:
        for p in range(count):
            divisions[p] = patch_divisions(
                patches,
                corners,
                p,
                cython.cast(cython.p_const_double, lin),
                tolerance,
                color_tolerances,
                max_divisions,
            )
            total += 2 * divisions[p] * divisions[p]
    width: cython.Py_ssize_t = 2 + k
    triangles_array = numpy.zeros((total, 3, width), dtype=numpy.float64)
    triangles: cython.double[:, :, ::1] = triangles_array
    grid_array = numpy.zeros(
        ((max_divisions + 1) * (max_divisions + 1), width), dtype=numpy.float64
    )
    grid: cython.double[:, ::1] = grid_array
    bu = cython.declare(cython.double[4])
    bv = cython.declare(cython.double[4])
    n: cython.Py_ssize_t
    a: cython.Py_ssize_t
    b: cython.Py_ssize_t
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    q: cython.Py_ssize_t
    g: cython.Py_ssize_t
    out: cython.Py_ssize_t = 0
    u: cython.double
    v: cython.double
    mu: cython.double
    mv: cython.double
    x: cython.double
    y: cython.double
    weight: cython.double
    corner = cython.declare(cython.Py_ssize_t[6])
    with cython.nogil:
        for p in range(count):
            n = divisions[p]
            for b in range(n + 1):
                v = cython.cast(cython.double, b) / cython.cast(cython.double, n)
                mv = 1.0 - v
                bv[0] = mv * mv * mv
                bv[1] = 3.0 * v * mv * mv
                bv[2] = 3.0 * v * v * mv
                bv[3] = v * v * v
                for a in range(n + 1):
                    u = cython.cast(cython.double, a) / cython.cast(cython.double, n)
                    mu = 1.0 - u
                    bu[0] = mu * mu * mu
                    bu[1] = 3.0 * u * mu * mu
                    bu[2] = 3.0 * u * u * mu
                    bu[3] = u * u * u
                    x = 0.0
                    y = 0.0
                    for i in range(4):
                        for j in range(4):
                            weight = bu[i] * bv[j]
                            x = x + patches[p, i * 4 + j, 0] * weight
                            y = y + patches[p, i * 4 + j, 1] * weight
                    g = b * (n + 1) + a
                    grid[g, 0] = x
                    grid[g, 1] = y
                    for j in range(k):
                        grid[g, 2 + j] = (
                            mu * mv * corners[p, 0, j]
                            + mu * v * corners[p, 1, j]
                            + u * v * corners[p, 2, j]
                            + u * mv * corners[p, 3, j]
                        )
            for b in range(n):
                for a in range(n):
                    corner[0] = b * (n + 1) + a
                    corner[1] = b * (n + 1) + a + 1
                    corner[2] = (b + 1) * (n + 1) + a + 1
                    corner[3] = b * (n + 1) + a
                    corner[4] = (b + 1) * (n + 1) + a + 1
                    corner[5] = (b + 1) * (n + 1) + a
                    for q in range(6):
                        for j in range(width):
                            triangles[out + q // 3, q % 3, j] = grid[corner[q], j]
                    out += 2
    return triangles_array
