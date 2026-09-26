# SPDX-License-Identifier: AGPL-3.0-only
"""A stroked path, painted whole (core_pdf.impl.render.target.stroke_path).

stroke_path walks each subpath in Python and paints it as primitives: a
fill_line per segment, a fill_join per interior point, caps at the ends.
Each primitive is a Python call that clips its box, picks a branch and
calls a kernel or numpy. A circuit schematic strokes 49,000 paths that way,
100,000 of them joins, and paint_stroke_once strokes every path of a group
the same way into scratch.

This is that walk and those primitives for the case most strokes take:
normal blending, no group planes, and no clip or a rectangular one, over a
deferred path's point columns. Each primitive keeps the branch its Python
takes, in the Python's arithmetic and order:

- fill_line: an axis-aligned or degenerate segment is a fill_rect; a box
  of more than 64 pixels is rasterize_unclipped_line_normal, its 4x4
  sample tests on the outer-sum bases and its float32 partial blend; a
  smaller one is stroke_segment_samples (``segment_samples``).
- fill_rect: a box whose edges are not pixel-aligned is
  fill_rect_coverage (``fill_rect_pixels``); an aligned opaque one is
  written; an aligned translucent one is blend_normal_solid_array_numpy,
  whose three regimes -- empty backdrop, opaque backdrop, general -- are
  chosen by scanning the box as numpy's any() and all() did.
- fill_circle: opaque, above 16 pixels its numpy inside test, below its
  loop; translucent, its blend_px loop.

A rectangular clip region's row spans are its pixel box, which holds every
box clipped to the region, so the clip masks the Python built for small
segments were all ones and are not needed. The paint window is a union of
boxes, so the kernel returns the union of the boxes each primitive would
have extended it by. Python's math.floor and ceil raise on a NaN or
infinite box edge; the kernel stops there with the same error for the
caller to raise, after the pixels and window of everything before it.

A two-point subpath whose points coincide, under a round cap, is a
fill_path of a circle in Python: the kernel stops before it and says
where, and the caller paints it and resumes after it.

setup.py builds with -ffp-contract=off. The square root of a segment's
length is ``**0.5`` in Python, libm's pow, so it is pow here with the
exponent passed in rather than written, which clang would turn into sqrt.
"""

from libc.math cimport ceil, fabs, floor, isinf, isnan, pow, rint, rintf
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

from core_pdf_cythonized._pixel_blend cimport blend_normal_pixel, coverage_alpha
from core_pdf_cythonized._rect cimport fill_rect_pixels
from core_pdf_cythonized._stroke_segment cimport segment_samples

__all__ = ("stroke_polylines",)

cdef enum:
    FAILED_NAN = 1
    FAILED_INFINITY = 2
    FAILED_MEMORY = 3


cdef struct Paint:
    unsigned char* pixels
    Py_ssize_t row_stride
    Py_ssize_t width
    Py_ssize_t height
    double crop_x0
    double crop_y1
    double scale
    bint clipped
    double clip[4]
    int red
    int green
    int blue
    int alpha
    double exponent
    bint painted
    Py_ssize_t window[4]
    int error


cdef inline double py_max(double a, double b) noexcept nogil:
    # Python's max(a, b): a unless b is greater.
    return b if b > a else a


cdef inline double py_min(double a, double b) noexcept nogil:
    return b if b < a else a


cdef inline void extend(Paint* p, Py_ssize_t y0, Py_ssize_t y1, Py_ssize_t x0, Py_ssize_t x1) noexcept nogil:
    if not p.painted:
        p.painted = True
        p.window[0] = y0
        p.window[1] = y1
        p.window[2] = x0
        p.window[3] = x1
        return
    if y0 < p.window[0]:
        p.window[0] = y0
    if y1 > p.window[1]:
        p.window[1] = y1
    if x0 < p.window[2]:
        p.window[2] = x0
    if x1 > p.window[3]:
        p.window[3] = x1


cdef inline bint checked(Paint* p, double value) noexcept nogil:
    # What math.floor and math.ceil refuse.
    if isnan(value):
        p.error = FAILED_NAN
        return False
    if isinf(value):
        p.error = FAILED_INFINITY
        return False
    return True


cdef inline Py_ssize_t clamped(double value, Py_ssize_t size) noexcept nogil:
    # size if v > size else max(v, 0), on an integral double.
    if value > <double> size:
        return size
    if value < 0.0:
        return 0
    return <Py_ssize_t> value


cdef int page_box_to_pixels(
    Paint* p, double x0, double y0, double x1, double y1, Py_ssize_t* out
) noexcept nogil:
    # ClipState.page_box_to_pixels: 1 with the box, 0 for None, -1 on error.
    cdef double value = (x0 - p.crop_x0) * p.scale
    if not checked(p, value):
        return -1
    out[0] = clamped(floor(value), p.width)
    value = (x1 - p.crop_x0) * p.scale
    if not checked(p, value):
        return -1
    out[2] = clamped(ceil(value), p.width)
    value = (p.crop_y1 - y1) * p.scale
    if not checked(p, value):
        return -1
    out[1] = clamped(floor(value), p.height)
    value = (p.crop_y1 - y0) * p.scale
    if not checked(p, value):
        return -1
    out[3] = clamped(ceil(value), p.height)
    if out[2] <= out[0] or out[3] <= out[1]:
        return 0
    return 1


cdef int clip_box(Paint* p, double* box) noexcept nogil:
    # intersect_box(box, region.box) in place: 0 when it is empty, else 1.
    if not p.clipped:
        return 1
    box[0] = py_max(box[0], p.clip[0])
    box[1] = py_max(box[1], p.clip[1])
    box[2] = py_min(box[2], p.clip[2])
    box[3] = py_min(box[3], p.clip[3])
    if box[2] <= box[0] or box[3] <= box[1]:
        return 0
    return 1


cdef void solid_blend(Paint* p, Py_ssize_t ix0, Py_ssize_t iy0, Py_ssize_t ix1, Py_ssize_t iy1) noexcept nogil:
    # blend_normal_solid_array_numpy for 0 < alpha < 255.
    cdef int sa = p.alpha
    if sa <= 0:
        return
    cdef Py_ssize_t y, x, c
    cdef unsigned char* pixel
    cdef bint any_alpha = False, all_opaque = True
    for y in range(iy0, iy1):
        for x in range(ix0, ix1):
            pixel = p.pixels + y * p.row_stride + x * 4
            if pixel[3]:
                any_alpha = True
            if pixel[3] != 255:
                all_opaque = False
    cdef int source[3]
    source[0] = p.red
    source[1] = p.green
    source[2] = p.blue
    if not any_alpha:
        for y in range(iy0, iy1):
            for x in range(ix0, ix1):
                pixel = p.pixels + y * p.row_stride + x * 4
                pixel[0] = <unsigned char> p.red
                pixel[1] = <unsigned char> p.green
                pixel[2] = <unsigned char> p.blue
                pixel[3] = <unsigned char> sa
        return
    cdef double source_alpha = sa / 255.0
    cdef double inverse_source_alpha = 1.0 - source_alpha
    # The Python floats meet float32 arrays, so each is narrowed first.
    cdef float scaled[3]
    for c in range(3):
        scaled[c] = <float> (source[c] * source_alpha)
    cdef float inverse = <float> inverse_source_alpha
    cdef float source_float = <float> source_alpha
    cdef float SCALE = 255.0
    cdef float destination_alpha, output_alpha, value
    if all_opaque:
        for y in range(iy0, iy1):
            for x in range(ix0, ix1):
                pixel = p.pixels + y * p.row_stride + x * 4
                for c in range(3):
                    value = rintf(scaled[c] + <float> pixel[c] * inverse)
                    pixel[c] = clip_float(value)
        return
    for y in range(iy0, iy1):
        for x in range(ix0, ix1):
            pixel = p.pixels + y * p.row_stride + x * 4
            destination_alpha = <float> pixel[3] / SCALE
            output_alpha = source_float + destination_alpha * inverse
            for c in range(3):
                value = (scaled[c] + <float> pixel[c] * destination_alpha * inverse) / output_alpha
                pixel[c] = clip_float(rintf(value))
            pixel[3] = clip_float(rintf(rintf(output_alpha * SCALE)))


cdef inline unsigned char clip_float(float value) noexcept nogil:
    # numpy.clip(v, 0, 255).astype(uint8) on a rounded float32.
    if value < 0.0:
        return 0
    if value > 255.0:
        return 255
    return <unsigned char> value


cdef int fill_rect(Paint* p, double x0, double y0, double x1, double y1) noexcept nogil:
    # RasterTarget.fill_rect, normal blend and no planes. -1 on error.
    cdef double box[4]
    box[0] = x0
    box[1] = y0
    box[2] = x1
    box[3] = y1
    if not clip_box(p, box):
        return 0
    cdef Py_ssize_t pixel_box[4]
    cdef int found = page_box_to_pixels(p, box[0], box[1], box[2], box[3], pixel_box)
    if found <= 0:
        return found
    cdef Py_ssize_t ix0 = pixel_box[0], iy0 = pixel_box[1], ix1 = pixel_box[2], iy1 = pixel_box[3]
    cdef double left = (box[0] - p.crop_x0) * p.scale
    cdef double right = (box[2] - p.crop_x0) * p.scale
    cdef double top = (p.crop_y1 - box[3]) * p.scale
    cdef double bottom = (p.crop_y1 - box[1]) * p.scale
    cdef Py_ssize_t y, x
    cdef unsigned char opaque[4]
    if not (
        left <= ix0 + 1e-9
        and right >= ix1 - 1e-9
        and top <= iy0 + 1e-9
        and bottom >= iy1 - 1e-9
    ):
        if fill_rect_pixels(
            p.pixels + iy0 * p.row_stride + ix0 * 4, p.row_stride, ix0, ix1, iy0, iy1,
            left, right, top, bottom, p.red, p.green, p.blue, p.alpha,
        ) < 0:
            p.error = FAILED_MEMORY
            return -1
    elif p.alpha == 255:
        opaque[0] = <unsigned char> p.red
        opaque[1] = <unsigned char> p.green
        opaque[2] = <unsigned char> p.blue
        opaque[3] = 255
        for y in range(iy0, iy1):
            for x in range(ix0, ix1):
                memcpy(p.pixels + y * p.row_stride + x * 4, opaque, 4)
    else:
        solid_blend(p, ix0, iy0, ix1, iy1)
    extend(p, iy0, iy1, ix0, ix1)
    return 0


cdef int fill_circle(Paint* p, double cx, double cy, double radius) noexcept nogil:
    # RasterTarget.fill_circle, normal blend and no planes. -1 on error.
    cdef double box[4]
    box[0] = cx - radius
    box[1] = cy - radius
    box[2] = cx + radius
    box[3] = cy + radius
    if not clip_box(p, box):
        return 0
    cdef Py_ssize_t pixel_box[4]
    cdef int found = page_box_to_pixels(p, box[0], box[1], box[2], box[3], pixel_box)
    if found <= 0:
        return found
    cdef Py_ssize_t ix0 = pixel_box[0], iy0 = pixel_box[1], ix1 = pixel_box[2], iy1 = pixel_box[3]
    cdef double radius2 = radius * radius
    cdef Py_ssize_t px, py
    cdef double page_x, page_y, dx, dy
    cdef unsigned char* pixel
    cdef unsigned char opaque[4]
    opaque[0] = <unsigned char> p.red
    opaque[1] = <unsigned char> p.green
    opaque[2] = <unsigned char> p.blue
    opaque[3] = 255
    if p.alpha >= 255 and (ix1 - ix0) * (iy1 - iy0) > 16:
        # The numpy inside test: (xs - cx) ** 2 + (ys - cy) ** 2 <= r2.
        for py in range(iy0, iy1):
            page_y = p.crop_y1 - (<double> py + 0.5) / p.scale
            dy = page_y - cy
            for px in range(ix0, ix1):
                page_x = p.crop_x0 + (<double> px + 0.5) / p.scale
                dx = page_x - cx
                if dx * dx + dy * dy <= radius2:
                    memcpy(p.pixels + py * p.row_stride + px * 4, opaque, 4)
        extend(p, iy0, iy1, ix0, ix1)
        return 0
    for py in range(iy0, iy1):
        page_y = p.crop_y1 - (<double> py + 0.5) / p.scale
        for px in range(ix0, ix1):
            page_x = p.crop_x0 + (<double> px + 0.5) / p.scale
            dx = page_x - cx
            dy = page_y - cy
            if dx * dx + dy * dy > radius2:
                continue
            extend(p, py, py + 1, px, px + 1)
            pixel = p.pixels + py * p.row_stride + px * 4
            if p.alpha >= 255:
                memcpy(pixel, opaque, 4)
            elif p.alpha > 0:
                blend_normal_pixel(pixel, p.red, p.green, p.blue, p.alpha)
    return 0


cdef int line_raster(
    Paint* p,
    double x0,
    double y0,
    double x1,
    double y1,
    double line_width,
    Py_ssize_t ix0,
    Py_ssize_t iy0,
    Py_ssize_t ix1,
    Py_ssize_t iy1,
) noexcept nogil:
    # rasterize_unclipped_line_normal with a butt cap and no planes.
    cdef double x_delta = x1 - x0
    cdef double y_delta = y1 - y0
    cdef double segment_length_squared = x_delta * x_delta + y_delta * y_delta
    if segment_length_squared <= 1e-12:
        return 0
    cdef double segment_length = pow(segment_length_squared, p.exponent)
    cdef double half = py_max(0.5 / p.scale, line_width * 0.5)
    cdef double cap_extension = 0.0
    cdef Py_ssize_t width = ix1 - ix0, height = iy1 - iy0
    cdef double* x_offset = <double*> malloc((width + height) * sizeof(double))
    if x_offset == NULL:
        p.error = FAILED_MEMORY
        return -1
    cdef double* y_offset = x_offset + width
    cdef unsigned char* covered = <unsigned char*> malloc(width * height)
    if covered == NULL:
        free(x_offset)
        p.error = FAILED_MEMORY
        return -1
    cdef Py_ssize_t i, j, sx, sy, c
    for j in range(width):
        x_offset[j] = (p.crop_x0 + (<double> (ix0 + j) + 0.125) / p.scale) - x0
    for i in range(height):
        y_offset[i] = (p.crop_y1 - (<double> (iy0 + i) + 0.125) / p.scale) - y0
    cdef double sample_step = 1.0 / (4.0 * p.scale)
    cdef double cross_limit = half * segment_length
    cdef double projection_extension = cap_extension * segment_length
    cdef double low_projection[16]
    cdef double high_projection[16]
    cdef double low_cross[16]
    cdef double high_cross[16]
    cdef double projection_shift, cross_shift
    for sy in range(4):
        for sx in range(4):
            projection_shift = sample_step * (<double> sx * x_delta - <double> sy * y_delta)
            low_projection[4 * sy + sx] = -projection_extension - projection_shift
            high_projection[4 * sy + sx] = (
                segment_length_squared + projection_extension - projection_shift
            )
            cross_shift = sample_step * (<double> sx * y_delta + <double> sy * x_delta)
            low_cross[4 * sy + sx] = -cross_limit - cross_shift
            high_cross[4 * sy + sx] = cross_limit - cross_shift
    cdef double projection_base, cross_base
    cdef int count
    cdef bint any_covered = False
    for i in range(height):
        for j in range(width):
            projection_base = y_offset[i] * y_delta + x_offset[j] * x_delta
            cross_base = -y_offset[i] * x_delta + x_offset[j] * y_delta
            count = 0
            for c in range(16):
                if (
                    projection_base >= low_projection[c]
                    and projection_base <= high_projection[c]
                    and cross_base >= low_cross[c]
                    and cross_base <= high_cross[c]
                ):
                    count += 1
            covered[i * width + j] = <unsigned char> count
            if count:
                any_covered = True
    cdef int alpha
    cdef unsigned char* pixel
    cdef float SCALE = 255.0
    cdef float ONE = 1.0
    cdef float source_fraction, destination_alpha, remaining, output_alpha, safe
    cdef float colour[3]
    colour[0] = <float> p.red
    colour[1] = <float> p.green
    colour[2] = <float> p.blue
    if any_covered:
        for i in range(height):
            for j in range(width):
                alpha = coverage_alpha(p.alpha, covered[i * width + j])
                if alpha <= 0:
                    continue
                pixel = p.pixels + (iy0 + i) * p.row_stride + (ix0 + j) * 4
                if alpha >= 255:
                    pixel[0] = <unsigned char> p.red
                    pixel[1] = <unsigned char> p.green
                    pixel[2] = <unsigned char> p.blue
                    pixel[3] = 255
                    continue
                # The float32 partial blend, as numpy evaluated it.
                source_fraction = <float> alpha / SCALE
                destination_alpha = <float> pixel[3] / SCALE
                remaining = ONE - source_fraction
                output_alpha = source_fraction + destination_alpha * remaining
                safe = output_alpha if output_alpha > 0.0 else ONE
                for c in range(3):
                    pixel[c] = clip_float(
                        rintf(
                            (colour[c] * source_fraction + <float> pixel[c] * destination_alpha * remaining)
                            / safe
                        )
                    )
                pixel[3] = clip_float(rintf(output_alpha * SCALE))
    free(covered)
    free(x_offset)
    return 0


cdef int fill_line(Paint* p, double x0, double y0, double x1, double y1, double line_width) noexcept nogil:
    # RasterTarget.fill_line with a butt cap, normal blend and no planes.
    cdef double dx = x1 - x0
    cdef double dy = y1 - y0
    cdef double half
    if fabs(dx) <= 1e-12 or fabs(dy) <= 1e-12:
        half = py_max(0.5 / p.scale, line_width * 0.5)
        if fabs(dy) <= 1e-12:
            return fill_rect(p, py_min(x0, x1) - 0.0, y0 - half, py_max(x0, x1) + 0.0, y0 + half)
        return fill_rect(p, x0 - half, py_min(y0, y1) - 0.0, x0 + half, py_max(y0, y1) + 0.0)
    cdef double seg_len2 = dx * dx + dy * dy
    half = py_max(0.5 / p.scale, line_width * 0.5)
    if seg_len2 <= 1e-12:
        return fill_rect(p, x0 - half, y0 - half, x0 + half, y0 + half)
    cdef double seg_len = pow(seg_len2, p.exponent)
    cdef double ux = dx / seg_len
    cdef double uy = dy / seg_len
    cdef double cap_extension = 0.0
    cdef double box[4]
    box[0] = py_min(x0, x1) - half - fabs(ux) * cap_extension
    box[1] = py_min(y0, y1) - half - fabs(uy) * cap_extension
    box[2] = py_max(x0, x1) + half + fabs(ux) * cap_extension
    box[3] = py_max(y0, y1) + half + fabs(uy) * cap_extension
    if not clip_box(p, box):
        return 0
    cdef Py_ssize_t pixel_box[4]
    cdef int found = page_box_to_pixels(p, box[0], box[1], box[2], box[3], pixel_box)
    if found <= 0:
        return found
    cdef Py_ssize_t ix0 = pixel_box[0], iy0 = pixel_box[1], ix1 = pixel_box[2], iy1 = pixel_box[3]
    cdef double half2 = half * half
    cdef double inv_seg_len2 = 1.0 / seg_len2
    cdef double projection_extension = cap_extension * seg_len
    if (ix1 - ix0) * (iy1 - iy0) > 64:
        if line_raster(p, x0, y0, x1, y1, line_width, ix0, iy0, ix1, iy1) < 0:
            return -1
        extend(p, iy0, iy1, ix0, ix1)
        return 0
    cdef Py_ssize_t covered_box[4]
    if segment_samples(
        p.pixels + iy0 * p.row_stride + ix0 * 4, p.row_stride, ix0, iy0, ix1, iy1,
        p.crop_x0, p.crop_y1, p.scale, x0, y0, x1, y1, dx, dy, seg_len2, inv_seg_len2,
        half2, projection_extension, False, p.red, p.green, p.blue, p.alpha,
        NULL, NULL, covered_box,
    ):
        extend(p, covered_box[1], covered_box[3], covered_box[0], covered_box[2])
    return 0


cdef int fill_terminal(Paint* p, double x, double y, double line_width, bint round_shape) noexcept nogil:
    cdef double radius = py_max(0.5 / p.scale, line_width * 0.5)
    if round_shape:
        return fill_circle(p, x, y, radius)
    return fill_rect(p, x - radius, y - radius, x + radius, y + radius)


cdef int fill_cap(Paint* p, double x, double y, double line_width, long line_cap) noexcept nogil:
    if line_cap == 0:
        return 0
    return fill_terminal(p, x, y, line_width, line_cap == 1)


def stroke_polylines(
    unsigned char[:, :, ::1] pixels,
    const double[::1] xs,
    const double[::1] ys,
    list spans,
    bint outline,
    double crop_x0,
    double crop_y1,
    double scale,
    clip,
    double line_width,
    int red,
    int green,
    int blue,
    int alpha,
    long line_cap,
    long line_join,
    Py_ssize_t first,
    double exponent,
):
    """Stroke the subpaths spans[first:] as stroke_path does, into `pixels`.

    ``clip`` is the rectangular clip region's page box, or None. Returns
    (window, stopped, error): the (y0, y1, x0, x1) union of the paint-window
    boxes, or None; the index of a coincident two-point subpath under a
    round cap, left for the caller, or len(spans); and 0, or the error the
    Python would have raised there -- 1 a NaN and 2 an infinity reaching
    floor or ceil, 3 out of memory.
    """
    if pixels.shape[2] != 4:
        raise ValueError("pixels must be RGBA")
    if xs.shape[0] != ys.shape[0]:
        raise ValueError("xs and ys differ in length")
    cdef Py_ssize_t count = len(spans)
    cdef Py_ssize_t* bounds = <Py_ssize_t*> malloc((3 * count + 1) * sizeof(Py_ssize_t))
    if bounds == NULL:
        raise MemoryError
    cdef Py_ssize_t k, start, end
    try:
        for k in range(count):
            start, end, flag = spans[k]
            if start < 0 or end > xs.shape[0] or end < start:
                raise ValueError("span runs past the points")
            bounds[3 * k] = start
            bounds[3 * k + 1] = end
            bounds[3 * k + 2] = 1 if (outline or flag) else 0
        return run(pixels, xs, ys, bounds, count, crop_x0, crop_y1, scale, clip, line_width,
                   red, green, blue, alpha, line_cap, line_join, first, exponent)
    finally:
        free(bounds)


cdef run(
    unsigned char[:, :, ::1] pixels,
    const double[::1] xs,
    const double[::1] ys,
    Py_ssize_t* bounds,
    Py_ssize_t count,
    double crop_x0,
    double crop_y1,
    double scale,
    clip,
    double line_width,
    int red,
    int green,
    int blue,
    int alpha,
    long line_cap,
    long line_join,
    Py_ssize_t first,
    double exponent,
):
    cdef Paint p
    p.pixels = &pixels[0, 0, 0] if pixels.shape[0] and pixels.shape[1] else NULL
    p.row_stride = pixels.strides[0]
    p.width = pixels.shape[1]
    p.height = pixels.shape[0]
    p.crop_x0 = crop_x0
    p.crop_y1 = crop_y1
    p.scale = scale
    p.clipped = clip is not None
    if p.clipped:
        p.clip[0] = clip[0]
        p.clip[1] = clip[1]
        p.clip[2] = clip[2]
        p.clip[3] = clip[3]
    p.red = red
    p.green = green
    p.blue = blue
    p.alpha = alpha
    p.exponent = exponent
    p.painted = False
    p.error = 0
    cdef Py_ssize_t k, start, n, index
    cdef bint closed
    cdef Py_ssize_t stopped = count
    cdef double x0, y0, x1, y1
    with nogil:
        for k in range(first, count):
            start = bounds[3 * k]
            n = bounds[3 * k + 1] - start
            closed = bounds[3 * k + 2]
            if n < 2:
                continue
            if n == 2 and not closed:
                x0 = xs[start]
                y0 = ys[start]
                x1 = xs[start + 1]
                y1 = ys[start + 1]
                if x0 == x1 and y0 == y1:
                    if line_cap == 1:
                        stopped = k
                        break
                    continue
                if fill_line(&p, x0, y0, x1, y1, line_width) < 0:
                    break
                if line_cap != 0:
                    if fill_cap(&p, x0, y0, line_width, line_cap) < 0:
                        break
                    if fill_cap(&p, x1, y1, line_width, line_cap) < 0:
                        break
                continue
            for index in range(start, start + n - 1):
                if fill_line(&p, xs[index], ys[index], xs[index + 1], ys[index + 1], line_width) < 0:
                    break
            if p.error:
                break
            x0 = xs[start]
            y0 = ys[start]
            x1 = xs[start + n - 1]
            y1 = ys[start + n - 1]
            if closed and (x0 != x1 or y0 != y1):
                if fill_line(&p, x1, y1, x0, y0, line_width) < 0:
                    break
            for index in range(start + 1, start + n - 1):
                if fill_terminal(&p, xs[index], ys[index], line_width, line_join == 1) < 0:
                    break
            if p.error:
                break
            if closed:
                if fill_terminal(&p, x0, y0, line_width, line_join == 1) < 0:
                    break
            elif line_cap != 0:
                if fill_cap(&p, x0, y0, line_width, line_cap) < 0:
                    break
                if fill_cap(&p, x1, y1, line_width, line_cap) < 0:
                    break
    window = (p.window[0], p.window[1], p.window[2], p.window[3]) if p.painted else None
    return window, stopped, p.error
