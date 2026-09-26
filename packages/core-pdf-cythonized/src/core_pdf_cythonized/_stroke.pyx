# SPDX-License-Identifier: AGPL-3.0-only
"""A stroked path, painted whole (core_pdf.impl.render.target.stroke_path).

stroke_path walked each subpath in Python and painted it as primitives: a
fill_line per segment, a fill_join per interior point, caps at the ends. A
circuit schematic strokes 49,000 paths that way, 100,000 of them joins, and
paint_stroke_once strokes every path of a group the same way into scratch.

The walk is here now, and only here: which primitives a subpath takes, in
what order, is decided by stroke_polylines for every stroke. Under normal
blending with no group planes -- nearly every stroke, and every one
paint_stroke_once makes -- the primitives are painted here too. Each keeps
the branch its Python takes, in the Python's arithmetic and order:

- fill_line: an axis-aligned or degenerate segment is a fill_rect; under no
  clip or a rectangular one, a box of more than 64 pixels is
  rasterize_unclipped_line_normal, its 4x4 sample tests on the outer-sum
  bases and its float32 partial blend; otherwise stroke_segment_samples
  (``segment_samples``), masked by the clip's rows as clip_pixel_mask made
  the mask, a bisect into each row's spans.
- fill_rect: under no clip or a rectangular one, a box whose edges are not
  pixel-aligned is fill_rect_coverage (``fill_rect_pixels``), an aligned
  opaque one is written, an aligned translucent one is
  blend_normal_solid_array_numpy, whose three regimes -- empty backdrop,
  opaque backdrop, general -- are chosen by scanning the box as numpy's any()
  and all() did. Under a clip path, the box's whole pixels in each row's
  spans: a span of 32 or more through blend_normal_solid_array_numpy, a
  shorter one pixel by pixel through blend_px.
- fill_circle: opaque under no clip or a rectangular one, above 16 pixels
  its numpy inside test and below its loop; otherwise its blend_px loop
  over the rows' spans.

A rectangular clip region's row spans are its pixel box, which holds every
box clipped to the region, so there the masks and spans are the box itself.
Under any other blend mode or with group planes, each primitive is the
Python one, called back with the same arguments.

The paint window is a union of boxes, so the kernel gathers the boxes each
primitive would have extended it by and hands the union to ``extend``
before anything that could raise, and at the end. Python's math.floor and
ceil raise on a NaN or infinite box edge; the kernel raises the same error
at the same point. A coincident two-point subpath under a round cap is a
fill_path of a circle, called back as ``dot``.

A built path's subpaths come as columns, with whether each one's ends
differ and, for two points, whether they coincide, worked out as Python's
tuple comparison worked them out -- which, for a float object that appears
twice, is identity first. A deferred path's points are fresh floats, so
there the kernel compares values.

setup.py builds with -ffp-contract=off. The square root of a segment's
length is ``**0.5`` in Python, libm's pow, so it is pow here with the
exponent passed in rather than written, which clang would turn into sqrt.
"""

from libc.math cimport ceil, fabs, floor, isinf, isnan, pow, rintf
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

from core_pdf_cythonized._byte_clamp cimport float_to_byte
from core_pdf_cythonized._pixel_blend cimport blend_normal_pixel, coverage_alpha
from core_pdf_cythonized._pymath cimport py_max, py_min, raise_not_integral
from core_pdf_cythonized._rect cimport fill_rect_pixels
from core_pdf_cythonized._stroke_segment cimport segment_samples

__all__ = ("stroke_polylines",)

cdef enum:
    FAILED_NAN = 1
    FAILED_INFINITY = 2
    FAILED_MEMORY = 3

cdef enum:
    CLIP_NONE = 0
    CLIP_RECT = 1
    CLIP_ROWS = 2


cdef struct Paint:
    unsigned char* pixels
    Py_ssize_t row_stride
    Py_ssize_t width
    Py_ssize_t height
    double crop_x0
    double crop_y1
    double scale
    int clip_mode
    bint clip_empty
    double clip[4]
    Py_ssize_t rows_origin
    Py_ssize_t rows_count
    const long long* row_offsets
    const long long* row_spans
    int red
    int green
    int blue
    int alpha
    double exponent
    bint painted
    Py_ssize_t window[4]
    int error


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
    if p.clip_mode == CLIP_NONE:
        return 1
    box[0] = py_max(box[0], p.clip[0])
    box[1] = py_max(box[1], p.clip[1])
    box[2] = py_min(box[2], p.clip[2])
    box[3] = py_min(box[3], p.clip[3])
    if box[2] <= box[0] or box[3] <= box[1]:
        return 0
    return 1


cdef int clipped_pixel_box(Paint* p, double* box, Py_ssize_t* pixel_box) noexcept nogil:
    # ClipState.clipped_pixel_box: 1 with both boxes, 0 for None, -1 on error.
    if p.clip_mode != CLIP_NONE and p.clip_empty:
        return 0
    if not clip_box(p, box):
        return 0
    return page_box_to_pixels(p, box[0], box[1], box[2], box[3], pixel_box)


cdef inline Py_ssize_t row_spans(Paint* p, Py_ssize_t py, const long long** spans) noexcept nogil:
    # clip_row_visible_spans for a clip path: the row's (start, end) pairs.
    if py < 0 or py >= p.height:
        return 0
    cdef Py_ssize_t row = py - p.rows_origin
    if row < 0 or row >= p.rows_count:
        return 0
    spans[0] = p.row_spans + 2 * p.row_offsets[row]
    return <Py_ssize_t> (p.row_offsets[row + 1] - p.row_offsets[row])


cdef inline bint pixel_in_row(const long long* spans, Py_ssize_t count, Py_ssize_t px) noexcept nogil:
    # clip_pixel_mask's test: bisect_left(spans, (px + 1, -1)), then the span before.
    cdef Py_ssize_t low = 0, high = count, middle
    while low < high:
        middle = (low + high) // 2
        if spans[2 * middle] < px + 1 or (spans[2 * middle] == px + 1 and spans[2 * middle + 1] < -1):
            low = middle + 1
        else:
            high = middle
    return low > 0 and spans[2 * (low - 1)] <= px < spans[2 * (low - 1) + 1]


cdef inline void blend_pixel(Paint* p, Py_ssize_t py, Py_ssize_t px) noexcept nogil:
    # blend_px in normal mode with no planes: the window, then the paint.
    extend(p, py, py + 1, px, px + 1)
    if p.alpha <= 0:
        return
    blend_normal_pixel(p.pixels + py * p.row_stride + px * 4, p.red, p.green, p.blue, p.alpha)


cdef void solid_blend(Paint* p, Py_ssize_t ix0, Py_ssize_t iy0, Py_ssize_t ix1, Py_ssize_t iy1) noexcept nogil:
    # blend_normal_solid_array_numpy over the box.
    cdef int sa = p.alpha
    if sa <= 0 or ix1 <= ix0 or iy1 <= iy0:
        return
    cdef Py_ssize_t y, x, c
    cdef unsigned char* pixel
    cdef unsigned char opaque[4]
    if sa >= 255:
        opaque[0] = <unsigned char> p.red
        opaque[1] = <unsigned char> p.green
        opaque[2] = <unsigned char> p.blue
        opaque[3] = <unsigned char> sa
        for y in range(iy0, iy1):
            for x in range(ix0, ix1):
                memcpy(p.pixels + y * p.row_stride + x * 4, opaque, 4)
        return
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
                    pixel[c] = float_to_byte(value)
        return
    for y in range(iy0, iy1):
        for x in range(ix0, ix1):
            pixel = p.pixels + y * p.row_stride + x * 4
            destination_alpha = <float> pixel[3] / SCALE
            output_alpha = source_float + destination_alpha * inverse
            for c in range(3):
                value = (scaled[c] + <float> pixel[c] * destination_alpha * inverse) / output_alpha
                pixel[c] = float_to_byte(rintf(value))
            pixel[3] = float_to_byte(rintf(rintf(output_alpha * SCALE)))


cdef int fill_rect(Paint* p, double x0, double y0, double x1, double y1) noexcept nogil:
    # RasterTarget.fill_rect, normal blend and no planes. -1 on error.
    cdef double box[4]
    box[0] = x0
    box[1] = y0
    box[2] = x1
    box[3] = y1
    cdef Py_ssize_t pixel_box[4]
    cdef int found = clipped_pixel_box(p, box, pixel_box)
    if found <= 0:
        return found
    cdef Py_ssize_t ix0 = pixel_box[0], iy0 = pixel_box[1], ix1 = pixel_box[2], iy1 = pixel_box[3]
    cdef Py_ssize_t y, x, k, count, start, end
    cdef const long long* spans = NULL
    if p.clip_mode == CLIP_ROWS:
        # Not rectangular: whole pixels, row by row, span by span.
        for y in range(iy0, iy1):
            count = row_spans(p, y, &spans)
            for k in range(count):
                start = <Py_ssize_t> spans[2 * k]
                end = <Py_ssize_t> spans[2 * k + 1]
                if start < ix0:
                    start = ix0
                if end > ix1:
                    end = ix1
                if end <= start:
                    continue
                if end - start >= 32:
                    solid_blend(p, start, y, end, y + 1)
                    extend(p, y, y + 1, start, end)
                else:
                    for x in range(start, end):
                        blend_pixel(p, y, x)
        return 0
    cdef double left = (box[0] - p.crop_x0) * p.scale
    cdef double right = (box[2] - p.crop_x0) * p.scale
    cdef double top = (p.crop_y1 - box[3]) * p.scale
    cdef double bottom = (p.crop_y1 - box[1]) * p.scale
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
    cdef Py_ssize_t px, py, k, count, start, end
    cdef const long long* spans = NULL
    cdef double page_x, page_y, dx, dy
    cdef unsigned char opaque[4]
    opaque[0] = <unsigned char> p.red
    opaque[1] = <unsigned char> p.green
    opaque[2] = <unsigned char> p.blue
    opaque[3] = 255
    if p.clip_mode == CLIP_ROWS:
        for py in range(iy0, iy1):
            page_y = p.crop_y1 - (<double> py + 0.5) / p.scale
            count = row_spans(p, py, &spans)
            for k in range(count):
                start = <Py_ssize_t> spans[2 * k]
                end = <Py_ssize_t> spans[2 * k + 1]
                if start < ix0:
                    start = ix0
                if end > ix1:
                    end = ix1
                for px in range(start, end):
                    page_x = p.crop_x0 + (<double> px + 0.5) / p.scale
                    dx = page_x - cx
                    dy = page_y - cy
                    if dx * dx + dy * dy > radius2:
                        continue
                    blend_pixel(p, py, px)
        return 0
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
            blend_pixel(p, py, px)
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
                    pixel[c] = float_to_byte(
                        rintf(
                            (colour[c] * source_fraction + <float> pixel[c] * destination_alpha * remaining)
                            / safe
                        )
                    )
                pixel[3] = float_to_byte(rintf(output_alpha * SCALE))
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
    cdef Py_ssize_t pixel_box[4]
    cdef int found = clipped_pixel_box(p, box, pixel_box)
    if found <= 0:
        return found
    cdef Py_ssize_t ix0 = pixel_box[0], iy0 = pixel_box[1], ix1 = pixel_box[2], iy1 = pixel_box[3]
    cdef double half2 = half * half
    cdef double projection_extension = cap_extension * seg_len
    if p.clip_mode != CLIP_ROWS and (ix1 - ix0) * (iy1 - iy0) > 64:
        if line_raster(p, x0, y0, x1, y1, line_width, ix0, iy0, ix1, iy1) < 0:
            return -1
        extend(p, iy0, iy1, ix0, ix1)
        return 0
    cdef unsigned char* allowed = NULL
    cdef Py_ssize_t box_width = ix1 - ix0, py, px, count
    cdef const long long* spans = NULL
    if p.clip_mode == CLIP_ROWS:
        allowed = <unsigned char*> malloc(box_width * (iy1 - iy0))
        if allowed == NULL:
            p.error = FAILED_MEMORY
            return -1
        for py in range(iy0, iy1):
            count = row_spans(p, py, &spans)
            for px in range(ix0, ix1):
                allowed[(py - iy0) * box_width + (px - ix0)] = pixel_in_row(spans, count, px)
    cdef Py_ssize_t covered_box[4]
    if segment_samples(
        p.pixels + iy0 * p.row_stride + ix0 * 4, p.row_stride, ix0, iy0, ix1, iy1,
        p.crop_x0, p.crop_y1, p.scale, x0, y0, dx, dy, seg_len2,
        half2, projection_extension, p.red, p.green, p.blue, p.alpha,
        allowed, NULL, covered_box,
    ):
        extend(p, covered_box[1], covered_box[3], covered_box[0], covered_box[2])
    free(allowed)
    return 0


cdef int fill_terminal(Paint* p, double x, double y, double line_width, bint round_shape) noexcept nogil:
    cdef double radius = py_max(0.5 / p.scale, line_width * 0.5)
    if round_shape:
        return fill_circle(p, x, y, radius)
    return fill_rect(p, x - radius, y - radius, x + radius, y + radius)


cdef struct Walk:
    const double* xs
    const double* ys
    const long long* bounds
    const unsigned char* ends_differ
    const unsigned char* coincident
    Py_ssize_t count
    double line_width
    bint native
    bint cap_nonzero
    bint cap_butt
    bint cap_round
    bint join_round


cdef class Callbacks:
    cdef object line
    cdef object join
    cdef object cap
    cdef object dot
    cdef object extend


cdef int flush(Paint* p, Callbacks calls) except -1:
    # Hand the gathered paint-window boxes to the target.
    if p.painted:
        p.painted = False
        calls.extend(p.window[0], p.window[1], p.window[2], p.window[3])
    return 0


cdef int failed(Paint* p, Callbacks calls) except -1:
    flush(p, calls)
    if p.error == FAILED_NAN or p.error == FAILED_INFINITY:
        return raise_not_integral(p.error == FAILED_NAN)
    raise MemoryError


cdef int line(Paint* p, Walk* w, Callbacks calls, double x0, double y0, double x1, double y1) except -1 nogil:
    if w.native:
        if fill_line(p, x0, y0, x1, y1, w.line_width) < 0:
            with gil:
                failed(p, calls)
        return 0
    with gil:
        calls.line(x0, y0, x1, y1)
    return 0


cdef int join(Paint* p, Walk* w, Callbacks calls, double x, double y) except -1 nogil:
    if w.native:
        if fill_terminal(p, x, y, w.line_width, w.join_round) < 0:
            with gil:
                failed(p, calls)
        return 0
    with gil:
        calls.join(x, y)
    return 0


cdef int cap(Paint* p, Walk* w, Callbacks calls, double x, double y) except -1 nogil:
    if w.native:
        if w.cap_butt:
            return 0
        if fill_terminal(p, x, y, w.line_width, w.cap_round) < 0:
            with gil:
                failed(p, calls)
        return 0
    with gil:
        calls.cap(x, y)
    return 0


cdef int walk(Paint* p, Walk* w, Callbacks calls) except -1 nogil:
    # stroke_path's loop over the subpaths.
    cdef Py_ssize_t k, start, n, index
    cdef bint closed, same
    cdef double x0, y0, x1, y1
    for k in range(w.count):
        start = w.bounds[3 * k]
        n = w.bounds[3 * k + 1] - start
        closed = w.bounds[3 * k + 2]
        if n < 2:
            continue
        x0 = w.xs[start]
        y0 = w.ys[start]
        x1 = w.xs[start + n - 1]
        y1 = w.ys[start + n - 1]
        if n == 2 and not closed:
            if w.coincident != NULL:
                same = w.coincident[k] != 0
            else:
                same = x0 == x1 and y0 == y1
            if same:
                if w.cap_round:
                    with gil:
                        flush(p, calls)
                        calls.dot(x0, y0)
                continue
            line(p, w, calls, x0, y0, x1, y1)
            if w.cap_nonzero:
                cap(p, w, calls, x0, y0)
                cap(p, w, calls, x1, y1)
            continue
        for index in range(start, start + n - 1):
            line(p, w, calls, w.xs[index], w.ys[index], w.xs[index + 1], w.ys[index + 1])
        if w.ends_differ != NULL:
            same = not w.ends_differ[k]
        else:
            same = not (x0 != x1 or y0 != y1)
        if closed and not same:
            line(p, w, calls, x1, y1, x0, y0)
        for index in range(start + 1, start + n - 1):
            join(p, w, calls, w.xs[index], w.ys[index])
        if closed:
            join(p, w, calls, x0, y0)
        elif w.cap_nonzero:
            cap(p, w, calls, x0, y0)
            cap(p, w, calls, x1, y1)
    return 0


def stroke_polylines(
    unsigned char[:, :, ::1] pixels,
    const double[::1] xs,
    const double[::1] ys,
    list spans,
    bint outline,
    const unsigned char[::1] ends_differ,
    const unsigned char[::1] coincident,
    double crop_x0,
    double crop_y1,
    double scale,
    int clip_mode,
    clip_box,
    bint clip_empty,
    Py_ssize_t rows_origin,
    const long long[::1] row_offsets,
    const long long[::1] row_spans,
    double line_width,
    int red,
    int green,
    int blue,
    int alpha,
    bint native,
    bint cap_nonzero,
    bint cap_butt,
    bint cap_round,
    bint join_round,
    double exponent,
    on_line,
    on_join,
    on_cap,
    on_dot,
    on_extend,
):
    """Stroke the subpaths `spans` cut from the columns, as stroke_path does.

    ``ends_differ`` and ``coincident``, one byte per span or None, answer a
    built path's point comparisons; None compares the columns. ``clip_mode``
    is 0 for no clip, 1 for a rectangular region and 2 for a clip path, with
    the region's page box, whether it is empty, and for a clip path its rows
    from ``rows_origin``: row r's spans are the pairs row_spans[2 *
    row_offsets[r]:2 * row_offsets[r + 1]]. With ``native``, normal blending
    and no group planes, the primitives are painted here; otherwise each is
    ``on_line(x0, y0, x1, y1)``, ``on_join(x, y)`` or ``on_cap(x, y)``.
    ``on_dot(x, y)`` paints a coincident two-point subpath under a round cap,
    and ``on_extend(y0, y1, x0, x1)`` takes the paint window's boxes.
    """
    if pixels.shape[2] != 4:
        raise ValueError("pixels must be RGBA")
    if xs.shape[0] != ys.shape[0]:
        raise ValueError("xs and ys differ in length")
    cdef Py_ssize_t count = len(spans)
    if ends_differ is not None and ends_differ.shape[0] != count:
        raise ValueError("ends_differ must hold one byte per span")
    if coincident is not None and coincident.shape[0] != count:
        raise ValueError("coincident must hold one byte per span")
    if clip_mode == CLIP_ROWS:
        if row_offsets is None or row_spans is None or row_offsets.shape[0] < 1:
            raise ValueError("a clip path needs its rows")
        if row_offsets[row_offsets.shape[0] - 1] * 2 > row_spans.shape[0]:
            raise ValueError("row offsets run past the spans")
    cdef long long* bounds = <long long*> malloc((3 * count + 1) * sizeof(long long))
    if bounds == NULL:
        raise MemoryError
    cdef Py_ssize_t k, start, end
    cdef Paint p
    cdef Walk w
    cdef Callbacks calls = Callbacks.__new__(Callbacks)
    calls.line = on_line
    calls.join = on_join
    calls.cap = on_cap
    calls.dot = on_dot
    calls.extend = on_extend
    try:
        for k in range(count):
            start, end, flag = spans[k]
            if start < 0 or end > xs.shape[0] or end < start:
                raise ValueError("span runs past the points")
            bounds[3 * k] = start
            bounds[3 * k + 1] = end
            bounds[3 * k + 2] = 1 if (outline or flag) else 0
        p.pixels = &pixels[0, 0, 0] if pixels.shape[0] and pixels.shape[1] else NULL
        p.row_stride = pixels.strides[0]
        p.width = pixels.shape[1]
        p.height = pixels.shape[0]
        p.crop_x0 = crop_x0
        p.crop_y1 = crop_y1
        p.scale = scale
        p.clip_mode = clip_mode
        p.clip_empty = clip_empty
        if clip_mode != CLIP_NONE:
            p.clip[0] = clip_box[0]
            p.clip[1] = clip_box[1]
            p.clip[2] = clip_box[2]
            p.clip[3] = clip_box[3]
        p.rows_origin = rows_origin
        if clip_mode == CLIP_ROWS:
            p.rows_count = row_offsets.shape[0] - 1
            p.row_offsets = &row_offsets[0]
            p.row_spans = &row_spans[0] if row_spans.shape[0] else NULL
        else:
            p.rows_count = 0
            p.row_offsets = NULL
            p.row_spans = NULL
        p.red = red
        p.green = green
        p.blue = blue
        p.alpha = alpha
        p.exponent = exponent
        p.painted = False
        p.error = 0
        w.xs = &xs[0] if xs.shape[0] else NULL
        w.ys = &ys[0] if ys.shape[0] else NULL
        w.bounds = bounds
        w.ends_differ = &ends_differ[0] if ends_differ is not None and count else NULL
        w.coincident = &coincident[0] if coincident is not None and count else NULL
        w.count = count
        w.line_width = line_width
        w.native = native
        w.cap_nonzero = cap_nonzero
        w.cap_butt = cap_butt
        w.cap_round = cap_round
        w.join_round = join_round
        with nogil:
            walk(&p, &w, calls)
        flush(&p, calls)
    finally:
        free(bounds)
