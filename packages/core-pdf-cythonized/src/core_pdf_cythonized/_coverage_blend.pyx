# SPDX-License-Identifier: AGPL-3.0-only
"""4x4 coverage counts blended pixel by pixel (core_pdf.impl.render.target.fill_path).

A fill under a clip that is not a set of rectangles cannot take fill_path's
plane kernels, which blend in float32 over a whole window. It falls back to
a Python loop that samples each pixel 4x4, tests it against the clip and
blends it through blend_coverage_pixel and blend_px -- in double, recording
into the group's float32 source planes one pixel at a time. On SCORE-Bench
wipo-2022-financial-report that loop was about half the page's rasterize.

supersampled_coverage_plane already produces the same 4x4 counts (checked
against the loop on every such fill of six corpus pages), so this is the
rest of the loop: the clip test, as a mask the caller builds from the same
row spans, and blend_px's normal-mode arithmetic and per-pixel plane
updates, from _pixel_blend.pxd. It returns the box of pixels whose paint
window blend_px would have extended.
"""

from core_pdf_cythonized._pixel_blend cimport (
    blend_normal_pixel,
    clamp_byte,
    coverage_alpha,
    plane_accumulate,
)

__all__ = ("blend_coverage_counts",)


def blend_coverage_counts(
    unsigned char[:, :, ::1] pixels,
    const unsigned char[:, ::1] counts,
    Py_ssize_t left,
    Py_ssize_t top,
    const unsigned char[::1] allowed,
    int red,
    int green,
    int blue,
    int alpha,
    float[:, :] source_alpha,
    float[:, :] source_shape,
    bint track_shape,
    double shape_alpha,
):
    """Blend `counts` (out of 16) into `pixels` at (left, top).

    `allowed`, if not None, holds a byte per count, zero where the clip
    rejects the pixel. `source_alpha` and `source_shape`, if not None, are
    the group planes, the same shape as `pixels`.
    """
    cdef Py_ssize_t rows = counts.shape[0]
    cdef Py_ssize_t cols = counts.shape[1]
    if left < 0 or top < 0 or top + rows > pixels.shape[0] or left + cols > pixels.shape[1]:
        raise ValueError("counts run past the pixels given")
    cdef bint masked = allowed is not None
    if masked and allowed.shape[0] != rows * cols:
        raise ValueError("allowed must hold one byte per count")
    cdef bint has_alpha = source_alpha is not None
    cdef bint has_shape = source_shape is not None
    if has_alpha and (source_alpha.shape[0] != pixels.shape[0] or source_alpha.shape[1] != pixels.shape[1]):
        raise ValueError("source_alpha differs from the pixels in shape")
    if has_shape and (source_shape.shape[0] != pixels.shape[0] or source_shape.shape[1] != pixels.shape[1]):
        raise ValueError("source_shape differs from the pixels in shape")
    cdef Py_ssize_t r, c, y, x
    cdef int covered, sa, shape
    cdef Py_ssize_t low_x = left + cols, low_y = top + rows, high_x = left, high_y = top
    with nogil:
        for r in range(rows):
            y = top + r
            for c in range(cols):
                covered = counts[r, c]
                if not covered:
                    continue
                if masked and not allowed[r * cols + c]:
                    continue
                x = left + c
                sa = coverage_alpha(alpha, covered)
                if has_shape:
                    shape = clamp_byte(<double> (255 * covered) / 16.0) if track_shape else 255
                    source_shape[y, x] = plane_accumulate(
                        source_shape[y, x], (shape / 255.0) * shape_alpha
                    )
                # blend_px extends the paint window here unless only the alpha
                # plane is recording, which extends it itself when sa > 0.
                if has_shape or not has_alpha or sa > 0:
                    if x < low_x:
                        low_x = x
                    if x + 1 > high_x:
                        high_x = x + 1
                    if y < low_y:
                        low_y = y
                    if y + 1 > high_y:
                        high_y = y + 1
                if sa <= 0:
                    continue
                if has_alpha:
                    source_alpha[y, x] = plane_accumulate(source_alpha[y, x], sa / 255.0)
                blend_normal_pixel(&pixels[y, x, 0], red, green, blue, sa)
    if high_x <= low_x:
        return None
    return low_x, low_y, high_x, high_y
