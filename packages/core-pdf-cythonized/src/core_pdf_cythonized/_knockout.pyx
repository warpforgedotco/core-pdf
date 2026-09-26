# SPDX-License-Identifier: AGPL-3.0-only
"""Knockout transparency group compositing (ISO 32000-2 11.4.x).

Moved here from core_pdf_spec.s_11_transparency.groups. It is the one kernel
in this package that owns a PDF-defined algorithm rather than mirroring a
helper, which is why the spec conformance tests moved with it.

Two entry points share one inner routine:

* ``composite_knockout_element`` is the function spec exported, signature and
  validation unchanged, operating on compacted sample arrays.
* ``composite_knockout_group`` is the caller core used to wrap it. Sixty
  percent of that wrapper's cost was marshalling -- building a mask, three
  boolean fancy-index copies, then a scatter back -- purely to hand compacted
  arrays to a Python callee. Fusing the mask, the gather, the arithmetic and
  the scatter into one pass is the entire point of compiling this.

All arithmetic is float64, matching the numpy original, and setup.py builds
with -ffp-contract=off so the compiler does not fold these expressions into
FMAs and shift the results.
"""

from cpython.mem cimport PyMem_Free, PyMem_Malloc
from libc.math cimport rint

from core_pdf_cythonized._knockout_math cimport clamp_byte, knockout_component

import numpy

__all__ = (
    "composite_elementary_knockout",
    "composite_knockout_element",
    "composite_knockout_group",
)


def unit_range(*arrays):
    for values in arrays:
        if values.size and not (values.min() >= 0.0 and values.max() <= 1.0):
            return False
    return True


def composite_knockout_element(
    components,
    alpha,
    *,
    backdrop_components,
    backdrop_alpha,
    element_components,
    element_alpha,
    shape,
    group_alpha,
    element_group_alpha,
    validate=True,
):
    color = numpy.asarray(components, dtype=numpy.float64)
    complete = numpy.asarray(alpha, dtype=numpy.float64)
    backdrop = numpy.asarray(backdrop_components, dtype=numpy.float64)
    initial = numpy.asarray(backdrop_alpha, dtype=numpy.float64)
    element = numpy.asarray(element_components, dtype=numpy.float64)
    element_complete = numpy.asarray(element_alpha, dtype=numpy.float64)
    coverage = numpy.asarray(shape, dtype=numpy.float64)
    accumulated = numpy.asarray(group_alpha, dtype=numpy.float64)
    element_accumulated = numpy.asarray(element_group_alpha, dtype=numpy.float64)
    # NB: the build sets wraparound=False, which disables negative-index
    # wrapping for Python sequences as well as memoryviews. color.shape[-1]
    # there does not read the last dimension -- it reads out of range, and the
    # garbage propagates into the loop bounds below as a segfault. Index the
    # shape tuple positively.
    if color.ndim < 1:
        raise ValueError("invalid knockout group samples")
    cdef Py_ssize_t channels = color.shape[color.ndim - 1]
    if (
        channels == 0
        or backdrop.shape != color.shape
        or element.shape != color.shape
        or complete.shape != color.shape[: color.ndim - 1]
        or any(
            values.shape != complete.shape
            for values in (initial, element_complete, coverage, accumulated, element_accumulated)
        )
        or (
            validate
            and (
                not unit_range(
                    color,
                    complete,
                    backdrop,
                    initial,
                    element,
                    element_complete,
                    coverage,
                    accumulated,
                    element_accumulated,
                )
                or bool(numpy.any(element_accumulated > coverage))
            )
        )
    ):
        raise ValueError("invalid knockout group samples")

    cdef Py_ssize_t rows = color.size // channels

    # Contiguity is needed only for the typed memoryviews below, and is
    # applied after validation for the reason noted above.
    flat_color = numpy.ascontiguousarray(color).reshape(rows, channels)
    flat_backdrop = numpy.ascontiguousarray(backdrop).reshape(rows, channels)
    flat_element = numpy.ascontiguousarray(element).reshape(rows, channels)
    result = numpy.zeros((rows, channels), numpy.float64)
    result_alpha = numpy.zeros(rows, numpy.float64)
    result_group_alpha = numpy.zeros(rows, numpy.float64)

    cdef double[:, ::1] c_view = flat_color
    cdef double[:, ::1] b_view = flat_backdrop
    cdef double[:, ::1] e_view = flat_element
    cdef double[::1] complete_view = numpy.ascontiguousarray(complete).reshape(rows)
    cdef double[::1] initial_view = numpy.ascontiguousarray(initial).reshape(rows)
    cdef double[::1] ec_view = numpy.ascontiguousarray(element_complete).reshape(rows)
    cdef double[::1] coverage_view = numpy.ascontiguousarray(coverage).reshape(rows)
    cdef double[::1] acc_view = numpy.ascontiguousarray(accumulated).reshape(rows)
    cdef double[::1] eacc_view = numpy.ascontiguousarray(element_accumulated).reshape(rows)
    cdef double[:, ::1] out_view = result
    cdef double[::1] out_alpha = result_alpha
    cdef double[::1] out_group = result_group_alpha

    cdef Py_ssize_t i, k
    cdef double remaining, rga, ra

    with nogil:
        for i in range(rows):
            remaining = 1.0 - coverage_view[i]
            rga = eacc_view[i] + remaining * acc_view[i]
            ra = initial_view[i] + (1.0 - initial_view[i]) * rga
            out_group[i] = rga
            out_alpha[i] = ra
            for k in range(channels):
                out_view[i, k] = knockout_component(
                    e_view[i, k], ec_view[i], remaining,
                    c_view[i, k], complete_view[i],
                    b_view[i, k], initial_view[i], ra,
                )

    return (
        result.reshape(color.shape),
        result_alpha.reshape(complete.shape),
        result_group_alpha.reshape(complete.shape),
    )


def composite_knockout_group(destination, backdrop, element, group_alpha, element_alpha, shape):
    cdef unsigned char[:, :, :] dst = destination
    cdef unsigned char[:, :, :] bak = backdrop
    cdef unsigned char[:, :, :] ele = element
    cdef float[:, :] group = group_alpha
    cdef unsigned char[:, :] ealpha = element_alpha
    cdef float[:, :] cover = shape

    cdef Py_ssize_t rows = dst.shape[0], cols = dst.shape[1]
    cdef Py_ssize_t i, j, k
    cdef double eff, sh, remaining, rga, ra, complete, initial, ec
    cdef double colour[3]

    with nogil:
        for i in range(rows):
            for j in range(cols):
                eff = <double> ealpha[i, j] / 255.0
                sh = <double> cover[i, j]
                if sh < 0.0:
                    sh = 0.0
                elif sh > 1.0:
                    sh = 1.0
                if eff > sh:
                    sh = eff
                if sh <= 0.0:
                    continue

                complete = <double> dst[i, j, 3] / 255.0
                initial = <double> bak[i, j, 3] / 255.0
                ec = <double> ele[i, j, 3] / 255.0
                remaining = 1.0 - sh
                rga = eff + remaining * <double> group[i, j]
                ra = initial + (1.0 - initial) * rga

                for k in range(3):
                    colour[k] = knockout_component(
                        <double> ele[i, j, k] / 255.0, ec, remaining,
                        <double> dst[i, j, k] / 255.0, complete,
                        <double> bak[i, j, k] / 255.0, initial, ra,
                    )
                for k in range(3):
                    dst[i, j, k] = <unsigned char> clamp_byte(colour[k])
                dst[i, j, 3] = <unsigned char> clamp_byte(ra)
                group[i, j] = <float> rga


def composite_elementary_knockout(
    unsigned char[:, :, :] destination,
    const unsigned char[:, :, :] backdrop,
    const unsigned char[:, :, :] rendered,
    const float[:, :] source_alpha,
    float[:, :] group_alpha,
    const float[:, :] shape,
    float[:, :] parent_shape,
):
    """An opaque normal elementary group composited into its knockout parent.

    composite_group made the element as a copy of the parent's initial
    backdrop, composite_elementary_normal copied the rendered pixels over it
    where the group's quantized alpha is non-zero, composite_knockout_group
    knocked the element into the parent, and numpy recorded the shape as
    parent_shape += (1.0 - parent_shape) * shape, all in float32. A text
    object's knockout group does this for every glyph that touches another.
    This is those steps per pixel in one pass: the same quantization, with
    every value checked before anything is written as the original checked
    it, the same knockout arithmetic in float64, the same float32 shape
    update when the parent records shape.
    """
    cdef Py_ssize_t rows = source_alpha.shape[0], cols = source_alpha.shape[1]
    if destination.shape[0] != rows or destination.shape[1] != cols or destination.shape[2] != 4:
        raise ValueError("destination must be source_alpha.shape + (4,)")
    if backdrop.shape[0] != rows or backdrop.shape[1] != cols or backdrop.shape[2] != 4:
        raise ValueError("backdrop must be source_alpha.shape + (4,)")
    if rendered.shape[0] != rows or rendered.shape[1] != cols or rendered.shape[2] != 4:
        raise ValueError("rendered must be source_alpha.shape + (4,)")
    if group_alpha.shape[0] != rows or group_alpha.shape[1] != cols:
        raise ValueError("group_alpha differs from source_alpha in shape")
    if shape.shape[0] != rows or shape.shape[1] != cols:
        raise ValueError("shape differs from source_alpha in shape")
    cdef bint has_parent_shape = parent_shape is not None
    if has_parent_shape and (parent_shape.shape[0] != rows or parent_shape.shape[1] != cols):
        raise ValueError("parent_shape differs from source_alpha in shape")
    if rows == 0 or cols == 0:
        return
    cdef unsigned char* quantized = <unsigned char*> PyMem_Malloc(rows * cols)
    if quantized == NULL:
        raise MemoryError
    cdef Py_ssize_t i, j, k
    cdef double scaled, eff, sh, remaining, rga, ra, complete, initial, ec
    cdef double colour[3]
    cdef const unsigned char* element
    cdef float ONE = 1.0
    cdef float previous, cover
    try:
        # composite_elementary_normal quantized, and rejected, every value
        # before it copied a pixel.
        for i in range(rows):
            for j in range(cols):
                scaled = rint(<double> source_alpha[i, j] * 255.0)
                if scaled < 0.0 or scaled > 255.0:
                    raise ValueError("source_alpha must lie in [0, 1]")
                quantized[i * cols + j] = <unsigned char> <int> scaled
        with nogil:
            for i in range(rows):
                for j in range(cols):
                    # The element: the rendered pixel where the group shows,
                    # the initial backdrop elsewhere.
                    if quantized[i * cols + j] > 0:
                        element = &rendered[i, j, 0]
                    else:
                        element = &backdrop[i, j, 0]
                    eff = <double> quantized[i * cols + j] / 255.0
                    sh = <double> shape[i, j]
                    if sh < 0.0:
                        sh = 0.0
                    elif sh > 1.0:
                        sh = 1.0
                    if eff > sh:
                        sh = eff
                    if sh > 0.0:
                        complete = <double> destination[i, j, 3] / 255.0
                        initial = <double> backdrop[i, j, 3] / 255.0
                        ec = <double> element[3] / 255.0
                        remaining = 1.0 - sh
                        rga = eff + remaining * <double> group_alpha[i, j]
                        ra = initial + (1.0 - initial) * rga
                        for k in range(3):
                            colour[k] = knockout_component(
                                <double> element[k] / 255.0, ec, remaining,
                                <double> destination[i, j, k] / 255.0, complete,
                                <double> backdrop[i, j, k] / 255.0, initial, ra,
                            )
                        for k in range(3):
                            destination[i, j, k] = <unsigned char> clamp_byte(colour[k])
                        destination[i, j, 3] = <unsigned char> clamp_byte(ra)
                        group_alpha[i, j] = <float> rga
                    if has_parent_shape:
                        previous = parent_shape[i, j]
                        cover = shape[i, j]
                        parent_shape[i, j] = previous + (ONE - previous) * cover
    finally:
        PyMem_Free(quantized)
