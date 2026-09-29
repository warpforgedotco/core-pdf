# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
import numpy
from cython.cimports.core_pdf_cythonized._byte_clamp import unit_to_byte, unit_to_byte_checked
from cython.cimports.core_pdf_cythonized._knockout_math import knockout_component
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Malloc

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
    if color.ndim < 1:
        raise ValueError("invalid knockout group samples")
    channels: cython.Py_ssize_t = color.shape[color.ndim - 1]
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

    rows: cython.Py_ssize_t = color.size // channels

    flat_color = numpy.ascontiguousarray(color).reshape(rows, channels)
    flat_backdrop = numpy.ascontiguousarray(backdrop).reshape(rows, channels)
    flat_element = numpy.ascontiguousarray(element).reshape(rows, channels)
    result = numpy.zeros((rows, channels), numpy.float64)
    result_alpha = numpy.zeros(rows, numpy.float64)
    result_group_alpha = numpy.zeros(rows, numpy.float64)

    c_view: cython.double[:, ::1] = flat_color
    b_view: cython.double[:, ::1] = flat_backdrop
    e_view: cython.double[:, ::1] = flat_element
    complete_view: cython.double[::1] = numpy.ascontiguousarray(complete).reshape(rows)
    initial_view: cython.double[::1] = numpy.ascontiguousarray(initial).reshape(rows)
    ec_view: cython.double[::1] = numpy.ascontiguousarray(element_complete).reshape(rows)
    coverage_view: cython.double[::1] = numpy.ascontiguousarray(coverage).reshape(rows)
    acc_view: cython.double[::1] = numpy.ascontiguousarray(accumulated).reshape(rows)
    eacc_view: cython.double[::1] = numpy.ascontiguousarray(element_accumulated).reshape(rows)
    out_view: cython.double[:, ::1] = result
    out_alpha: cython.double[::1] = result_alpha
    out_group: cython.double[::1] = result_group_alpha

    i: cython.Py_ssize_t
    k: cython.Py_ssize_t
    remaining: cython.double
    rga: cython.double
    ra: cython.double

    with cython.nogil:
        for i in range(rows):
            remaining = 1.0 - coverage_view[i]
            rga = eacc_view[i] + remaining * acc_view[i]
            ra = initial_view[i] + (1.0 - initial_view[i]) * rga
            out_group[i] = rga
            out_alpha[i] = ra
            for k in range(channels):
                out_view[i, k] = knockout_component(
                    e_view[i, k],
                    ec_view[i],
                    remaining,
                    c_view[i, k],
                    complete_view[i],
                    b_view[i, k],
                    initial_view[i],
                    ra,
                )

    return (
        result.reshape(color.shape),
        result_alpha.reshape(complete.shape),
        result_group_alpha.reshape(complete.shape),
    )


def composite_knockout_group(destination, backdrop, element, group_alpha, element_alpha, shape):
    dst: cython.uchar[:, :, :] = destination
    bak: cython.uchar[:, :, :] = backdrop
    ele: cython.uchar[:, :, :] = element
    group: cython.float[:, :] = group_alpha
    ealpha: cython.uchar[:, :] = element_alpha
    cover: cython.float[:, :] = shape

    rows: cython.Py_ssize_t = dst.shape[0]
    cols: cython.Py_ssize_t = dst.shape[1]
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    eff: cython.double
    sh: cython.double
    remaining: cython.double
    rga: cython.double
    ra: cython.double
    complete: cython.double
    initial: cython.double
    ec: cython.double
    colour = cython.declare(cython.double[3])

    with cython.nogil:
        for i in range(rows):
            for j in range(cols):
                eff = cython.cast(cython.double, ealpha[i, j]) / 255.0
                sh = cython.cast(cython.double, cover[i, j])
                if sh < 0.0:
                    sh = 0.0
                elif sh > 1.0:
                    sh = 1.0
                if eff > sh:
                    sh = eff
                if sh <= 0.0:
                    continue

                complete = cython.cast(cython.double, dst[i, j, 3]) / 255.0
                initial = cython.cast(cython.double, bak[i, j, 3]) / 255.0
                ec = cython.cast(cython.double, ele[i, j, 3]) / 255.0
                remaining = 1.0 - sh
                rga = eff + remaining * cython.cast(cython.double, group[i, j])
                ra = initial + (1.0 - initial) * rga

                for k in range(3):
                    colour[k] = knockout_component(
                        cython.cast(cython.double, ele[i, j, k]) / 255.0,
                        ec,
                        remaining,
                        cython.cast(cython.double, dst[i, j, k]) / 255.0,
                        complete,
                        cython.cast(cython.double, bak[i, j, k]) / 255.0,
                        initial,
                        ra,
                    )
                for k in range(3):
                    dst[i, j, k] = cython.cast(cython.uchar, unit_to_byte(colour[k]))
                dst[i, j, 3] = cython.cast(cython.uchar, unit_to_byte(ra))
                group[i, j] = cython.cast(cython.float, rga)


def composite_elementary_knockout(
    destination: cython.uchar[:, :, :],
    backdrop: cython.const[cython.uchar][:, :, :],
    rendered: cython.const[cython.uchar][:, :, :],
    source_alpha: cython.const[cython.float][:, :],
    group_alpha: cython.float[:, :],
    shape: cython.const[cython.float][:, :],
    parent_shape: typing.Optional[cython.float[:, :]],
) -> None:
    rows: cython.Py_ssize_t = source_alpha.shape[0]
    cols: cython.Py_ssize_t = source_alpha.shape[1]
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
    has_parent_shape: cython.bint = parent_shape is not None
    if has_parent_shape and (parent_shape.shape[0] != rows or parent_shape.shape[1] != cols):
        raise ValueError("parent_shape differs from source_alpha in shape")
    if rows == 0 or cols == 0:
        return
    quantized: cython.p_uchar = cython.cast(cython.p_uchar, PyMem_Malloc(rows * cols))
    if quantized == cython.NULL:
        raise MemoryError
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    eff: cython.double
    sh: cython.double
    remaining: cython.double
    rga: cython.double
    ra: cython.double
    complete: cython.double
    initial: cython.double
    ec: cython.double
    colour = cython.declare(cython.double[3])
    element: cython.p_const_uchar
    ONE: cython.float = 1.0
    previous: cython.float
    cover: cython.float
    try:
        for i in range(rows):
            for j in range(cols):
                quantized[i * cols + j] = cython.cast(
                    cython.uchar,
                    unit_to_byte_checked(cython.cast(cython.double, source_alpha[i, j])),
                )
        with cython.nogil:
            for i in range(rows):
                for j in range(cols):
                    if quantized[i * cols + j] > 0:
                        element = cython.address(rendered[i, j, 0])
                    else:
                        element = cython.address(backdrop[i, j, 0])
                    eff = cython.cast(cython.double, quantized[i * cols + j]) / 255.0
                    sh = cython.cast(cython.double, shape[i, j])
                    if sh < 0.0:
                        sh = 0.0
                    elif sh > 1.0:
                        sh = 1.0
                    if eff > sh:
                        sh = eff
                    if sh > 0.0:
                        complete = cython.cast(cython.double, destination[i, j, 3]) / 255.0
                        initial = cython.cast(cython.double, backdrop[i, j, 3]) / 255.0
                        ec = cython.cast(cython.double, element[3]) / 255.0
                        remaining = 1.0 - sh
                        rga = eff + remaining * cython.cast(cython.double, group_alpha[i, j])
                        ra = initial + (1.0 - initial) * rga
                        for k in range(3):
                            colour[k] = knockout_component(
                                cython.cast(cython.double, element[k]) / 255.0,
                                ec,
                                remaining,
                                cython.cast(cython.double, destination[i, j, k]) / 255.0,
                                complete,
                                cython.cast(cython.double, backdrop[i, j, k]) / 255.0,
                                initial,
                                ra,
                            )
                        for k in range(3):
                            destination[i, j, k] = cython.cast(
                                cython.uchar, unit_to_byte(colour[k])
                            )
                        destination[i, j, 3] = cython.cast(cython.uchar, unit_to_byte(ra))
                        group_alpha[i, j] = cython.cast(cython.float, rga)
                    if has_parent_shape:
                        previous = parent_shape[i, j]
                        cover = shape[i, j]
                        parent_shape[i, j] = previous + (ONE - previous) * cover
    finally:
        PyMem_Free(quantized)
