# SPDX-License-Identifier: AGPL-3.0-only

cimport numpy as cnp


cdef inline bint is_float64_vector(object value) noexcept:
    return (
        cnp.PyArray_Check(value)
        and cnp.PyArray_TYPE(<cnp.ndarray> value) == cnp.NPY_FLOAT64
        and cnp.PyArray_NDIM(<cnp.ndarray> value) == 1
        and cnp.PyArray_ISCARRAY_RO(<cnp.ndarray> value)
    )


cdef inline Py_ssize_t vector_length(object value) noexcept:
    return cnp.PyArray_DIM(<cnp.ndarray> value, 0)


cdef inline double* float64_data(object value) noexcept:
    return <double*> cnp.PyArray_DATA(<cnp.ndarray> value)


cdef inline cnp.ndarray empty_float64(Py_ssize_t count):
    cdef cnp.npy_intp dims[1]
    dims[0] = count
    return cnp.PyArray_EMPTY(1, dims, cnp.NPY_FLOAT64, 0)


cdef inline cnp.ndarray empty_float64_rows(Py_ssize_t rows, Py_ssize_t columns):
    cdef cnp.npy_intp dims[2]
    dims[0] = rows
    dims[1] = columns
    return cnp.PyArray_EMPTY(2, dims, cnp.NPY_FLOAT64, 0)
