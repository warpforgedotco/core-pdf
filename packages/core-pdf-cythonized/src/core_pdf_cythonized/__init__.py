# SPDX-License-Identifier: AGPL-3.0-only

"""Compiled kernels for core-pdf hot paths.

Nothing here adds behaviour. Each kernel reproduces, bit for bit, the
pure-Python function it replaced -- and that function is then deleted, so
core-pdf depends on this distribution and imports from it directly. There is
no fallback path to keep in sync, and no compiled/interpreted divergence to
reason about at runtime.

What replaces the fallback as a safety net is a golden vector file, generated
by the original before it was removed. See the package README.

A kernel earns its place here by clearing three bars, in this order:

1. Measured. The function is a named share of a profiled workload, and
   compiling it is worth a stated percentage of a real page. Compiling
   something because it looks numeric is how this package turns into dead
   weight that still has to be built on every platform.
2. Scalar, on data too small for its abstraction. The work is arithmetic and
   the data is too small to amortize the per-call overhead of whatever runs
   it. Usually that is the CPython interpreter; numpy counts too, since
   thirty array operations over a forty-pixel buffer is the same disease.
   Check the size distribution first -- numpy wins once arrays are large.
   Python object churn does not belong here: Cython loses to CPython's
   specializing interpreter on it, by roughly 2x for per-item construction.
3. Pinned. Before the original is deleted it generates golden vectors, and a
   test drives the kernel over them demanding identical output. Float
   arithmetic must match CPython exactly, which is why the build disables
   compiler float contraction; the vectors are what catch a regression.
"""

from core_pdf_cythonized._bezier import cubic_sample_times
from core_pdf_cythonized._coverage import signed_area_coverage

__all__ = ("cubic_sample_times", "signed_area_coverage")
