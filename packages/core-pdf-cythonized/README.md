# core-pdf-cythonized

Compiled kernels for `core-pdf` hot paths.

`core-pdf` depends on this distribution and imports from it directly. There is
no pure-Python fallback: when a kernel lands here, the Python it replaced is
deleted. That makes `core-pdf` a compiled distribution — it needs a wheel for
the target platform, or a C compiler at install time.

## What belongs here

A kernel earns its place by clearing three bars, in order:

1. **Measured.** The function is a named share of a profiled workload, and
   compiling it is worth a stated percentage of a real page. Compiling things
   that merely look numeric is how this package becomes dead weight that still
   has to be built on every platform.
2. **Scalar.** The inner loop touches no Python objects. Cython loses to
   CPython's specializing interpreter on object-heavy code — measured at about
   2x slower for per-item construction — so object churn does not belong here.
3. **Pinned.** Before the Python original is deleted, it generates a golden
   vector file, and a test drives the kernel over those vectors demanding
   identical output. That file becomes the definition of correct.

## Float arithmetic is part of the contract

The kernels must reproduce CPython's float semantics bit for bit, and a C
compiler will not do that on its own. Left alone it contracts expressions like
`b*b - 4*a*c` into an FMA — more accurate, and wrong here, because it moves a
root by one ULP. `setup.py` builds with `-ffp-contract=off` for that reason.
The golden vectors are what catch a regression, on whichever platform built
the wheel.

## Current contents

| kernel | replaces | measured |
| ------ | -------- | -------- |
| `cubic_sample_times` | `core_pdf.impl.fonts.font_program` (deleted) | 7.57x on the kernel; 3.7% of a glyph-outline page, 0% on pages needing no outlines |

## Building and testing

Needs a C compiler. From the workspace root:

```sh
uv sync --all-packages --all-groups
uv run --locked --no-sync pytest packages/core-pdf-cythonized/tests
```
