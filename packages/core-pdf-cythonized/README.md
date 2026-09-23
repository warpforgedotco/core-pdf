# core-pdf-cythonized

Compiled kernels for `core-pdf` hot paths.

`core-pdf` depends on this distribution and imports from it directly. There is
no pure-Python fallback: when a kernel lands here, the Python it replaced is
deleted. That makes `core-pdf` a compiled distribution — it needs a wheel for
the target platform, or a C compiler at install time.

## One kernel owns its algorithm

`composite_knockout_element` is the exception to everything below. It is not
a mirror of code owned elsewhere — it *is* the ISO 32000-2 11.4.x knockout
algorithm, moved out of `core-pdf-spec` rather than copied, and its
conformance tests (`test_knockout_groups.py`) moved with it. That was a
deliberate trade: spec dropped a declared `__all__` export, and in exchange
spec stays pure Python instead of becoming a compiled distribution.

It came here because 60% of the cost sat in core's wrapper — a mask, three
boolean fancy-index copies and a scatter, all of it marshalling to hand
compacted arrays to a Python callee. Marshalling cannot be removed while the
callee stays in Python, so the two had to be compiled together.

## What belongs here

A kernel earns its place by clearing three bars, in order:

1. **Measured.** The function is a named share of a profiled workload, and
   compiling it is worth a stated percentage of a real page. Compiling things
   that merely look numeric is how this package becomes dead weight that still
   has to be built on every platform.
2. **Scalar, on data too small for its abstraction.** The work is arithmetic,
   and the data is too small to amortize the per-call overhead of whatever is
   running it. That abstraction is usually the CPython interpreter, but numpy
   counts too: `signed_area_coverage` had no Python loop at all — it ran about
   thirty array operations to fill a forty-pixel buffer, and a scalar C loop
   beat it 4.3x. The inverse also holds, so check the size distribution before
   assuming: numpy wins once the arrays are genuinely large.

   What does *not* belong here is Python object churn. Cython loses to
   CPython's specializing interpreter on object-heavy code — measured at about
   2x slower for per-item construction.
3. **Pinned.** Before the Python original is deleted, it generates a golden
   vector file, and a test drives the kernel over those vectors demanding
   identical output. That file becomes the definition of correct.

## Float arithmetic is part of the contract

Match the width the original computed in. numpy promoted uint8 buffers to
**float32** for compositing, so `_blend` uses C `float` and typed float
constants — a bare `1.0` is a double in C and would promote the expression,
compute in double and narrow on assignment, landing on different bytes.
Cython has no float-literal suffix, so named constants are how that is said.

The kernels must also reproduce CPython's float semantics bit for bit, and a C
compiler will not do that on its own. Left alone it contracts expressions like
`b*b - 4*a*c` into an FMA — more accurate, and wrong here, because it moves a
root by one ULP. `setup.py` builds with `-ffp-contract=off` for that reason.
The golden vectors are what catch a regression, on whichever platform built
the wheel.

## Current contents

| kernel | replaces | measured |
| ------ | -------- | -------- |
| `cubic_sample_times` | `core_pdf.impl.fonts.font_program` (deleted) | 7.57x on the kernel; 3.7% of a glyph-outline page, 0% on pages needing no outlines |
| `signed_area_coverage` | `core_pdf.impl.render.paths` (deleted, with `group_offsets`) | 4.26x on the kernel; ~20% off full render on glyph-heavy pages |
| `blend_normal_alpha_array_numpy` | `core_pdf.impl.render.blend` (deleted) | 9.04x on the kernel, over 26,315 calls averaging 38 elements |
| `rect_coverage_plane` | `core_pdf.impl.render.paths` (deleted) | 8.57x on the kernel; -21% on a vector-heavy page render, -7 to -9% elsewhere |
| `composite_knockout_element`, `composite_knockout_group` | `core_pdf_spec.s_11_transparency.groups` and `core_pdf.impl.render.target` (both deleted) | 7.06x on the fused wrapper |

Together the two render kernels take full `render().rasterize()` down by
21-32% across the corpus, with byte-identical pixels:

| page | before | after |
| ---- | ------ | ----- |
| i1040nr | 907 ms | ~615 ms |
| lyft_2021 | 635 ms | ~443 ms |
| billionaires | 505 ms | ~388 ms |
| issue-301 | 1007 ms | ~794 ms |

## Wheels

`core-pdf` depends on this distribution, so it cannot be installed from source
without a compiler. `.github/workflows/wheels.yml` builds wheels with
cibuildwheel for CPython 3.14 on Linux (x86_64, aarch64), macOS (arm64) and
Windows (AMD64), plus an sdist for anything else.

There is no macOS x86_64 wheel. Its runner image is retired, and every way of
producing one from an arm64 runner gives up running the golden vectors on the
wheel that was built -- which is the one thing this matrix exists to do. Intel
Macs build from the sdist instead.

Every wheel runs the golden vectors before it is kept. That is not ceremony:
the rest of the repo tests on Linux only, and float behaviour is exactly what
a different compiler on a different platform changes. `setup.py` picks the
contraction flag per compiler for the same reason -- passing the GCC/Clang
spelling to MSVC would be worse than passing nothing, because MSVC warns and
carries on, and the build would succeed with semantics the vectors do not
describe.

The sdist carries the tests and their golden data, so a source build can check
itself on a platform no wheel covers.

## Building and testing

Needs a C compiler. From the workspace root:

```sh
uv sync --all-packages --all-groups
uv run --locked --no-sync pytest packages/core-pdf-cythonized/tests
```
