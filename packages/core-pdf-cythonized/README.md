# core-pdf-cythonized

Compiled kernels for `core-pdf` hot paths.

`core-pdf` depends on this distribution and imports from it directly. There is
no pure-Python fallback: when a kernel lands here, the Python it replaced is
deleted -- with one exception, `ObjectScanner`, described below. That makes `core-pdf` a compiled distribution — it needs a wheel for
the target platform, or a C compiler at install time.

## Two kernels own their algorithms

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

`decode_arithmetic_generic_template0` is the second, on the same terms: the
ITU-T T.88 MQ arithmetic decoder and generic region template 0, moved out of
`core-jbig2`. Compiling it in place would have made `core-jbig2` -- and spec,
which depends on it -- compiled distributions; mirroring it would have left
two decoders to drift. So `core-jbig2` 0.2.0 dropped `JBIG2MQDecoder`, the
`MQ_*` tables and both `decode_arithmetic_generic_*` functions, and its base
decoder now reports arithmetic generic regions as unsupported, as it already
did MMR and text regions. Core's `RecoveryJBIG2PageDecoder` checks the region
header and calls the kernel. The one conformance test the decoder had moved
here with it; spec's test now asserts that it declines.

## One kernel keeps its Python

`ObjectScanner` is the other exception, in the opposite direction: it mirrors
code that is not deleted. The Python it mirrors is `core-pdf-spec`'s object
lexer, which stays the parser for everyone using spec without core, and core's
reader subclass of it still parses everything the scanner declines. So the
scanner owns a subset of the grammar exactly -- well-formed dictionaries and
arrays under the reader's rules -- and returns None for the rest, after which
the reader parses the same bytes in Python from the same position.

That makes divergence possible where elsewhere it is not, and two guards stand
against it. `test_object_scanner_kernel.py` pins both the accepted results and
which inputs are declined, standalone, on every wheel.
`tests/src/core_pdf/test_object_scanner_contracts.py` compares the composed
reader -- scanner, fallback and all -- with the Python alone, error for error.

It also looks like it breaks the object-churn rule below, and does not: it
builds dicts, lists, ints and floats, which cost the same from C as from the
interpreter, and a repeated name is a dict hit. Rebuilding the object graphs
of the benchmark corpus in Python costs 12% of parsing them; the other 88% was
per-token dispatch, which is what compiling removes.

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
| `fill_rect_coverage` | the blend and group-plane records of `fill_rect`'s partially covered rectangles, and their two `rect_coverage_plane` calls (deleted); `rect_coverage_plane` stays as the golden-pinned reference | five kernel round trips per rectangle become one pass; -5% render on final-r3-report-emergency p4, test_2533, chinese-tables |
| `outline_edges` | `core_pdf.impl.render.commands` (edge half only) | 2.01x on the kernel; -2.5 to -4.4% on text-page render |
| `glyph_coverage_plane` | the device-edge preparation in `core_pdf.impl.render.target.fill_path` (deleted) | 8.3us of numpy prep per call removed, against 3.3us of actual coverage; -10.6% / -7.5% on text-page render |
| `fill_glyph_coverage` | the uint8 quantization of that plane, its blend and both group-plane records in `fill_path` (deleted) | four kernel round trips and two numpy passes per glyph become one pass; text-page render -6% (ISO 32000-2) / -8% (lyft, glyphs in knockout groups) |
| `sample_opaque_pixels` | the tiled advanced-indexing gather of `core_pdf.impl.render.target.blit_opaque_sampled_tiles` (deleted) | one byte copy per pixel instead of a scratch tile and a copy; PyMuPDF test_5001 render 0.80 to 0.65 s |
| `supersampled_coverage_plane` | the 4x4 supersampling deltas loop of `core_pdf.impl.render.target.fill_path` (deleted), which is what even-odd fills took | 15,426 corpus fills in 0.2 s against about 4.7 s; -31% on test_3806 rasterize (5.55 to 3.85 s), the slowest page in the corpus |
| `flatten_path_commands` | `flatten_path`, `CapturedPath.derived_lines` and the CTM transform in `core_pdf.impl.capture`, and `line_coordinate_columns` in `core_pdf.impl.extract.grids` (all deleted) | one pass instead of four walks and a `CapturedLine` per segment (733,122 on one page); -39% / -34% on the two stroke-heavy extraction pages, nothing elsewhere |
| `ContentScanner` | the regular-expression fast path in `core_pdf.impl.capture.recovery.iter_content_operations` (deleted) | 1.1x to 2.1x on the tokenizer, by document; -16% on a vector-heavy page render, nothing on text pages |
| `composite_elementary_normal` | the opaque-normal branch of `core_pdf.impl.render.target.composite_nonisolated_group` (deleted) | nine numpy passes over an 8-to-32-pixel plane removed; -4.2 to -4.8% on text-page render |
| `composite_normal_group` | `core_pdf.impl.render.blend.composite_normal_group_numpy` (deleted), all four of its routes | 9.3x on 256 corpus groups (9.56 to 1.03 ns/px); test_3450's 2,556 isolated groups |
| `composite_masked_normal` | the isolated normal branch of `core_pdf.impl.render.target.composite_masked_group` (deleted) | 16.6x on the kernel over 400 corpus planes (39.4 to 2.4 ns/px); every soft-masked group on test_3450 |
| `ObjectScanner` | nothing deleted: mirrors `parse_dictionary` and `parse_array` of `core_pdf.impl.document.recovery.lexer.PdfLexer`, which falls back to them | 5.5x on parsing the indirect objects of the benchmark corpus (201 to 37 ms), 99% of calls taking the compiled path; -30 to -46% on document open, -3 to -55% on first-page extraction |
| `composite_knockout_element`, `composite_knockout_group` | `core_pdf_spec.s_11_transparency.groups` and `core_pdf.impl.render.target` (both deleted) | 7.06x on the fused wrapper |
| `decode_arithmetic_generic_template0` | `core_jbig2.codec` (deleted, with `JBIG2MQDecoder` and the `MQ_*` tables) | 69x over the 18 generic regions the JBIG2 fixtures decode (168 to 2.4 ns/px); render of no_bad_redactions.4.1 1424 to 77 ms, the two SCORE-Bench JBIG2 scans -31% and -40% |

`glyph_coverage_plane`, `fill_glyph_coverage` and `signed_area_coverage` share one accumulation core;
`glyph_coverage_plane` is kept as the golden-pinned reference the fused fill is checked against.
Core reaches it through the fused entry point only; the device-space entry stays
public because it is what the coverage golden vectors pin directly, and pinning
the core through an affine transform instead would weaken them.

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
cibuildwheel for CPython 3.14 on Linux (x86_64, aarch64) and Windows (AMD64),
plus an sdist for everything else. macOS installs from the sdist, which carries
the kernel sources and their golden vectors and checks itself.

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
