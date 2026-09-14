# Coverage and unused-code findings

## PDFMiner startxref boundary contracts

Added eight differential cases for exact offsets, offsets inside the `xref`
token, zero, EOF, the signed 32-bit boundary, and negative offsets. Seven cases
preserve the reference library's extracted page text. The negative-offset case
records a compatibility difference: pdfminer recovers the page, while native
loading raises `PdfParseError` before the facade's recovery policy runs. This
needs a scoped reader/facade recovery-policy design; the test does not claim parity.
No production code changed.

The focused recovery file passed **12 tests in 6.33s**, with coverage appended
to the existing full-workspace database. The resulting local measurement is
**35,303 / 38,842 statements (90.89%)** and
**12,453 / 14,958 branches (83.25%)**. This was not a new full-suite run;
the committed coverage floor remains unchanged. All four source roots,
vendor-only omissions, and **554 excluded lines** remain unchanged.

## Content recovery boundaries and lexer resume contracts

Added **63 deterministic cases** for inline-image separators, default/custom
operator recognition, known-length payload recovery, cursor progress, malformed
container EOF handling, operand limits, structural tokens, and discarded pending
operands after broken inline images. Corrected a misleading comment: braces are
ordinary word tokens, not skipped delimiters. Executable syntax and source
locations were verified identical before and after this comment-only change.
No runtime behavior changed.

The focused coverage run passed **63 tests in 5.85s**, appended to workspace
coverage from the verified **7,485-test full run** and later focused tests; this
was not another full-suite run. Content recovery now has **94 / 94 statements
and 44 / 44 branches covered**. Workspace coverage increased
**90.77% → 90.89% statements** and **83.05% → 83.24% branches**:
**35,302 / 38,842 statements** and **12,451 / 14,958 branches**. All four source
roots, vendor-only omissions, and **554 excluded lines** remain unchanged. Both
exact coverage ratchets and all pre-push quality gates passed, including mypy, ty,
and import contracts. The static unused-symbol scan returned no candidates;
that heuristic does not prove the absence of unused code.
The 100% goal remains incomplete: **3,540 statements and 2,507 branches** remain
uncovered. These are local results.

## Standards declaration recovery and profile-claim identity

Added **36 deterministic cases** for profile revisions, repeated/conflicting
properties, XMP namespace-prefix independence, attribute/element forms, nested
resource exclusion, WTPDF targets, and extension diagnostics. They verify that
raw claims stay visible, unknown/conflicting targets remain unidentified, and
valid extension entries survive malformed neighbors. Declaration discovery is
not validation. No production change was needed.

The focused coverage run passed **36 tests in 5.75s**, appended to the verified
**7,485-test full workspace/differential run** on unchanged production sources;
this was not another full-suite run. Workspace coverage increased
**90.71% → 90.77% statements** and **82.90% → 83.05% branches**:
**35,258 / 38,842 statements** and **12,422 / 14,958 branches**. All four source
roots, vendor-only omissions, and **554 excluded lines** remain unchanged. Both
exact coverage ratchets and all pre-push quality gates passed, including mypy, ty,
and import contracts. The static unused-symbol scan returned no candidates;
that heuristic does not prove the absence of unused code.
The 100% goal remains incomplete: **3,584 statements and 2,536 branches** remain
uncovered. These are local results.

## Wrapped-cell logical rows and unreachable height guard

Added **24 deterministic cases** for logical row merging with tall cells, ragged
rows, blank content, missing cell boxes, spans, geometry unions, and retained
metadata. Positive controls verify text stays within its column; rejection
controls cover numeric tables, missing row geometry, short/narrow tables, and
insufficient evidence of wrapping. Removed the unreachable empty-height guard:
at least four rows are required, and every row either contributes a nonempty
set of box heights or returns before that point. This removes **two statements
and two branches** from the production denominator.

The locked full workspace and differential suite with real Tesseract and veraPDF
passed **7,485 tests, no skips, in 445.46s**. Workspace coverage increased
**90.62% → 90.71% statements** and **82.76% → 82.90% branches**:
**35,232 / 38,842 statements** and **12,400 / 14,958 branches**. All four source
roots, vendor-only omissions, and **554 excluded lines** remain unchanged. Both
exact coverage ratchets and all pre-push quality gates passed, including mypy, ty,
and import contracts. The static unused-symbol scan returned no candidates;
that heuristic does not prove the absence of unused code.
The 100% goal remains incomplete: **3,610 statements and 2,558 branches** remain
uncovered. These are local results.

## Reader PDF-function adapter contracts

Added **54 deterministic cases** for callable/constant function inputs, output
shape, Type2 defaults and unequal component extension, malformed Type3 stitching
intervals, repeated missing subfunctions, sampled-function setup recovery, and
stream-decoder error boundaries. Tests use explicit scalar results, retain input
dictionaries unchanged, preserve known failure causes, and verify unexpected
decoder defects propagate unchanged. No production change was needed.

The focused coverage run passed **54 tests in 5.81s**, appended to the verified
**7,407-test full workspace/differential run** on unchanged production sources;
this was not another full-suite run. The function adapter now has
**142 / 142 statements and 64 / 64 branches covered**. Workspace coverage increased
**90.50% → 90.62% statements** and **82.60% → 82.76% branches**:
**35,200 / 38,844 statements** and **12,381 / 14,960 branches**. All four source
roots, vendor-only omissions, and **554 excluded lines** remain unchanged. Both
exact coverage ratchets and all pre-push quality gates passed, including mypy, ty,
and import contracts. The static unused-symbol scan returned no candidates;
that heuristic does not prove the absence of unused code.
The 100% goal remains incomplete: **3,644 statements and 2,579 branches** remain
uncovered. These are local results.

## Numeric table reading order and large-block repair contracts

Added **24 deterministic cases** covering numeric grids, interleaved magazine
prose, overlapping scanned-column fragments, heuristic rejection boundaries,
line identity, and retained block metadata. Four initial grid failures showed
that the numeric-table repair sorted by x then ascending y, starting at the
bottom of the first column despite promising row-wise reading order. It now
uses the existing row-band ordering kernel: top-to-bottom rows, left-to-right
cells, with tolerance for small vertical differences. Positive controls cover
both row-major and column-major source sequences and 20/25-column grids.

The locked full workspace and differential suite with real Tesseract and veraPDF
passed **7,407 tests, no skips, in 430.75s**. Block layout now has **98.23% statement
and 96.15% branch coverage**. Workspace coverage increased
**90.34% → 90.50% statements** and **82.43% → 82.60% branches**:
**35,152 / 38,844 statements** and **12,357 / 14,960 branches**. All four source
roots, vendor-only omissions, and **554 excluded lines** remain unchanged. Both
exact coverage ratchets and all pre-push quality gates passed, including mypy, ty,
and import contracts. The static unused-symbol scan returned no candidates;
that heuristic does not prove the absence of unused code.
The 100% goal remains incomplete: **3,692 statements and 2,603 branches** remain
uncovered. These are local results.

## LlamaIndex font contracts and shared CMap recovery

Added **62 deterministic cases** for CID width ranges and overrides, malformed
width entries, simple-font metric truncation, glyph-name aliases, encoding
fallbacks, mapped-text widths, Type3 interpretability, and optional CMap recovery.
The shared reader CMap parser already retains explicit mappings when codespace
metadata is malformed. The facade's duplicate codespace-stripping retry instead
turned one invalid empty-map case into an empty CMap object. Removed that retry
and its tokenizer imports: invalid optional maps now return None so declared/base
encoding remains available. Explicit usable mappings remain preserved. The net
production denominator shrank by **26 statements and 10 branches**.

The locked full workspace and differential suite with real Tesseract and veraPDF
passed **7,383 tests, no skips, in 432.82s**. Workspace coverage increased
**90.22% → 90.34% statements** and **82.26% → 82.43% branches**:
**35,089 / 38,842 statements** and **12,332 / 14,960 branches**. All four source
roots, vendor-only omissions, and **554 excluded lines** remain unchanged. Both
exact coverage ratchets and all pre-push quality gates passed, including mypy, ty,
and import contracts. The static unused-symbol scan returned no candidates; its
reference heuristic does not prove the absence of unused code.
The 100% goal remains incomplete: **3,753 statements and 2,628 branches** remain
uncovered. These are local results.

## Pdfplumber layout and annotation projections

Added **20 differential cases** against installed pdfplumber, using deterministic
PDFs and pypdf-authored page boxes/link annotations. They compare layout-object
text and bounds with layout disabled or enabled, optional art/trim/bleed box values
(treating absent reference attributes as None), and annotation/link coordinates
on full and cropped pages. No production changes were needed.

The focused coverage run passed **20 tests in 11.93s**, appended to the existing
workspace coverage from the verified **7,243-test full run** and subsequent
focused tests on unchanged production sources. This was not another full-suite
run. Workspace coverage increased **90.07% → 90.22% statements** and
**82.09% → 82.26% branches**: **35,066 / 38,868 statements** and
**12,314 / 14,970 branches**. All four source roots, vendor-only omissions, and
**554 excluded lines** remain unchanged. Both exact coverage ratchets and all
pre-push quality gates passed, including both type checkers and import contracts.
The static unused-symbol scan returned no candidates; that heuristic does not
prove the absence of unused code.
The 100% goal remains incomplete: **3,802 statements and 2,656 branches** remain
uncovered. These are local results.

## Capture resource identity and recovery transitions

Added **58 deterministic cases** for indirect font identity, sorted subset-font
companions, cache reuse across equivalent page resources, scope-local reuse of
direct/stream fonts, and isolation when sibling dependencies change. Complete
capture states also exercise font lookup/resolution failures, color fallbacks,
invalid component handling, retained paint state, and shading/tiling pattern
selection. Pattern tests distinguish colored and uncolored paints and reject
missing geometry or zero steps while preserving negative tiling steps.
No production changes were needed for these contracts.

The focused coverage run passed **58 tests in 7.31s**, appended to the verified
**7,243-test full workspace/differential run** on unchanged production sources;
this was not another full-suite run. Workspace coverage increased
**89.97% → 90.07% statements** and **81.92% → 82.09% branches**:
**35,008 / 38,868 statements** and **12,289 / 14,970 branches**. All four source
roots, vendor-only omissions, and **554 excluded lines** remain unchanged. Both
exact coverage ratchets and all pre-push quality gates passed, including mypy, ty,
and import contracts. The static unused-symbol scan returned no candidates; that
heuristic does not prove the absence of unused code.
The 100% goal remains incomplete: **3,860 statements and 2,681 branches** remain
uncovered. These are local results.

## Color-space recovery and Pattern branch simplification

Added **78 deterministic cases** for nested color descriptions, cyclic bases and
alternates, built-in singleton forms, indexed palette recovery, ICC component
counts, calibrated parameter copying, image bit depth, and conservative DeviceN
paint probing. Three initial failures showed NaN/infinity passing through the
shared numeric color-parameter parser. That parser now rejects non-finite values,
covering component ranges and calibrated-image parameters without changing host
scalar coercion. The later Pattern branch now handles only the two-element form:
the earlier built-in branch already handles its one-element form. This removes
redundant optional-base handling.

The locked full workspace and differential suite with real Tesseract and veraPDF
passed **7,243 tests, no skips, in 430.15s**. Color-space recovery now has
**196 / 196 statements and 94 / 94 branches covered**. Workspace coverage increased
**89.82% → 89.97% statements** and **81.62% → 81.92% branches**:
**34,971 / 38,868 statements** and **12,264 / 14,970 branches**. All four source
roots, vendor-only omissions, and **554 excluded lines** remain unchanged. Both
exact coverage ratchets and all pre-push quality gates passed, including mypy, ty,
and import contracts. The static unused-symbol scan returned no candidates; its
reference heuristic does not establish that every remaining symbol is necessary.
The 100% goal remains incomplete: **3,897 statements and 2,706 branches** remain
uncovered. These are local results.

## Stream recovery and incomplete compression headers

Added **146 deterministic cases** for ASCIIHex junk/odd nibbles, RunLength
truncated literals and repeats, ASCII85 delimiters and invalid groups, native and
Python LZW routes, real raw/zlib/gzip Flate data, checksum recovery, and content
recognition across strings, names, containers, memoryview slices, and scan limits.
Testing exposed a one-byte invalid Flate stream being accepted as empty output
while a wrapped decoder was still waiting for the rest of its header. Recovery
now requires at least two bytes for an incomplete wrapped stream, retaining the
existing eight-byte minimum for incomplete raw data. Tests check every possible
one-byte prefix in all three decoder modes and preserve complete empty streams.

The locked full workspace and differential suite with real Tesseract and veraPDF
passed **7,165 tests, no skips, in 430.63s**. The recovery module now has
**173 / 173 statements and 76 / 76 branches covered**. Workspace coverage increased
**89.69% → 89.82% statements** and **81.45% → 81.62% branches**:
**34,912 / 38,867 statements** and **12,219 / 14,970 branches**. All four source
roots, vendor-only omissions, and **554 excluded lines** remain unchanged. Both
exact coverage ratchets and all pre-push quality gates passed, including both
type checkers and import contracts. The static unused-symbol scan returned no
candidates; it is not proof that every remaining symbol is necessary.
The 100% goal remains incomplete: **3,955 statements and 2,751 branches** remain
uncovered. These are local results.

## Polygon painting across scanline, sampled, and analytic routes

Added **356 deterministic cases** comparing painted pixels with explicit rectangle
and hole geometry. They cover both winding rules, contour orientation, rectangular
and triangular clips, transparent and partial opacity, normal and Multiply modes,
scalar and NumPy span routes, precomputed edges, the opaque-black shortcut, and
isolated/non-isolated group compositing. Quarter-pixel boundaries provide an
independent exact-area oracle for both analytic and 4x4 sampled coverage.
No production changes were needed for these contracts.

The focused coverage run passed **356 tests in 11.70s**, appended to the verified
**6,663-test full workspace/differential run** on unchanged production sources;
this was not another full-suite run. Workspace coverage increased
**89.36% → 89.69% statements** and **81.04% → 81.45% branches**:
**34,858 / 38,866 statements** and **12,193 / 14,970 branches**. The path-fill
module now reaches **95.93% statements / 88.73% branches**. All four source roots,
vendor-only omissions, and **554 excluded lines** remain unchanged. Both exact
coverage ratchets and all pre-push quality gates passed, including mypy, ty, and
import contracts. The static unused-symbol scan returned no candidates; that
heuristic does not prove the absence of dead code.
The 100% goal remains incomplete: **4,008 statements and 2,777 branches** remain
uncovered. These are local results.

## Finite numeric recovery in capture state

Added **55 deterministic cases** for numeric errors, state preservation, line-width
and miter clamping, dash recovery, scope underflow, incomplete paths, matrix contexts,
font-size errors, and accepted tolerant numeric encodings. Seven failures with the
complete capture state exposed direct float/int shortcuts admitting NaN/infinity
or raising OverflowError. Capture now uses one host conversion path followed by an
explicit finite check, so existing recovery handlers preserve prior state. Host
scalar semantics and tolerant exponent-string support remain unchanged.

The locked full workspace and differential suite with real Tesseract and veraPDF
passed **6,663 tests, no skips, in 426.68s**. Workspace coverage increased
**89.28% → 89.36% statements** and **80.93% → 81.04% branches**:
**34,731 / 38,866 statements** and **12,131 / 14,970 branches**. All four roots,
vendor-only omissions, and **554 excluded lines** remain unchanged. Both exact
ratchets and all pre-push quality gates passed, including both type checkers and
import contracts. The static unused-symbol scan returned no candidates.
The 100% goal remains incomplete: **4,135 statements and 2,839 branches** remain
uncovered. These are local results.


## Pypdf metadata projection and encrypted-reader lifecycle

Added **51 differential cases** using independently pypdf-written RC4-40, RC4-128,
AES-128, AES-256-R5, and AES-256 files. Tests cover user/owner passwords, failed
attempts, locked-page access, constructor authentication, page materialization,
source forms, and plain/encrypted metadata. Twenty initial failures exposed that
facade metadata used bare Title keys and included native info/xmp wrapper fields.
Metadata now projects the PDF Info dictionary directly with slash-prefixed keys,
matching pypdf, instead of merging the native structured metadata wrapper.

The locked full workspace and differential suite with real Tesseract and veraPDF
passed **6,608 tests, no skips, in 421.05s**. Workspace coverage increased
**89.21% → 89.28% statements** and **80.87% → 80.93% branches**:
**34,700 / 38,867 statements** and **12,117 / 14,972 branches**. All four roots,
vendor-only omissions, and **554 excluded lines** remain unchanged. Both exact
ratchets and all pre-push quality gates passed, including both type checkers and
import contracts. The static unused-symbol scan returned no candidates.
The 100% goal remains incomplete: **4,167 statements and 2,855 branches** remain
uncovered. These are local results.


## Recorded font contour contracts

Added **38 deterministic cases** comparing quadratic/cubic sampling with independent
de Casteljau evaluation. Recording tests cover implicit quadratic midpoints, chained
cubics, translated controls, endpoint preservation, explicit/implicit closure,
unstarted and incomplete commands, and single-point contour rejection.
No production changes were needed.

The 38 tests passed in **9.23s**, appending coverage to the verified locked
**6,384-test** full workspace report with real Tesseract and veraPDF and subsequent
96 primitive-shape and 39 TrueType recovery cases. Production sources are unchanged
since that full run; the full suite was not rerun for this test-only change.
Workspace coverage increased **89.15% → 89.21% statements** and
**80.78% → 80.87% branches**: **34,675 / 38,868 statements** and
**12,108 / 14,972 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan
returned no candidates. The 100% goal remains incomplete: **4,193 statements and
2,864 branches** remain uncovered. These are local results.


## TrueType CMap and glyph recovery contracts

Added **39 deterministic cases** for Unicode/symbol CMap precedence, Macintosh
fallback, generated glyph names, explicit byte-code aliases, Unicode inversion
ranking, raw glyph-header bounds, unusable locations, and damaged glyph-order
recovery. Mapping tests use real fontTools table objects; a controlled damaged-font
boundary checks fallback order generation. No production changes were needed.

The 39 tests passed in **9.09s** with coverage and again after making the fixture's
CMap constructor concrete for type checking. Coverage was appended to the verified
locked **6,384-test** full workspace report with real Tesseract and veraPDF and the
subsequent 96 primitive-shape cases. Production sources are unchanged since that
full run; the full suite was not rerun for this test-only change. Workspace coverage
increased **89.09% → 89.15% statements** and **80.70% → 80.78% branches**:
**34,649 / 38,868 statements** and **12,094 / 14,972 branches**. All four roots,
vendor-only omissions, and **554 excluded lines** remain unchanged. Both exact
ratchets and all pre-push quality gates passed, including both type checkers and
import contracts. The static unused-symbol scan returned no candidates.
The 100% goal remains incomplete: **4,219 statements and 2,878 branches** remain
uncovered. These are local results.


## Primitive-shape raster contracts

Added **96 deterministic cases** comparing circle rendering with explicit pixel-center
geometry and rectangle painting with explicit clip boundaries. Tests force scalar
and array circle routing, include rectangular/triangular clips, fractional centers,
zero-radius/off-page shapes, opaque/partial/zero alpha, and four rectangle blend modes.
All final tests pass; no production changes were needed.

The 96 tests passed in **12.29s** with coverage and again after style/type-only edits.
Coverage was appended to the verified locked **6,384-test** full workspace report
with real Tesseract and veraPDF. Production sources are unchanged since that full
run; the full suite was not rerun for this test-only change. Workspace coverage
increased **88.88% → 89.09% statements** and **80.44% → 80.70% branches**:
**34,627 / 38,868 statements** and **12,082 / 14,972 branches**. All four roots,
vendor-only omissions, and **554 excluded lines** remain unchanged. Both exact
ratchets and all pre-push quality gates passed, including both type checkers and
import contracts. The static unused-symbol scan returned no candidates.
The 100% goal remains incomplete: **4,241 statements and 2,890 branches** remain
uncovered. These are local results.


## Consistent bitmap height across rendering routes

Added **24 deterministic cases** comparing glyph bitmap rendering with explicit
cell geometry across opaque/partial alpha, integer/fractional placement, missing or
excess rows, excess column bits, inferred dimensions, and unpaintable inputs.
Three initial failures showed that fallback routes painted rows beyond the declared
bitmap height, while aligned opaque rendering already truncated them. Fallback
iteration now uses the same height bound; tests assert pixel equivalence and no
paint below the declared glyph box.

The locked full workspace and differential suite with real Tesseract and veraPDF
passed **6,384 tests, no skips, in 465.09s**. Workspace coverage increased
**88.68% → 88.88% statements** and **80.26% → 80.44% branches**:
**34,547 / 38,868 statements** and **12,043 / 14,972 branches**. All four roots,
vendor-only omissions, and **554 excluded lines** remain unchanged. Both exact
ratchets and all pre-push quality gates passed, including both type checkers and
import contracts. The static unused-symbol scan returned no candidates.
The 100% goal remains incomplete: **4,321 statements and 2,929 branches** remain
uncovered. These are local results.


## Shared lazy font recovery in x-ray inspection

Six initial regression failures showed that both x-ray glyph recovery routes used
setdefault with an eagerly evaluated _recover_font call, reparsing the raw font even
when its result was cached. A shared page-local lookup now resolves each font once
and caches misses as well as successes. Tests cover control-character recovery,
operand overrides, and mixed routes, checking both lookup counts and recovered text.
A search found no other obvious recovery calls used as setdefault defaults.

The locked full workspace and differential suite with real Tesseract and veraPDF
passed **6,360 tests, no skips, in 462.36s**. Workspace coverage increased
**88.62% → 88.68% statements** and **80.17% → 80.26% branches**:
**34,470 / 38,868 statements** and **12,016 / 14,972 branches**. All four roots,
vendor-only omissions, and **554 excluded lines** remain unchanged. Both exact
ratchets and all pre-push quality gates passed, including both type checkers and
import contracts. The static unused-symbol scan returned no candidates.
The 100% goal remains incomplete: **4,398 statements and 2,956 branches** remain
uncovered. These are local results.


## X-ray raw recovery and occlusion contracts

Added **44 deterministic cases** for paint ordering, same-fill exceptions, strict
occlusion thresholds, raw unindexed font/CMap recovery, unusable font candidates,
multi-operand text recovery, uniform raster crops, and top-level syntax validation.
Font recovery uses real PDF documents with appended unindexed objects. Boundary
checks use controlled records; these are contract tests, not upstream differential
parity claims. No production changes were needed.

The 44 tests passed in **9.82s** with coverage and again after a formatting-only
fixture edit. Coverage was appended to the verified locked **5,995-test** full
workspace report with real Tesseract and veraPDF and subsequent 69 outline, 76 codec,
94 blend, 43 word-grouping, and 33 diagnostic cases. Production sources are unchanged
since that full run; the full suite was not rerun for this test-only change.
Workspace coverage increased **88.51% → 88.62% statements** and
**80.01% → 80.17% branches**: **34,441 / 38,864 statements** and
**12,002 / 14,970 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan
returned no candidates. The 100% goal remains incomplete: **4,423 statements and
2,968 branches** remain uncovered. These are local results.


## Layout diagnostic contracts

Added **33 deterministic cases** for nonpositive run/advance/ink geometry, visibility
and whitespace controls, unsupported Unicode, confidence thresholds, glyph-cluster
text mismatch and geometry, writing-axis checks, and page issue locations/counts.
Valid text and empty lines provide controls against false positives. These helpers
remain used by public page diagnostics. All tests pass without production changes.

The 33 tests passed in **9.14s**, appending coverage to the verified locked
**5,995-test** full workspace report with real Tesseract and veraPDF and subsequent
69 outline, 76 codec, 94 blend, and 43 word-grouping cases. Production sources are
unchanged since that full run; the full suite was not rerun for this test-only change.
Workspace coverage increased **88.24% → 88.51% statements** and
**79.62% → 80.01% branches**: **34,399 / 38,864 statements** and
**11,978 / 14,970 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan
returned no candidates. The 100% goal remains incomplete: **4,465 statements and
2,992 branches** remain uncovered. These are local results.


## Differential word grouping contracts

Added **43 cases** comparing the pdfplumber facade with the installed reference
using controlled glyph records. Tests cover upright/rotated text, punctuation,
ligatures, whitespace, extra font attributes, tolerances, right-to-left and
bottom-to-top reading, input order, page projection, and source-character identity.
All final cases pass without production changes; these cases establish parity for
the tested options, not the complete word-extraction API.

The 43 tests passed in **9.79s**, appending coverage to the verified locked
**5,995-test** full workspace report with real Tesseract and veraPDF and the
subsequent 69 outline, 76 codec, and 94 blend cases. Production sources are unchanged
since that full run; the full suite was not rerun for this test-only change.
Workspace coverage increased **88.20% → 88.24% statements** and
**79.57% → 79.62% branches**: **34,293 / 38,864 statements** and
**11,919 / 14,970 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan
returned no candidates. The 100% goal remains incomplete: **4,571 statements and
3,051 branches** remain uncovered. These are local results.


## Array compositing and alpha-stage contracts

Added **94 deterministic cases** comparing Normal, Multiply, and Screen array
compositing against scalar source-over expectations. Coverage includes transparent,
partial, and opaque source/backdrop pixels, strided destinations and untouched
neighbors, separate rounding at each group-alpha stage, source immutability,
normal/general group equivalence, and coverage-alpha caps. Zero coverage preserves
the destination, including hidden RGB. All cases pass without production changes.

The 94 tests passed in **9.49s**, appending coverage to the verified locked
**5,995-test** full workspace report with real Tesseract and veraPDF and the
subsequent 69 outline and 76 codec cases. Production sources are unchanged since
that full run; the full suite was not rerun for this test-only change. Workspace
coverage increased **88.05% → 88.20% statements** and **79.45% → 79.57% branches**:
**34,278 / 38,864 statements** and **11,912 / 14,970 branches**. All four roots,
vendor-only omissions, and **554 excluded lines** remain unchanged. Both exact
ratchets and all pre-push quality gates passed, including both type checkers and
import contracts. The static unused-symbol scan returned no candidates.
The 100% goal remains incomplete: **4,586 statements and 3,058 branches** remain
uncovered. These are local results.


## Host codec and predictor contracts

Added **76 deterministic cases** for sample precision conversion, signed clipping,
float rounding, contiguous output, preserved uint16 samples, unsupported decoder
shapes/types, invalid-versus-unsupported errors and exception causes, and bounded
JPX thread configuration. Real PNG decoding checks exact bytes at 1/2/4/8/16 bits;
TIFF tests cover per-channel accumulation, modular overflow, row resets, truncation,
and sub-byte row padding using independently packed expected samples.
All cases passed without production changes.

The 76 tests passed in **9.55s**, appending coverage to the verified locked
**5,995-test** full workspace report with real Tesseract and veraPDF and the
subsequent 69 outline cases. Production sources are unchanged since that full run;
the full suite was not rerun for this test-only change. Workspace coverage increased
**87.88% → 88.05% statements** and **79.25% → 79.45% branches**:
**34,221 / 38,864 statements** and **11,893 / 14,970 branches**. All four roots,
vendor-only omissions, and **554 excluded lines** remain unchanged. Both exact
ratchets and all pre-push quality gates passed, including both type checkers and
import contracts. The static unused-symbol scan returned no candidates.
The 100% goal remains incomplete: **4,643 statements and 3,077 branches** remain
uncovered. These are local results.


## Outline geometry and text paint contracts

Added **69 deterministic cases** comparing scalar and cached-array glyph outlines
under identity, rotation/scaling, fractional shear/translation, and collapsed
transforms. Controls include empty, single-point, closed, and degenerate contours,
exact bounds and edge geometry, outline code precedence, missing inputs, and bitmap
fallback. All eight text render modes are exercised with painting enabled/disabled
and visible/invisible glyphs, distinguishing paint from accumulated clipping.
The scalar fallback and cached-array route agree; no production change was needed.

All 69 tests passed in **9.88s**, appending coverage to the verified locked
**5,995-test** full workspace report with real Tesseract and veraPDF. Production
sources are unchanged since that full run; the full suite was not rerun for this
test-only change. Workspace coverage increased **87.80% → 87.88% statements** and
**79.06% → 79.25% branches**: **34,155 / 38,864 statements** and
**11,864 / 14,970 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan
returned no candidates. The 100% goal remains incomplete: **4,709 statements and
3,106 branches** remain uncovered. These are local results.


## Pattern command translation keeps cached edges aligned

Added **21 deterministic cases** for translated path geometry and cached edges,
pixel equivalence with path-derived edges, source immutability, soft-mask offsets,
explicit blend modes, and axial/radial shading coordinates. Five initial failures
showed that pattern-cell translation moved a path and its bounds but retained the
original edge-array coordinates. Translation now produces a translated edge array
alongside the translated path, preserving the shared capture. Integer, fractional,
negative, single-axis, and zero translations have explicit controls.

The locked full workspace suite with real Tesseract and veraPDF passed **5,995 tests,
no skips, in 441.64s**. Workspace coverage increased **87.74% → 87.80% statements**
and **78.97% → 79.06% branches**: **34,123 / 38,864 statements** and
**11,836 / 14,970 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan
returned no candidates. The 100% goal remains incomplete: **4,741 statements and
3,134 branches** remain uncovered. These are local results.


## CID Unicode mapping and voting contracts

Added **42 deterministic cases** for effective CMap inversion, later-range and explicit
mapping precedence, codespace filtering, orientation-specific votes, weighted ties,
opposite-orientation and zero-weight fallback, overrides, cached misses, legacy
encodings, and packaged collection lookup identity. Small controlled maps exercise
real Unicode decoding and ranking; packaged GB1 maps provide integration controls.
The distinct voting tiers remain necessary. No production changes were needed.
The CID Unicode recovery module now has **100% statement and branch coverage**
(101 statements and 46 branches).

All 42 tests passed in **6.42s**, appending coverage to the verified locked
**5,909-test** full workspace report with real Tesseract and veraPDF and the subsequent
23 font-width cases. Production sources are unchanged since that full run; the full
suite was not rerun for this test-only change. Workspace coverage increased
**87.49% → 87.74% statements** and **78.60% → 78.97% branches**:
**34,100 / 38,864 statements** and **11,822 / 14,970 branches**. All four roots,
vendor-only omissions, and **554 excluded lines** remain unchanged. Both exact
ratchets and all pre-push quality gates passed, including both type checkers and
import contracts. The static unused-symbol scan returned no candidates.
The 100% goal remains incomplete: **4,764 statements and 3,148 branches** remain
uncovered. These are local results.


## Shared finite-number validation for native font widths

Added **90 deterministic cases** covering compact, sparse, and range CID widths,
invalid numeric values, coercion, clipping at CID boundaries, resynchronization,
last-assignment precedence, vertical metric triples, and simple-font defaults.
Eight initial failures showed that compact and sparse arrays admitted NaN/infinity
and raised OverflowError for huge integers while range entries already recovered.
All three forms now use the same finite-number parser. Valid contiguous entries
retain compact storage; invalid entries are skipped without losing later widths.
Strict spec parsing remains unchanged. Both native font-width modules now have
**100% statement and branch coverage** (196 statements and 92 branches together).

The locked full workspace suite with real Tesseract and veraPDF passed **5,909 tests,
no skips, in 217.01s**. After that run, 23 additional test-only boundary/vertical
cases were added; all 90 new cases passed in **5.89s**, appending coverage against
unchanged production sources. Workspace coverage increased **87.14% → 87.49%
statements** and **78.06% → 78.60% branches**: **34,004 / 38,864 statements** and
**11,766 / 14,970 branches**. The consolidation removed three statements and four
branches. All four roots, vendor-only omissions, and **554 excluded lines** remain
unchanged. Both exact ratchets and all pre-push quality gates passed, including mypy,
ty, and import contracts. The static unused-symbol scan returned no candidates.
The 100% goal remains incomplete: **4,860 statements and 3,204 branches** remain
uncovered. These are local results.


## Source ownership and security alias contracts

Added **36 deterministic cases** for security-name alias collisions in either order,
nested and cyclic security dictionaries, supported binary source forms, borrowed file
ownership and position, empty sources, unavailable file descriptors and positions,
and read failures. The retained fallback routes serve supported reader protocols;
these tests did not establish additional dead code. Production sources are unchanged.

All 36 tests passed in **6.20s**, with coverage appended to the verified report from
the locked **5,713-test** full workspace run with real Tesseract and veraPDF and the
subsequent 39 table-section and 54 scanline cases. The full suite was not rerun for
this test-only change. Workspace coverage increased **87.05% → 87.14% statements**
and **77.94% → 78.06% branches**: **33,869 / 38,867 statements** and
**11,689 / 14,974 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including mypy, ty, and import contracts. The static unused-symbol scan returned no
candidates; that is not proof that no dead code remains. The 100% goal is incomplete:
**4,998 statements and 3,285 branches** remain uncovered. These are local results.


## Scanline crossing and winding contracts

Added **54 deterministic cases** comparing scalar, row-at-a-time, and batched crossing
routes with explicit rectangle and diagonal intersections. Cases cover half-open
scanline boundaries, edge order, translations, empty rows/edges, and the memory-limit
fallback. Independent occupied-interval expectations distinguish holes, overlapping
contours, shared boundaries, and coincident edges under nonzero/even-odd fill rules.
Both batching and its bounded-memory fallback remain supported and necessary.

All 54 tests passed in **9.46s** with coverage appended to the verified report.
Production sources are unchanged since the locked **5,713-test** full workspace run
with real Tesseract and veraPDF; the report already contained the subsequent 39 table
section cases. Combined workspace coverage increased **87.03% → 87.05% statements**
and **77.90% → 77.94% branches**: **33,835 / 38,867 statements** and
**11,670 / 14,974 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan remains
empty. The full 100% goal remains incomplete: **5,032 statements and 3,304 branches**
remain uncovered. These are local results.


## Table section and prose-classification contracts

Added **39 deterministic cases** for section boundaries, row identity, cell geometry,
confidence/metadata preservation, title/caption placement, repeated identical headers,
spanning titles, and positive controls distinguishing prose grids from numeric tables.
All cases pass without production changes.

The 39 tests passed in **10.08s** with coverage appended to the verified report.
Production sources are unchanged since the locked **5,713-test** full workspace run
with real Tesseract and veraPDF. Combined workspace coverage increased
**87.01% → 87.03% statements** and **77.86% → 77.90% branches**:
**33,825 / 38,867 statements** and **11,665 / 14,974 branches**.
All four roots, vendor-only omissions, and **554 excluded lines** remain unchanged.
Both exact ratchets and all pre-push quality gates passed, including both type checkers
and import contracts. The static unused-symbol scan remains empty. The full 100% goal
remains incomplete: **5,042 statements and 3,309 branches** remain uncovered.
These are local results.


## Retired facade-local snapshot mutation helpers

A manual call/export audit found four unexported, unreferenced StructuredState methods:
replace_pages, delete_page, update_metadata, and apply_redactions. Removed that orphan
mutation chain and its unused Sequence import (**44 source lines removed**). The unused
redaction implementation also incorrectly pooled rectangles across pages; it was not a
supported facade redaction API. Shared synthetic snapshot construction remains necessary:
pikepdf imports StructuredState and uses it for new documents and page-list mutations.

Added **34 differential cases** comparing rectangle accessors with installed pypdf and
synthetic page-list insert/replace/delete/slice behavior with installed pikepdf. These
verify the retained cross-facade path through real public facade entry points.

The locked full workspace suite with real Tesseract and veraPDF passed **5,713 tests,
no skips, in 536.23s**. Workspace coverage increased **86.91% → 87.01% statements**
and **77.85% → 77.86% branches**: **33,818 / 38,867 statements** and
**11,659 / 14,974 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The lexical unused-symbol scan remains
empty; the manual audit found the orphan method chain that it did not identify.
The full 100% goal remains incomplete: **5,049 statements and 3,315 branches** remain
uncovered. These are local results.


## Table merge confidence and continuation contracts

Added **32 deterministic cases** for adjacent-table confidence, repeated headers,
row indexes, cell spans and geometry, title/caption selection, metadata preservation,
merge rejection, missing geometry, and wrapped stream-row text/bounds.

Seven confidence controls initially failed because zero was treated as missing by
`confidence or 1.0`. Merging now takes the minimum supplied confidence, defaulting to
1.0 only when both are absent. Consolidated the duplicated bounding-box guard and
corrected historical comments that inaccurately described different column counts
as mergeable. Explicit zero confidence survives the merge.

The locked full workspace suite with real Tesseract and veraPDF passed **5,679 tests,
no skips, in 567.43s**. Workspace coverage increased **86.86% → 86.91% statements**
and **77.77% → 77.85% branches**: **33,789 / 38,878 statements** and
**11,657 / 14,974 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan remains
empty. The full 100% goal remains incomplete: **5,089 statements and 3,317 branches**
remain uncovered. These are local results.


## Decoded-image normalization contracts

Added **72 deterministic cases** for gray/RGB JPX opacity selectors at 8/16-bit
precision, zero/partial/full alpha, premultiplication reversal, ignored opacity,
invalid component layouts and selectors, precision endpoints, and native sample storage
validation. Version controls cover JPX Decode with/without explicit ColorSpace under
PDF 1.7, PDF 2.0, unknown versions, and absent context. These operate on explicit
sample arrays without codec mocks or external fixtures and preserve the input samples.

All 72 cases passed in **13.52s** with coverage appended to the verified report.
Production sources are unchanged since the locked **5,575-test** full workspace run
with real Tesseract and veraPDF. No production correction was needed for these contracts.

Combined workspace coverage increased **86.75% → 86.86% statements** and
**77.63% → 77.77% branches**: **33,768 / 38,878 statements** and
**11,646 / 14,974 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan remains
empty. The full 100% goal remains incomplete: **5,110 statements and 3,328 branches**
remain uncovered. These are local results.


## Image preparation and declared-filter routing

Added **59 deterministic cases** for canonical layouts and read-only samples, stencil
row padding/polarity, malformed masks, 1/2/4/8/16-bit matte alpha decoding, soft-mask
scaling and color-key precedence, and source dictionary preservation.

A damaged Flate soft mask exposed encoded bytes being interpreted as alpha when their
length matched the gray/RGB sample-count heuristic. The same shortcut also corrupted
valid Flate images whose encoded and decoded lengths happened to coincide. Combined
the raw-byte shortcuts behind an unfiltered-data condition: declared filters now run
before sample-length recovery. Three valid compressed-image controls verify exact pixels;
the damaged mask is discarded while its valid parent image remains available.

The locked full workspace suite with real Tesseract and veraPDF passed **5,575 tests,
no skips, in 422.51s**. Workspace coverage increased **86.63% → 86.75% statements**
and **77.46% → 77.63% branches**: **33,726 / 38,878 statements** and
**11,625 / 14,974 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan remains
empty. The full 100% goal remains incomplete: **5,152 statements and 3,349 branches**
remain uncovered. These are local results.


## Attachment projection contracts

Added **39 deterministic cases** for Unicode/legacy filename and stream precedence,
name-tree aliases, shared filespec/stream identity, absent trees, malformed entries,
and strict versus recovered handling of independent siblings. The record builder
always returns a RawEmbeddedFile or raises; its return annotation now states that
contract, and the caller no longer checks an unreachable None result.

The locked full workspace suite with real Tesseract and veraPDF passed **5,516 tests,
no skips, in 349.66s**. Workspace coverage increased **86.56% → 86.63% statements**
and **77.37% → 77.46% branches**: **33,683 / 38,880 statements** and
**11,601 / 14,976 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan remains
empty. The full 100% goal remains incomplete: **5,197 statements and 3,375 branches**
remain uncovered. These are local results.


## Path geometry and square-cap coverage

Added **72 deterministic cases** for dash intervals and phase, odd and zero-length
patterns, duplicate/collinear vertices, closed-path seams, circle geometry, analytic
rectangle coverage, line caps, supplied array views, zero opacity, and empty targets.

Square caps incorrectly used the round-cap nearest-point distance, clipping their
corners. Butt and square caps now share rectangular projection coverage, with square
caps extending by half the line width. Removed the square-cap branch from the round-cap
loop. An independent convex-rectangle sampler verifies horizontal, vertical, diagonal,
and reversed square-cap lines. Zero-opacity calls retain their existing None alpha
return while still recording geometric shape coverage.

The locked full workspace suite with real Tesseract and veraPDF passed **5,477 tests,
no skips, in 357.60s**. Path kernels now cover **303/332 statements and 99/118 branches**.
Workspace coverage increased **86.35% → 86.56% statements** and
**77.03% → 77.37% branches**: **33,657 / 38,881 statements** and
**11,588 / 14,978 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan remains
empty. The full 100% goal remains incomplete: **5,224 statements and 3,390 branches**
remain uncovered. These are local results.


## Embedded font recovery and encoding contracts

Added **86 deterministic cases** for exact CFF table extraction, absent/truncated
OpenType tables, damaged Type 1/TrueType/CFF/OpenType programs, malformed descriptors,
font-name precedence, simple encoding defaults, CID Unicode and ligature overrides,
and CID system metadata normalization. Tests use actual parsers and synthetic bytes;
no font-parser mocks or external fixture downloads are needed. These contracts pass
without production changes, so the existing recovery paths remain intact.

All 86 tests passed in **6.53s** with coverage appended to the existing verified report.
Production sources are unchanged since the locked **5,302-test** full workspace run
with real Tesseract and veraPDF; that report already included the 17 subsequently
appended reconstruction boundary tests. Font decoding now covers **707/770 statements
and 239/298 branches**.

Combined workspace coverage increased **86.17% → 86.35% statements** and
**76.84% → 77.03% branches**: **33,575 / 38,883 statements** and
**11,539 / 14,980 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan remains
empty. The full 100% goal remains incomplete: **5,308 statements and 3,441 branches**
remain uncovered. These are local results.


## Reconstruction atom invariants and geometry boundaries

Added **57 deterministic cases** for text-atom preservation, Unicode whitespace and
control cleanup, glyph clusters, superscript thresholds, stacked-fraction geometry,
long-line duplicate-history trimming, tiny-footer suppression, and rotated-cell spacing.
For nonempty normalized text, atom construction always emits at least one nonempty
atom. The builder now relies on that invariant: removed redundant empty/whitespace
checks and the always-true emitted-run flag. No output policy changed.

The locked full workspace suite with real Tesseract and veraPDF passed **5,302 tests,
no skips, in 281.55s**. With production sources unchanged, the 17 boundary cases added
after collection passed and their coverage was appended to the full-run database.
The other 40 new cases and 19 existing reconstruction contracts also passed together.
Reconstruction now covers **445/513 statements and 220/276 branches**.

Combined workspace coverage increased **86.06% → 86.17% statements** and
**76.61% → 76.84% branches**: **33,504 / 38,883 statements** and
**11,510 / 14,980 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan remains
empty. The full 100% goal remains incomplete: **5,379 statements and 3,470 branches**
remain uncovered. These are local results.


## Navigation recovery and visibility consolidation

Added **74 deterministic cases** for page-label ranges and reader defaults, nested
and cyclic destination dictionaries, alias identity, unresolved names, outline ordering,
malformed siblings/children, strict versus recovered traversal, and page lookup by
identity, structure parents, and source-content references.

Four destination controls initially raised RecursionError: a valid 1,500-level wrapper
chain and three dictionary cycles. Destination dictionary traversal is now iterative,
retains the original destination array, and rejects cycles with ValueError. Named-alias
resolution is unchanged. Optional-content ON/OFF overrides now share one ordered
validation/update loop, preserving OFF precedence and strict/recovery behavior.

The locked full workspace suite with real Tesseract and veraPDF passed **5,229 tests,
no skips, in 433.31s**. A subsequent variable rename resolved a mypy name collision
without changing behavior or line locations. All **108 navigation/visibility contract
tests** then passed with coverage appended, including the 33 outline/lookup tests added
after full-suite collection. Document composition now covers **800/913 statements
and 344/432 branches**.

Combined workspace coverage increased **85.88% → 86.06% statements** and
**76.23% → 76.61% branches**: **33,474 / 38,894 statements** and
**11,483 / 14,988 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact ratchets and all pre-push quality gates passed,
including both type checkers and import contracts. The static unused-symbol scan remains
empty. The full 100% goal remains incomplete: **5,420 statements and 3,505 branches**
remain uncovered. These are local results.


## OCR image placement and optional-content contracts

Added **16 placement cases** checking all eight affine image orientations in grayscale
and RGB. Four initially failed: the two transpose-plus-single-flip transforms disagreed
with detection's target-to-source corner mapping. Raster orientation now derives its
transpose and flips from that same mapping, removing the duplicated seven-case dispatch.
Existing enum-only expected values for those two transforms were corrected using the
independent placement tests. Pixels and resolution are preserved. Raster preparation
now covers **296/296 statements and 74/74 branches**; the OCR package covers
**4,145/4,145 statements and 1,347/1,350 branches**.

Added **34 document cases** for optional-content base states, ON/OFF precedence, and
strict versus recovery behavior for malformed configuration dictionaries, arrays,
base-state values, entries, and names. These use the actual document resolver.

The locked full workspace suite with real Tesseract and veraPDF passed **5,154 tests,
no skips, in 545.43s**. With production sources unchanged, the 34 additional document
tests passed and their coverage was appended to that full-run database. The focused
OCR suite separately passed **735 tests**, including real-engine cases.

Combined workspace coverage increased **85.79% → 85.88% statements** and
**75.99% → 76.23% branches**: **33,404 / 38,898 statements** and
**11,430 / 14,994 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. Both exact coverage ratchets pass, as do all pre-push quality
gates, including both type checkers and import contracts. The static unused-symbol scan
remains empty. The full 100% goal remains incomplete: **5,494 statements and 3,564
branches** remain uncovered. These are local results.


## pdfplumber search spans and CSV contracts

Added **56 deterministic cases** for search spans, regex groups, compiled patterns,
case handling, omitted result fields, CSV quoting, precision, object/attribute selection,
and source-object preservation. Search cases compare with the installed pdfplumber
TextMap implementation, including multi-character glyphs and source identity.

Search now maps text offsets to their originating glyph objects, honors `main_group`
and `return_groups`, rejects incompatible compiled-pattern options, and avoids sending
compiled patterns through the literal fallback. Empty and whitespace-only matches are
filtered before building geometry, making the separate empty-character-span check
redundant. The existing formatted-text fallback remains; these tests do not establish
complete search/layout parity with pdfplumber. CSV tests establish current facade
contracts without claiming full reference serialization parity.

The locked full workspace suite with real Tesseract and veraPDF passed **5,138 tests,
no skips, in 500.83s**. Workspace coverage increased **85.66% → 85.79% statements**
and **75.82% → 75.99% branches**: **33,378 / 38,907 statements** and
**11,401 / 15,004 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged. The exact verified fractions ratchet the baseline.

A subsequent type-only cast and fixture annotations passed the 56 focused tests.
All pre-push quality gates passed, including both type checkers and import contracts;
the static unused-symbol scan remains empty.
The full 100% goal remains incomplete: **5,529 statements and 3,603 branches** remain
uncovered. These are local results.


## Encoding CMaps and font metric contracts

Added **86 cases** for malformed character/range mappings, notdef range semantics,
identity aliases, custom inheritance and cycles, mixed-width byte splitting, single-code
mapping, writing modes, spacing, and text advances with cached/uncached glyph inputs.
Aggregate advances agree with per-glyph sums within tight floating-point tolerance;
the implementation comment now accurately describes the historical operation-order
difference instead of claiming bit-identical arithmetic.

Three controls exposed nonfinite Type 1 Length1 metadata escaping font recovery.
Length conversion now occurs inside the existing failure boundary and catches overflow,
allowing standard-font decoding after an unusable embedded program. Encoding CMap recovery
now covers **78/78 statements and 32/32 branches**.

The locked full workspace suite with real Tesseract and veraPDF passed **5,082 tests,
no skips, in 390.42s**. Workspace coverage increased **85.47% → 85.66% statements**
and **75.48% → 75.82% branches**: **33,324 / 38,902 statements** and
**11,374 / 15,002 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged; exact verified fractions ratchet the baseline.

All pre-push quality gates passed, including both type checkers and import contracts.
The static unused-symbol scan is empty. The full 100% goal remains incomplete:
**5,578 statements and 3,628 branches** remain uncovered. These are local results.


## ToUnicode recovery and range expansion

Added **68 deterministic cases** for destination byte recovery, null preservation,
mapping-length precedence, malformed codespaces, later-definition precedence, partial
and excessive range arrays, malformed nested delimiters, numeric CID recovery, parent
inheritance and cycles, range limits, and Unicode replacement behavior.

Removed the redundant one-byte lookup after the decoder's mapping-length loop: constructed
maps already include every mapping-key length. Range expansion now normalizes its invariant
prefix once and varies only the final scalar. The full corpus suite preserves existing
recovery behavior. ToUnicode recovery covers **99.01% statements / 96.25% branches**;
range expansion covers **100% of both**.

The locked full workspace suite with real Tesseract and veraPDF passed **4,996 tests,
no skips, in 357.88s**. Workspace coverage increased **85.23% → 85.47% statements**
and **75.16% → 75.48% branches**: **33,250 / 38,902 statements** and
**11,323 / 15,002 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged; exact verified fractions ratchet the baseline.

All pre-push quality gates passed, including both type checkers and import contracts.
The static unused-symbol scan is empty. The full 100% goal remains incomplete:
**5,652 statements and 3,679 branches** remain uncovered. These are local results.


## Recovered object-stream identity and bounded scanning

Added **39 cases** for embedded-object indexes, existing-entry preservation, recovery
limits, invalid/stale containers, bounded header searches across bytes and memoryview
representations, and EOF fallback selection. Five index controls initially failed:
recovered entries used the container generation as every embedded object's index.
Recovery now records each object's position in the parsed container, preserving positions
when earlier entries are skipped. The tests resolve each recovered index against its
expected object identity and value. An older trailer fixture now uses a PDF reference
for Root, matching its declared object type.

The locked full workspace suite with real Tesseract and veraPDF passed **4,928 tests,
no skips, in 441.68s**. Workspace coverage increased **85.20% → 85.23% statements**
and **75.11% → 75.16% branches**: **33,163 / 38,912 statements** and
**11,279 / 15,006 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged; exact verified fractions ratchet the baseline.
Recovery xref coverage is **93.72% / 91.38%**, with 38 statements and 25 branches remaining.

Ruff, formatting, both type checkers, and import contracts passed; the static unused-symbol
scan is empty. The full 100% goal remains incomplete: **5,749 statements and 3,727 branches**
remain uncovered. These are local results, including all compatibility corpus suites.


## Cross-reference recovery contracts

Added **123 deterministic cases** across fixed/loose entry formats, line endings,
missing markers, invalid numbers, subsection-count repair, missing trailer keywords,
stream widths/indexes/truncation, EOF recovery, nearby sections, compressed stream
salvage, and object-looking bytes inside valid stream payloads.

Two discrepancies became failing controls: fixed-width rows accepted negative offsets
that loose rows rejected; an overstated subsection count prevented recovery at a trailer
dictionary without its keyword. Both row parsers now share numeric validation, and
subsection parsing recognizes the same dictionary boundary as table parsing. Strict spec
parsing is unchanged. Recovery xref coverage is **92.41% statements / 88.97% branches**
(46 statements and 32 branches remain in the module).

The locked full workspace suite with real Tesseract and veraPDF passed **4,889 tests,
no skips, in 636.95s**. Workspace coverage increased **84.98% → 85.20% statements**
and **74.68% → 75.11% branches**: **33,155 / 38,913 statements** and
**11,271 / 15,006 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged; exact verified fractions ratchet the baseline.

Ruff, formatting, both type checkers, and import contracts passed. The static unused-symbol
scan is empty. The full 100% goal remains incomplete: **5,758 statements and 3,735 branches**
remain uncovered. These are local results, including the compatibility corpus suites.


## CFF offset validation and table boundaries

Added **54 cases** for FDSelect formats and ranges, FDArray dictionaries, Private/Subrs
relative offsets, malformed construction, and glyph/font-dictionary bounds. Eight new
cases initially failed: nonfinite offsets leaked OverflowError, empty Subrs operands
leaked IndexError, and extra FD table operands were accepted.

Top DICT, FDSelect, FDArray, and Subrs offsets now share finite, nonnegative,
single-integer validation. Private dictionary validation checks finiteness before integer
conversion. This preserves the documented ValueError boundary for malformed input and
keeps valid relative offsets and table projections unchanged. Font-program coverage is
now **96.22% statements / 93.65% branches** (25 statements and 24 branches remain).

The locked full workspace suite with real Tesseract and veraPDF passed **4,766 tests,
no skips, in 603.37s**. Workspace coverage increased **84.94% → 84.98% statements**
and **74.60% → 74.68% branches**: **33,067 / 38,912 statements** and
**11,208 / 15,008 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged; exact verified fractions ratchet the baseline.

The isolated spec suite passed **2,798 tests**. All pre-push checks passed, including
both type checkers and import contracts; the static unused-symbol scan is empty.
The full 100% goal remains incomplete: **5,845 statements and 3,800 branches** remain
uncovered. These are local results; elapsed suite time is not a performance benchmark.


## Type 2 interpreter and CFF binary contracts

Added **141 cases** for arithmetic and conditional operators, stack order, shared
transient storage and per-glyph reset, invalid indices and underflow, all flex variants,
axis-specific moves/curves, optional width handling, and the 48-operand limit. Path
callbacks assert exact emitted displacements and errors before geometry emission.
Operator expectations follow [Adobe Technical Note 5177](https://adobe-type-tools.github.io/font-tech-notes/pdfs/5177.Type2.pdf).
Undefined-input handling is characterized as implementation behavior, not a mandated default.
CFF tests independently encode number boundaries and INDEX tables, including empty
entries, all offset sizes, memoryviews, truncation, and signed/fixed-point values.

Horizontal/vertical moves now share width handling, validation, and stack cleanup.
Parallel-tangent curves share validation and first-curve offset handling while preserving
axis-specific displacements. This removes **29 net production lines** without changing
public entry points or recovery boundaries. Font-program coverage is **94.14% statements /
90.10% branches**, with **39 statements and 38 branches** remaining in that module.

The locked full workspace suite with real Tesseract and veraPDF passed **4,712 tests,
no skips, in 533.80s**. Workspace coverage increased **84.75% → 84.94% statements**
and **74.25% → 74.60% branches**: **33,057 / 38,916 statements** and
**11,200 / 15,014 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged; the exact verified fractions ratchet the baseline.

The isolated spec suite passed **2,744 tests**. Ruff, formatting, both type checkers,
import contracts, and pre-push checks passed. The static unused-symbol scan is empty.
The full 100% goal remains incomplete: **5,859 statements and 3,814 branches** remain
uncovered. These are local measurements; the longer run is not a performance benchmark.


## JBIG2 packed bitmap composition

Added **121 independent pixel-oracle and buffer-contract cases** across scalar, NumPy,
and dispatched composition. They cover OR/XOR, aligned and unaligned placement, all
clipping directions, partial bytes, padded destination strides, incomplete rows,
mutable views, non-contiguous arrays, and dtype conversion.

The contracts exposed a row-count bug: a uint16 array was counted using its storage
bytes even though conversion produces one uint8 per element. Counting elements now
matches conversion in both scalar and bulk paths. NumPy composition selects its bitwise
operator once, sharing aligned, masked-tail, and unaligned application logic.
The bitmap kernel now covers **95/95 statements and 40/40 branches**.

The locked full workspace suite with real Tesseract and veraPDF passed **4,571 tests,
no skips, in 236.32s**. Workspace coverage increased **84.59% → 84.75% statements**
and **74.04% → 74.25% branches**: **32,991 / 38,928 statements** and
**11,154 / 15,022 branches**. All four roots, vendor-only omissions, and **554 excluded
lines** remain unchanged; the exact verified fractions ratchet the baseline.

The isolated spec suite passed **2,603 tests**. Ruff, formatting, mypy, ty, and import
contracts passed; the static unused-symbol candidate scan is empty.
The full 100% goal remains incomplete: **5,937 statements and 3,868 branches** remain
uncovered. These measurements are local, not a claimed GitHub execution.


## Filter codec and predictor contracts

Added **176 deterministic cases** covering ASCII85 scalar/bulk/zero-shorthand decoding,
LZW code-width transitions and dictionary saturation, strict DecodeParms, and TIFF
sample depths, channels, row padding, and truncation. The standard-library ASCII85
encoder supplies an independent oracle; TIFF fixtures encode horizontal differences
from known sample rows. Existing PNG contracts cover all five row filters.

Consolidated bit extraction and ASCII85 validation/tail decoding, removed redundant
parameter branches, shared TIFF word accumulation, and kept PNG previous-row state
as a memoryview. Public kernel entry points and strict-wrapper error boundaries remain.
The three changed production modules have **100% statement and branch coverage**.

The locked full workspace suite with real Tesseract and veraPDF passed **4,450 tests,
no skips, in 222.09s**. Workspace coverage increased **84.17% → 84.59% statements**
and **73.44% → 74.04% branches**: **32,933 / 38,934 statements** and
**11,127 / 15,028 branches**. All four source roots, vendor-only omissions, and
**554 excluded lines** remain unchanged. Exact verified fractions ratchet the baseline.

The isolated spec environment, with core and OCR absent, passed **2,482 tests**.
All pre-push checks passed, including both type checkers and import contracts; no
source-shadowing binaries or static unused-symbol candidates were found.
The workspace 100% goal remains incomplete: **6,001 statements and 3,901 branches**
are still uncovered. These are local results; no new GitHub run is claimed.


## Structural consolidation and default complete-suite coverage

Implemented the six structural improvements: shared vector/image color conversion,
unified line classification, complete-suite CI coverage, PDF MAC interoperability tests,
a raster equivalence/failure matrix, and supported pdfplumber edge merging/image display.
The remaining obsolete image-kernel module was removed after verifying its callers were gone;
CMYK profile selection and failure recovery now share one implementation across sample depths.

The raster matrix demonstrated four cleanup failures: soft-mask preparation and group
composition left shape state changed; failed clip setup and scope composition left scopes
or suspended buffers active. Cleanup now restores those boundaries, including nested groups.
The short formula case now produces `2√x` regardless of inserted empty capture records.
Missing/unusable tint transforms consistently use subtractive grayscale recovery.

Added **251 deterministic cases** across these contracts. Existing independently generated
MAC fixtures are unchanged. Their checks cover both passwords, extracted text, six tampering
variants, strict byte ranges, CMS failures, supported digest algorithms, and rejection before
installing decryption state. PDFMiner recovery and lazy resource ownership remain facade policy.

The final locked full run with pinned Tesseract and real veraPDF enabled passed
**4,274 tests with no skips in 223.22s**. Coverage.py 7.16.1 reported:

| Package | Statements | Branches |
| --- | ---: | ---: |
| Core | 80.26% | 68.11% |
| OCR | 100.00% | 99.71% |
| Spec | 87.56% | 77.18% |
| Validate | 100.00% | 100.00% |
| **Workspace** | **84.17%** | **73.44%** |

Workspace coverage increased from **82.04% / 71.05%**. The report contains
**32,816 / 38,987 statements** and **11,062 / 15,062 branches** covered;
**6,171 statements and 4,000 branches remain uncovered**. All four source roots,
the vendor-only omit rule, and **554 excluded lines** remain unchanged.
The exact verified fractions are now the independent coverage floors.

A fresh spec-only environment, with core and OCR confirmed absent, passed **2,306 tests**.
The real veraPDF adapter run separately passed **48 cases**. Local veraPDF was 1.30.2 on
Java 26; CI remains configured for Java 17. The pinned OCR installer was exercised against
the checksum-verified model download; the locked Linux wheel also bundles Tesseract 5.5.1.
Ruff, mypy, ty, all 19 import contracts, full pre-push hooks, and all four distribution
builds passed. Built wheels contain no obsolete image-kernel module; the core wheel declares Pillow.
The static unused-symbol candidate scan is empty; this is not proof that all remaining
uncovered code is dead. The workspace 100% goal remains incomplete.

CI preserves the existing required status name, runs all authored suites by default,
combines workspace and real-validator coverage, publishes reports, and rejects either metric
regressing or an incomplete source inventory. GitHub execution awaits the next push;
the recorded measurements above are local, not a claimed Linux CI result.

## Workspace coverage completion: structure diagnostics and cache consistency

Added **12 cases** for role namespaces, unknown-version recovery, cached role diagnostics,
stream attributes, repeated invalid class access, root-parent lookup propagation, ancestor
search, and page lookup. Invalid class names previously raised once, then returned cached
`None`; validation now precedes caching, using one path for scalar and revision-array names.
Removed unused `role_value` state and a redundant page bounds check: successful lookup
indexes come from the page-node sequence used to construct the corresponding pages.

All **70 structure tests** passed. The full suite with pinned Tesseract enabled passed
**3,960 tests**, with **32 veraPDF integration skips**, in **209.25s**. `structure.py` now
has **100% statement and branch coverage** (408 statements and 176 branches). Workspace
coverage rose **81.99% → 82.04% statements** and **70.98% → 71.05% branches**. Reports
describe that single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,055 statements and 4,397 branches** remain uncovered across the original scope.

## Workspace coverage completion: parent trees and child references

Added **24 cases** for parent-tree slices and validation, nested child order, marked-content
page/stream references, strict versus recovery behavior, depth limits, direct and indirect
annotation references, shared resolved parents, and depth-first searches. An `OBJR` child
previously rejected an indirect annotation dictionary. It now resolves `Obj` before checking
the dictionary, with one shared invalid-object path instead of duplicate checks.

All **58 structure tests** passed. The full suite with pinned Tesseract enabled passed
**3,948 tests**, with **32 veraPDF integration skips**, in **212.45s**. `structure.py`
reached **96.63% statements / 95.05% branches**. Workspace coverage rose
**81.76% → 81.99% statements** and **70.56% → 70.98% branches**. Reports describe that
single completed full-suite run. The structure tests passed again after correcting fixture
types identified by the hooks (a valid bytes attribute key and an explicitly typed parent).

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,073 statements and 4,409 branches** remain uncovered across the original scope.

## Workspace coverage completion: structure metadata and page projections

Added **34 deterministic tests** using the real document resolver for structure content
validation, literal names, decoded metadata caching, page references, normalized attribute
keys, revision precedence and ties, malformed attribute arrays, class names, parent caching,
role maps, and per-page sequence behavior. Repeated parent dictionaries preserve shared
element identity. Page wrappers are compared through their underlying dictionary/document,
since page access may create a fresh wrapper. No production changes were needed.

The full suite with pinned Tesseract enabled passed **3,924 tests**, with **32 veraPDF
integration skips**, in **201.30s**. These tests cover **157 additional statements and 96
branches** across the workspace. `structure.py` reached **74.76% statements / 60.22%
branches**; parent trees and child content references remain. Workspace coverage rose
**81.36% → 81.76% statements** and **69.93% → 70.56% branches**. Reports describe that
single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,165 statements and 4,474 branches** remain uncovered across the original scope.

## Workspace coverage completion: formulas, cleanup, and footer rules

Added **90 deterministic cases** for numeric labels, private/control characters, repeated
spaces and marks, rotated baselines, formulas, scripts, units, chemical prefixes, and footer
geometry. Formula reordering previously duplicated runs and could overwrite intervening
text; it now moves the numerator with `pop`/`insert`, preserving every run exactly once.
An empty following run also incorrectly satisfied the punctuation condition for a time
symbol numerator. Membership now checks a set of punctuation characters, rejecting empties.

All **238 targeted text-rule tests** passed. The full suite with pinned Tesseract enabled
passed **3,890 tests**, with **32 veraPDF integration skips**, in **199.84s**. The entire
`text_rules.py` module now has **100% statement and branch coverage** (563 statements,
260 branches). Workspace coverage rose **81.17% → 81.36% statements** and
**69.56% → 69.93% branches**. Reports describe that single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,322 statements and 4,570 branches** remain uncovered across the original scope.

## Workspace coverage completion: lexical joining and spacing

Added **74 deterministic cases** for alphabetic boundaries, phrase continuation, split-word
visibility/case/geometry gates, whole-word versus fragment ranks, digit joining, frequent-word
spacing, and hidden-text overlap. Controlled rank mappings keep expectations independent of
packaged dictionaries. Removed a special `T`-fragment branch already covered by alphabetic
boundary checks and a short-suffix rank comparison guaranteed by the earlier rank limit.

All **148 targeted word-rank, geometry, and lexical tests** passed. The full suite with
pinned Tesseract enabled passed **3,800 tests**, with **32 veraPDF integration skips**, in
**206.63s**. Lexical spacing has full statement and branch coverage. The whole
`text_rules.py` module reached **86.73% statements / 78.46% branches**; formula handling,
cleanup, and footer rules remain. Workspace coverage rose **81.03% → 81.17% statements**
and **69.32% → 69.56% branches**. Reports describe that single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,397 statements and 4,626 branches** remain uncovered across the original scope.

## Workspace coverage completion: text geometry and width estimation

Added **49 deterministic cases** for reading direction, stream-order ties, blank-run
filtering, overlap tolerances, positive gaps, tracked glyph detection, explicit spaces,
word/column gap thresholds, and suspect character-width estimates. A zero-width glyph line
with no positive space metrics previously raised `StatisticsError` while calculating a
median. It now uses the position-derived median estimate. Removed two redundant length
guards after the existing nonempty alphabetic input filter.

The full suite with pinned Tesseract enabled passed **3,726 tests**, with **32 veraPDF
integration skips**, in **204.62s**. Reading direction through character-width estimation
now has full statement and branch coverage. The whole `text_rules.py` module reached
**77.07% statements / 64.12% branches**; lexical joining and other text rules remain.
Workspace coverage rose **80.93% → 81.03% statements** and **69.13% → 69.32% branches**.
Reports describe that single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,452 statements and 4,664 branches** remain uncovered across the original scope.

## Workspace coverage completion: word-rank lookup and resource loading

Added **25 deterministic tests** for byte-backed and mapped indexes, empty mappings, exact
lookup and missing keys, truncated headers, invalid formats/offsets, unmappable files,
case normalization, source-list filtering and precedence, materialized gzip resources,
compiled-distribution candidates, and recovery from corrupt packaged indexes. No production
changes were needed. The focused scan covers every statement and branch in the word-rank
index and resource-loading section of `text_rules.py`.

The full suite with pinned Tesseract enabled passed **3,677 tests**, with **32 veraPDF
integration skips**, in **210.47s**. These tests cover **70 previously unexecuted statements
and 32 branches**. The whole `text_rules.py` module now has **70.19% statement / 53.41%
branch coverage**; text geometry and spacing rules remain. Workspace coverage rose
**80.76% → 80.93% statements** and **68.92% → 69.13% branches**. Reports describe that
single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,491 statements and 4,693 branches** remain uncovered across the original scope.

## Workspace coverage completion: OCR region boundaries

Added **5 deterministic regression cases** for patterned strokes, off-page grids, and
zero-width/height pages. Region clipping correctly rejected zero-area crops, but the final
page fallback recreated an invalid region. The fallback now uses the same clipping
validation. Removed the unreachable orientation fallback after checking the complete enum
contract with both type checkers; coverage still reports its implicit match fallthrough.

All **719 OCR tests** passed with pinned Tesseract enabled. The full suite passed
**3,652 tests**, with **32 veraPDF integration skips**, in **209.36s**. OCR now has
**100% statement coverage**, with four uncovered branches in learned-Unicode handling and
the orientation match. Region selection has **100% statement and branch coverage**.
Workspace coverage rose **80.75% → 80.76% statements** and **68.89% → 68.92% branches**.
Reports describe that single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,561 statements and 4,725 branches** remain uncovered across the original scope.

## Workspace coverage completion: Newstroke recognition boundaries

Added **7 deterministic tests** for barrier handling, missing templates, ambiguous preceding
glyphs, repeated pen-up markers, isolated glyphs on dense pages, fractional template deltas,
and multiple scales sharing one stroke style. No production changes were needed.

The full suite with pinned Tesseract enabled passed **3,647 tests**, with **32 veraPDF
integration skips**, in **203.33s**. Newstroke now has **100% statement and branch coverage**
(391 statements and 130 branches). Workspace coverage rose **80.73% → 80.75% statements**
and **68.83% → 68.89% branches**. Reports describe that single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
The final test fixture uses the path API's `Matrix` type; all 50 Newstroke tests passed again
after that typing correction. No source-shadowing binaries or new unused-symbol candidates
were found. Coverage sources and exclusions remain unchanged. The **100% workspace goal
remains incomplete**: **7,565 statements and 4,729 branches** remain uncovered. OCR's
remaining gaps are three statements and seven branches in capture, raster orientation,
and region clipping. Core, spec, and compatibility paths remain in scope.

## Workspace coverage completion: OCR entry points and raster scaling

Added **10 deterministic tests** for public companion page extraction, CLI and both module
entry points, import behavior, lazy stroke-profile reuse, recognition dispatch, legacy
reason normalization, segment coverage, exact integer enlargement, and array reduction
without source mutation. No production changes were needed.

The full suite with pinned Tesseract enabled passed **3,640 tests**, with **32 veraPDF
integration skips**, in **204.05s**. Companion API/CLI entry points, extraction contracts,
and the page pipeline now have **100% statement and branch coverage**. Workspace coverage
rose **80.65% → 80.73% statements** and **68.76% → 68.83% branches**. Reports describe
that single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,573 statements and 4,738 branches** remain uncovered. Within OCR, the remaining gaps
are in capture, Newstroke, raster orientation, and region clipping; core, spec, and
compatibility paths remain in scope.

## Workspace coverage completion: OCR atlas and raster boundaries

Added **25 deterministic tests** for atlas clipping, stroke reversal/thickness, closed and
degenerate paths, unsupported display items, decoded sample representations, fractional
upscaling, malformed placement quads, and compositor failure recovery. A NaN placement
coordinate previously passed orientation detection as an identity transform. Orientation
now rejects nonfinite coordinates before calculating bounds from the four valid points.

The full suite with pinned Tesseract enabled passed **3,630 tests**, with **32 veraPDF
integration skips**, in **199.27s**. OCR atlas coverage is **100% statements and branches**;
raster coverage is **98.37% statements / 96.43% branches**. Workspace coverage rose
**80.61% → 80.65% statements** and **68.68% → 68.76% branches**. Reports describe that
single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,601 statements and 4,749 branches** remain uncovered. Remaining OCR processing, core,
spec, and compatibility paths remain in scope.

## Workspace coverage completion: OCR layout and selection enrichment

Added **21 deterministic tests** for rotated OCR order, Unicode direction votes, stable
ties, unchanged native order, word construction, source labels, selection-local font seeds,
invalid glyph/prediction filtering, structural recognition, cached-result reuse, exact page
order, and selective enrichment. Rebuilt captures retain their underlying program and
observations. No production changes were needed.

The full suite with pinned Tesseract enabled passed **3,605 tests**, with **32 veraPDF
integration skips**, in **201.04s**. OCR block layout and selection enrichment now each
have **100% statement and branch coverage**. Workspace coverage rose **80.52% → 80.61%
statements** and **68.53% → 68.68% branches**. Reports describe that single full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,619 statements and 4,760 branches** remain uncovered. Remaining OCR processing, core,
spec, and compatibility paths remain in scope.

## Workspace coverage completion: OCR routing and resampling

Added **51 deterministic tests** for resampling shape contracts, identity, noncontiguous
inputs, nearest-neighbor replication, box-filter stroke preservation, bilinear pixel centers,
single-pixel channels, mixed-axis resizing, OCR scale thresholds, noisy-native routing,
ActualText overrides, embedded images, bounded overlap calculation, rectangular artwork,
and whole-batch replacement by stronger OCR. Smooth resizing now validates shape before
reading both spatial axes, consistently raising ValueError for invalid 1D input instead
of leaking IndexError. Valid-image resampling behavior is unchanged.

The full suite with pinned Tesseract enabled passed **3,584 tests**, with **32 veraPDF
integration skips**, in **201.69s**. OCR routing/fusion and resampling now each have **100%
statement and branch coverage**. Workspace coverage rose **80.44% → 80.52% statements**
and **68.38% → 68.53% branches**. Reports describe that single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,652 statements and 4,784 branches** remain uncovered. Remaining OCR processing, core,
spec, and compatibility paths remain in scope.

## Workspace coverage completion: OCR chart tables and native routing

Added **29 deterministic tests** for chart numeric splitting, geometry, spatial duplicate
suppression, invalid observations, row ordering, shared table finalization, trusted-vector
bypass, native text around artwork, and rotated-label routing. Chart column indexes now
come from the accepted-cell count, removing a redundant counter; the empty-box guard was
removed because each accepted cell contributes a box.

The full suite with pinned Tesseract enabled passed **3,533 tests**, with **32 veraPDF
integration skips**, in **202.15s**. OCR chart-table detection now has **100% statement and
branch coverage**. OCR observation planning/fusion reached **88.95% statements / 82.05%
branches**. Workspace coverage rose **80.24% → 80.44% statements** and **68.11% → 68.38%
branches**. Reports describe that single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,686 statements and 4,807 branches** remain uncovered. Remaining OCR processing, core,
spec, and compatibility paths remain in scope.

## Workspace coverage completion: Tesseract binding

Added **37 deterministic tests** for text confidence/noise gates, optional hOCR failures,
character-filter recall, symbol iteration, word-boundary propagation, filtered recognition,
optional cleanup, language-data discovery failures, main-thread initialization, partial
stderr setup, and invalid timeout crop recovery. Word and line filtering previously accepted
NaN and infinite confidence; it now rejects them consistently with symbol filtering.
Redundant positive-size checks were removed after rectangle dimensions are clamped to one.

The full suite with pinned Tesseract enabled passed **3,504 tests**, with **32 veraPDF
integration skips**, in **203.45s**. Tesseract binding coverage rose **86.29% → 100% statements**
and **73.38% → 100% branches**. Workspace coverage rose **80.10% → 80.24% statements**
and **67.84% → 68.11% branches**. Reports describe that single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,765 statements and 4,847 branches** remain uncovered across the authored workspace.
Remaining OCR processing, core, spec, and compatibility paths remain in scope.

## Workspace coverage completion: OCR session integration

Added **25 deterministic tests** for declared projection caching, undeclared projection
errors, page-box selection, real text-band preflight scaling, direct/rendered page tasks,
weak-region retry provenance, bounded timeout recovery, packed-vector options, image-region
filtering/cropped fallback, and distributed-outline routing. No production changes were needed.

The full suite with pinned Tesseract enabled passed **3,467 tests**, with **32 veraPDF
integration skips**, in **204.29s**. OCR session coverage rose from **70.00% statements /
55.56% branches to 100% of both**. Workspace coverage rose **79.99% → 80.10% statements**
and **67.74% → 67.84% branches**. Reports describe that single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,819 statements and 4,888 branches** remain uncovered. OCR engine details and all
remaining core, spec, and compatibility paths remain in scope.

## Workspace coverage completion: packed-vector and table-grid orchestration

Added **12 deterministic integration tests** covering packed-coordinate remapping, isolated
label supplementation, full-vector fallback availability, real raster grid detection and
cell-task construction, replacement quality gates, retained outside text, unavailable passes,
and page augmentation policy. Recognition results are controlled at the OCR boundary.
The selected-task guard around table retry was removed: selection occurs only after a
nonempty task batch completes, and resetting selection clears both candidate and tasks.

The full suite with pinned Tesseract enabled passed **3,442 tests**, with **32 veraPDF
integration skips**, in **205.04s**. OCR pipeline coverage rose from **75.28% statements /
75.68% branches to 100% of both**. Workspace coverage rose **79.87% → 79.99% statements**
and **67.62% → 67.74% branches**. Reports describe that single completed full-suite run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal remains incomplete**:
**7,864 statements and 4,904 branches** remain uncovered. OCR session/engine integration
and the remaining core, spec, and compatibility paths remain in scope.

## Workspace coverage completion: OCR rescue policy and orchestration

Added **29 deterministic tests** for spatial utility distribution, clipping, raster
identity deduplication, rescue thresholds, hidden-text verification, and full-page versus
weak-region adaptive retries. Retry tests check unavailable rasters and results that
replace, augment, or fail to improve the original observations. The rescue decision now
returns directly instead of carrying a temporary boolean; decision thresholds are unchanged.

The final full suite with pinned Tesseract enabled passed **3,430 tests**, with **32 veraPDF
integration skips**, in **209.33s**. An earlier run was interrupted after a lint-driven
source adjustment; only the completed run against the final source is reported here.

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| OCR rescue policy | 31.71% → 100.00% | 0.00% → 100.00% |
| OCR pipeline | 58.43% → 75.28% | 56.76% → 75.68% |
| Entire workspace | 79.64% → 79.87% | 67.42% → 67.62% |

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged. The **100% workspace goal is incomplete**: **7,909
statements and 4,923 branches** remain uncovered. Packed-vector and table-grid pipeline
integration are next; all remaining workspace packages remain in scope.

## Workspace coverage completion: OCR candidate reconciliation

Added **46 deterministic tests**: 37 for candidate reconciliation and nine for remaining
region-task fallbacks. Both modules now have **100% statement and branch coverage**.
Tests cover Unicode/punctuation normalization, contiguous token containment, hidden-text
minimum-count/token/spatial thresholds, nearest unused repeated-token matching, overlapping
tile reconciliation, utility and mode selection, symbol preservation, stale replacement
entries, augmentation confidence/coverage boundaries, direct-scan opt-out, dominant-image
fallback, and unavailable raster handling. No production changes were needed.

The full suite with pinned Tesseract enabled passed **3,401 tests**, with **32 veraPDF
integration skips**, in **214.29s**. Workspace statement coverage rose **79.38% → 79.64%**
and branch coverage **67.09% → 67.42%**. These numbers describe that single full-suite run.
Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing binaries or new unused-symbol candidates were found. Coverage sources
and exclusions remain unchanged.

The **100% workspace goal remains incomplete**: **7,999 statements and 4,954 branches**
remain uncovered. OCR pipeline orchestration, engine handling, and all remaining core/spec
and compatibility paths remain in scope.

## Workspace coverage completion: OCR region tasks

Added **18 deterministic tests** for same-image/mode batching, count and pixel budgets,
text-band estimation, preprocessing options, utility-grid coordinates, dense-page rescue
limits, weak-cell selection, source-raster deduplication, rescue crop mapping, and layered
image compositing. A regression test exposed transparent RGBA text influencing estimated
text height; estimation now uses the shared white-background intensity conversion.

The full suite with pinned Tesseract enabled passed **3,355 tests**, with **32 veraPDF
integration skips**, in **206.80s**.

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| OCR region tasks | 45.65% → 96.74% | 32.26% → 90.32% |
| Entire workspace | 79.14% → 79.38% | 66.85% → 67.09% |

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No compiled modules shadowed sources; the unused-symbol scan found no new candidates.
Coverage sources and exclusions remain unchanged. The **100% workspace goal is incomplete**:
**8,102 statements and 5,003 branches** remain uncovered, including six statements and six
branches in region tasks. Candidate scoring and the remaining region fallback paths remain
in scope.

## Workspace coverage completion: raster preprocessing

Added **33 deterministic tests** for alpha compositing, ink grids, adaptive thresholds,
image compaction, text signals, and safe crop selection. Transparent black pixels previously
counted as ink; all preprocessing consumers now share white-background intensity conversion.

The full suite with pinned Tesseract enabled passed **3,337 tests**, with **32 veraPDF
integration skips**, in **208.01s**. Workspace statement coverage rose **78.88% → 79.14%**
and branch coverage **66.70% → 66.85%**. Ruff, mypy, ty, all 19 import contracts,
repository hooks, and diff checks passed. No source-shadowing binaries were found.
Coverage sources and exclusions remain unchanged.

The **100% workspace goal remains incomplete**: **8,199 statements and 5,040 branches**
remain uncovered. Region-task orchestration and candidate selection are next.

## Workspace coverage completion: raster OCR grids

This checkpoint adds **31 deterministic tests** for raster ruling detection and cell OCR.
The full suite with pinned Tesseract enabled passed **3,304 tests**, with **32 veraPDF
integration skips**, in **205.48s**. Reports describe that single full-suite run.

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| OCR raster grids | 46.51% → 100.00% | 20.27% → 100.00% |
| Entire workspace | 78.59% → 78.88% | 66.32% → 66.70% |

Tests cover bounded row gaps, independent row runs, ruling clustering, grayscale/RGB/RGBA
grid detection, skew estimation/shearing, cell insets and ink gates, task coordinate mapping,
cell-count limits, table regularity, column-straddling thresholds, and row observation merging.

Gap-closing regression tests reproduced two faults: gaps larger than the requested limit
were filled, and existing ink near row edges was erased. The helper now finds neighboring
ink positions and fills only bounded gaps within the specified limit, preserving source ink
and row margins. Blank, full, and zero-width rows are positive controls. The RasterImage
contract guarantees a three-dimensional array, so the unreachable two-dimensional input
branch was removed. The minimum-column gate also makes an empty interior-column list
impossible; its redundant guard was removed.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No compiled modules shadowed source files, and the unused-symbol scan found no new candidates.
Coverage sources and exclusions remain unchanged.

The **100% workspace goal remains incomplete**: **8,301 statements and 5,064 branches**
remain uncovered. OCR raster preprocessing, region tasks, candidate scoring, document
structure, layout, rendering, and compatibility behavior remain in scope.

## Workspace coverage completion: learned Unicode alignment

This checkpoint adds **29 tests** for learned Unicode overlays and capture enrichment.
The full suite with pinned Tesseract enabled passed **3,273 tests**, with **32 veraPDF
integration skips**, in **209.36s**. Reports describe that single full-suite run.

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| OCR capture analysis | 67.20% → 99.60% | 51.00% → 96.00% |
| Entire workspace | 78.38% → 78.59% | 66.01% → 66.32% |

Tests preserve spacing, geometry, cluster identity, and original runs when applying learned
text. They reject invalid replacement strings, ActualText overrides, mixed source clusters,
misaligned text, and reversed text-show geometry. Ligature observations receive one semantic
replacement, and glyph evidence reclassifies only applied substitutions. Program-level tests
verify that learned text reaches observations and evidence before routing, while image filter
metadata, fields, annotations, and the original capture program are retained. Controlled
template-decoder results test trusted versus untrusted promotion separately from the existing
real template-decoding suite. Capture entry-point options are checked at the native boundary.

No production changes were needed. Ruff lint/format, mypy, ty, all 19 import contracts,
repository hooks, and diff checks passed. No source-shadowing compiled modules or new
unused-symbol candidates were found. Coverage sources and exclusions remain unchanged.

The **100% workspace goal remains incomplete**: **8,416 statements and 5,123 branches**
remain uncovered. OCR capture retains one uncovered statement and four branches. OCR raster,
grid extraction, region tasks, document structure, layout, and rendering remain in scope.

## Workspace coverage completion: OCR capture evidence

This checkpoint adds **37 deterministic tests** for capture-time OCR evidence. The full
suite with pinned Tesseract enabled passed **3,244 tests**, with **32 veraPDF integration
skips**, in **213.29s**. Reports describe that single full-suite run.

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| OCR capture analysis | 25.69% → 67.20% | 4.90% → 51.00% |
| Entire workspace | 78.11% → 78.38% | 65.70% → 66.01% |

Tests cover eligible stroke styles, compact-path ratios, dominant/secondary styles,
distribution and rotation gates, vector workload counts, bounded overlap subtraction,
background/control rejection, high-resolution routing, hidden numeric scan checks, and
promotion of normalized observation references with preserved geometry. Template-text
promotion preserves captured drawings and updates native-character and vector evidence.

Dominant-style selection now takes the maximum of the statistics values directly, avoiding
an unused key binding and deletion. A redundant selected-count guard was removed: a trusted
dominant style already contains at least 300 compact paths, satisfies the selected-style
criteria, and all compact paths fit the render-size limit. No thresholds were changed.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No compiled modules shadowed sources, and the unused-symbol scan found no new candidates.
Coverage source roots and exclusions remain unchanged.

The **100% workspace goal remains incomplete**: **8,498 statements and 5,170 branches**
remain uncovered. Capture analysis still has 82 uncovered statements and 49 branches,
primarily learned Unicode alignment, replacement-aware glyph evidence, and enrichment paths.

## Workspace coverage completion: OCR region selection

This checkpoint adds **26 deterministic tests** for OCR region selection. The full suite
with pinned Tesseract enabled passed **3,207 tests**, with **32 veraPDF integration skips**,
in **238.98s**. HTML and JSON reports describe that single full-suite run.

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| OCR region selection | 18.10% → 100.00% | 9.62% → 98.04% |
| Entire workspace | 77.62% → 78.11% | 65.08% → 65.70% |

Tests decode real grayscale image bytes and preserve their page mapping. They reject clipped,
unoriented, undersized, missing, or ambiguously overlapping direct images. Geometry checks
cover score-ordered merging and reason deduplication, region count/area limits, page fallback,
image padding, native-text suppression, header/body density, distributed outlined text,
compact/large grids, fine label neighborhoods, and off-page components.

A redundant grid-component emptiness guard was removed. The producer only returns components
with horizontal and vertical members, and this consumer had already evaluated their bounds
before the guard. Crop rejection checks remain; two such branches are still uncovered.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No compiled modules shadowed sources, and the unused-symbol scan found no new candidates.
Coverage source roots and exclusions remain unchanged.

The **100% workspace goal remains incomplete**: **8,604 statements and 5,218 branches**
remain uncovered. OCR capture analysis is now the largest remaining OCR module gap; document
structure, layout, rendering, and compatibility behavior also remain in scope.

## Workspace coverage completion: OCR vector raster and remapping

This checkpoint adds **51 deterministic tests** for the vector OCR adapter. The full suite
with pinned Tesseract enabled passed **3,181 tests**, with **32 veraPDF integration skips**,
in **198.62s**. HTML and JSON reports describe that single full-suite run.

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| OCR vector adapter | 18.37% → 100.00% | 1.35% → 100.00% |
| Entire workspace | 76.74% → 77.62% | 64.26% → 65.08% |

Tests cover shelf packing, dense and isolated variants, both raster backends, crop bounds,
ambiguous/outside cell rejection, reference identity, sequence remapping, pin-label filters,
symbol assembly, supplemental learning, substitution thresholds, and decoded-text recovery.
Real raster output is checked for ink and pixel budgets. Recovery checks retain unchanged
OCR objects where possible, replace eligible misreads once, and add previously absent runs.

A raster-budget regression produced 20,049 pixels against a 20,000-pixel limit. The fast
renderer silently exceeded the budget while the general renderer raised an allocation error.
Both vector paths now use the integer-dimension adjustment already used by page OCR,
extracted into one shared helper. Pixel fitting checks cover single-pixel axes and reject
nonpositive budgets before allocation. The area estimate still chooses the initial scale;
the shared helper checks the renderer's actual rounded dimensions.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No compiled modules shadowed sources, and the unused-symbol scan found no new candidates.
Coverage source roots, exclusions, and recognition acceptance thresholds remain unchanged.

The **100% workspace goal remains incomplete**: **8,799 statements and 5,313 branches**
remain uncovered. Remaining work includes OCR regions/capture, document structure and
recovery, layout, rendering, and compatibility facade behavior.

## Workspace coverage completion: OCR stroke learning

This checkpoint adds **35 deterministic tests** for page-local stroke learning. The full
suite with pinned Tesseract enabled passed **3,130 tests**, with **32 veraPDF integration
skips**, in **222.32s**. HTML and JSON reports describe that single full-suite run.

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| OCR stroke learner | 26.92% → 100.00% | 0.00% → 100.00% |
| Entire workspace | 76.02% → 76.74% | 63.39% → 64.26% |

Tests construct captured paths and OCR seed records directly. They cover independent votes,
duplicate source sequences, the 75% consensus threshold, tied labels, anchored-word learning,
conflicting anchored words, supplemental evidence, document alphabets, and geometry-based
alignment. Shape checks cover approximate variants, ambiguous labels, topology mismatches,
empty geometry with explicit bounds, overlapping components, capture-order gaps, isolated
wire rejection, token limits, and seed-run dimensions. Caller alphabets remain unchanged.

A long-chain regression reproduced the eight-round learning cap truncating a supported
alphabet. Learning now continues until no signatures are added. Each productive round adds
previously unknown signatures from the finite sample alphabet, so it terminates without an
arbitrary iteration count. The regression shares source sequence identities across chained
samples, preventing repeated observations from manufacturing independent consensus votes.
Anchored unanimity now checks for exactly one proposed label directly, removing redundant
sorting and vote-count comparison. Existing consensus and conflict policies are preserved.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing compiled modules or new top-level unused-symbol candidates were found.
Coverage source roots and exclusions remain unchanged.

The **100% workspace goal remains incomplete**: **9,140 statements and 5,438 branches**
remain uncovered. The next OCR target is vector raster packing and observation remapping,
which connect the now-tested stroke learner to engine results and page coordinates.

## Workspace coverage completion: OCR Newstroke recognition

This checkpoint adds **43 deterministic tests** for Newstroke vector recognition. The full
suite with pinned Tesseract enabled passed **3,095 tests**, with **32 veraPDF integration
skips**, in **601.85s**. HTML and JSON reports come from that full-suite run.

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| OCR Newstroke module | 21.72% → 97.95% | 0.00% → 93.08% |
| Entire workspace | 75.24% → 76.02% | 62.57% → 63.39% |

Tests render the bundled coordinate strings into independent captured line objects, then
exercise the actual decoder without an OCR engine. Dense pages exceed the real acceptance
thresholds in all four right-angle orientations. Assertions cover text, sequence order,
font size, spacing, color, padded bounds, confidence, and provenance. Separate checks reject
unrelated geometry, invalid paths, invisible strokes, implausible scales/aspects/shear,
corrupted glyphs, mixed styles, and ambiguous template labels. Backward decoding preserves
prefix glyphs, longest-template selection, and one/two-space gaps. Trust checks exercise
each page-level evidence threshold independently.

Two redundant operations were removed after checking their upstream invariants. The
positive scale and orthogonality checks already reject singular transforms. Every match
in a decoded sequence has the same stroke style, including width, so bounds padding uses
that width directly instead of maintaining a maximum during the bounds scan. Tests verify
padding on alternating styles and preserve barriers between unrelated drawings.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No compiled modules shadowed source files, and the top-level unused-symbol scan found no
new candidates. Coverage source roots, exclusions, and recognition thresholds are unchanged.

The **100% workspace goal remains incomplete**: **9,426 statements and 5,571 branches**
remain uncovered. Newstroke still has eight uncovered statements and nine branches. The
next large OCR target is the separate stroke learner: independent seed votes, conflicting
labels, anchored-word learning, supplemental seeds, and approximate shape matching.

## Workspace coverage completion: CFF dictionary recovery and accents

This checkpoint adds **24 tests**, bringing the focused CFF suite to **98 tests**. The
full suite with pinned Tesseract enabled passed **3,052 tests**, with **32 veraPDF
integration skips**, in **726.08s**. Reports come from this single full-suite run.

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| Core CFF font-program module | 87.44% → 98.50% | 80.49% → 95.38% |
| Entire workspace | 75.05% → 75.24% | 62.28% → 62.57% |

Tests reproduced two recovery bugs. Infinite FDArray offsets escaped as OverflowError
before reaching the existing finite-offset validation; the reader now routes that error
through recovery. Negative glyph IDs indexed from the end of FDSelect (or raised for more
negative values); recovery now uses the default font dictionary consistently with matrix
recovery and positive out-of-range IDs.

Additional checks cover malformed dictionary entries without shifting FD indices,
predefined charset overflow, custom and expert encodings, composed accent placement and
bounds, child matrix recovery/composition, deterministic random charstrings, invalid headers,
and complete minimal CFF construction. Byte-reader IndexError handlers were removed where
every byte access already has a length check. Feature-grid clamps were removed because the
coordinates are normalized against their own bounds and remain inside the grid, including
subunit dimensions. These are source simplifications, not coverage exclusions.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing compiled modules or new top-level unused-symbol candidates were found.
The four coverage source roots and all exclusions remain unchanged.

The **100% workspace goal remains incomplete**: **9,732 statements and 5,697 branches**
remain uncovered. CFF retains nine uncovered statements and 11 branches. Larger gaps remain
in OCR stroke processing, pdfplumber, document structure, layout, and rendering. Newstroke
has no dedicated OCR tests yet; deterministic vector paths can exercise its template fitting
and page-level acceptance thresholds in the next pass.

## Workspace coverage completion: CFF recovery and glyph matching

This checkpoint adds **74 deterministic tests** for truncated CFF charsets and encodings,
FDSelect ranges, private subroutine offsets, glyph geometry, and Unicode repair. The full
suite with pinned Tesseract enabled passed **3,028 tests**, with **32 veraPDF integration
skips**, in **707.58s**. The HTML and JSON reports describe this single full-suite run.

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| Core CFF font-program module | 35.28% → 87.44% | 14.23% → 80.49% |
| Entire workspace | 73.83% → 75.05% | 60.61% → 62.28% |

Malformed-table checks preserve complete entries, bound glyph IDs and ranges, resolve
encoding supplements through SIDs, and reject predefined charsets for CID fonts. Unicode
checks protect legitimate letters and ligatures, enforce strict replacement thresholds,
and compare scalar and matrix matching while retaining aliased content-stream codes.
Geometry checks cover transformed cubic extrema, rasterized outlines, missing glyphs,
terminated and unterminated paths, and recovery that retains only completed contours.

The charstring geometry builder now shares its bounds/result construction between normal
and recovered execution. Only unfinished valid paths flush their current contour; malformed
partial contours remain discarded. This removes duplicated result construction without
changing recovery policy. Strict spec parsing remains separate from reader recovery.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
No source-shadowing compiled extensions were present, and the top-level unused-symbol scan
found no new candidates. Coverage sources and exclusions are unchanged.

The workspace-wide **100% goal remains incomplete**: **9,810 statements and 5,744 branches**
are still uncovered. The CFF module itself has 77 statements and 48 branches remaining,
including font dictionary recovery, composed accent glyphs, and built-in encodings. Fonts,
document structure/recovery, rendering, and deeper OCR paths remain in scope.

## Workspace coverage completion: annotations and tables

Source/test commit: `9ff4ed2b`. This checkpoint adds **41 tests** for image annotations,
explicit table grids, table settings, and invalid configurations. The full suite with
pinned Tesseract enabled passed **2,954 tests**, with **32 veraPDF integration skips**,
in **297.64s**. Generated HTML and JSON reports describe this run.

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| pdfplumber facade module | 61.78% → 74.12% | 44.86% → 61.49% |
| Entire workspace | 73.39% → 73.83% | 60.07% → 60.61% |

`TableSettings.resolve` and `PageImage._render_drawings`, including its local drawing
helpers, now have **100% statement and branch coverage**. Serialization and the validation
package retain full coverage. This does not mean the entire image or table API is complete.

Pixel-level differential checks reproduced and fixed missing rectangle top edges and
unwanted diagonals, rejected polyline point sequences, ignored circle centers/radii/colors,
and RGB byte values incorrectly treated as normalized colors. Additional tests cover
batch annotations, cropped coordinates, RGB/RGBA canvases, file output, debug helpers,
and invalid channel layouts. The unreachable drawing-kind guard was removed after checking
that the private drawing list receives only the four kinds emitted by its public helpers.

Table tests reproduced duplicate grid rows, prose incorrectly recognized as tables under
the default line strategy, and boundary characters copied into adjacent cells. Grid rows
are emitted once and cell text is selected by character centers. Word-based fallback runs
only when both requested strategies are text-based. Broader table strategies remain a gap.

Settings now supply the reference defaults, axis fallbacks, and nonnegative measurements.
Explicit-line validation checks both axes when table finding starts. For omitted line
sequences, the reference raises TypeError while the facade deliberately raises ValueError;
tests record this diagnostic difference while verifying both reject the input. Resolved
settings are passed directly into table finding, preserving text options without converting
them back to a dictionary. Resolution's unreachable merge fallback was also removed.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks passed.
The top-level unused-code scan found no additional candidates. Coverage sources and
exclusions remain unchanged.

The workspace-wide **100% goal remains active**: **10,291 statements and 5,999 branches**
are still uncovered. The combined headline is **70.14%**, with 29,035/39,326 statements and
9,229/15,228 branches covered. Remaining facade work includes native table projection,
text-based table strategies, layout/search/metadata paths, and fuller annotation styling.
Fonts, document structure/recovery, rendering, and deeper OCR paths also remain in scope.

## Workspace coverage completion: first checkpoint

Source commit: `e316dd60`; test commit: `5d1de1c2`. The workspace-wide goal remains
**100% statement and branch coverage** under the existing four-package configuration.
No coverage exclusions or source roots were changed. This checkpoint adds **133 tests**:
106 pdfplumber differential cases, four serialization cases, and 23 validation cases.

The final full scan, including the pinned Tesseract tests, passed:
**2,913 passed, 32 skipped in 201.17s**. All skips remain the opt-in veraPDF execution
checks. Unlike the earlier interim merge, the generated HTML/JSON reports now come
from one full-suite run containing all the new tests.

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| Serialization | 98.25% → 100.00% | 93.42% → 100.00% |
| Validation package | 93.09% → 100.00% | 80.49% → 100.00% |
| pdfplumber facade module | 45.23% → 61.78% | 26.67% → 44.86% |
| Entire workspace | 72.65% → 73.39% | 59.29% → 60.07% |

Serialization now covers malformed element dispatch, absent table-associated text,
Markdown headings/lists/paragraphs, and list prefixes inside styled spans. Validator
coverage includes malformed counts and rule records, contradictory summaries, incomplete
jobs, engine permissions, non-POSIX timeout cleanup, duplicate declarations, and backend
profile mismatches. This proves adapter execution coverage; installed veraPDF integration
remains separate from recorded reports and controlled process failures.

The pdfplumber comparisons exposed and fixed behavior beyond extraction snapshots:

- Moves and resizes preserve optional PDF coordinates. Snapping moves whole objects in
  cluster order; edge filters honor type, orientation, and minimum length.
- Cropping clips object bounds and preserves parent selection in nested crops. Containment
  reuses the shared geometry helper; pdfplumber-specific edge-touch semantics remain local.
- Deduplication honors default font/size and upright attributes and transitive position
  clusters, replacing the stateful per-character filter with grouped selection.
- JSON attribute filtering copies records before filtering, preserving cached page objects.
- Image dimensions honor resolution, CropBox/MediaBox selection, and antialiasing. Copies
  preserve an already-cropped raster, and annotation coordinates account for crop origins.

Ruff lint/format, mypy, ty, all 19 import contracts, all repository hooks, and diff checks
passed. No source-shadowing compiled extensions were present. The top-level unused-code
scan found no new candidates after reusing the shared containment helper.

**The overall goal is not complete:** 10,458 statements and 6,076 branches remain uncovered.
The combined headline is **69.68%** (28,848/39,306 statements and 9,142/15,218 branches).
Remaining work includes table finding and image annotations in the facade, fonts,
document structure/recovery, rendering, and deeper OCR execution paths. Continue adding
behavioral tests and reviewing proven unreachable code; uncovered supported paths remain
part of the target.

## OCR scheduling and raster execution pass

Source commit: `f1b12732`. This pass adds **59 deterministic checks** and **four
opt-in real-engine checks**. The final full run enabled the pinned Tesseract suite:
**2,780 passed, 32 skipped in 215.83s**. All skips were the existing veraPDF integration
checks. The ordinary OCR suite passes **115 tests** and skips the four real-engine checks.

The deterministic tests exercise character/addition fallback thresholds, strict utility
gains, native seeding, duplicate rejection, selected-task provenance, and pass orchestration.
Both `internal_OcrPassState.prepare` and `.complete` now have **100% statement and branch
coverage**. Raster checks cover pixel budgets, UserUnit/crop rounding, all eight direct-image
orientations, tile overlap, page-coordinate remapping, and safe crop selection.

The tests reproduced and fixed these issues:

- Cancellation could continue through a recognition batch. The session now passes its
  cancellation check into the engine batch; checks run before engine creation and between
  tasks. Tests verify both shared-image and separate-image batches and engine cleanup.
- A one-pixel-wide or one-pixel-high image could exceed a raster or timeout-retry budget
  after the other dimension was rounded. Both reductions now bound individual dimensions
  as well as area.
- Rounding an enlargement to a whole-number scale could exceed the remaining pixel budget.
  Whole-number replication now requires enough headroom; otherwise fractional resampling
  stays within the cap.

The `schematic-regions-fallback` pass was unreachable: its additions threshold was zero,
while additions start at zero and can only be nonnegative. The scheduling guard always
skipped it. Its configuration is removed. Successful region completion now creates the
next immutable state once instead of creating an intermediate replacement first. Other
selection state remains necessary for the tested fallback and provenance behavior.

The real-engine tests use **Tesseract 5.5.1 linked by tesserocr** and a checksum-pinned English
model. Authored image-only PDFs exercise upright and rotated image placement through the
public OCR API; a raster crop checks actual recognition and coordinates. A delegating
wrapper verifies real API release on success and cancellation. All three fixture files
reproduce byte-for-byte with the checked-in generator. See the
[fixture and engine instructions](../packages/core-pdf-ocr/tests/fixtures/README.md).

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| OCR package | 38.34% → 50.57% | 19.14% → 32.66% |
| OCR pass pipeline | 16.76% → 58.43% | 2.70% → 56.76% |
| OCR session | 30.00% → 67.33% | 0.00% → 47.22% |
| OCR raster preparation | 17.38% → 60.33% | 0.00% → 60.47% |
| Tesseract adapter | 78.21% → 86.29% | 66.67% → 73.38% |
| Entire workspace | 71.35% → 72.65% | 58.06% → 59.29% |

The final scan includes the opt-in real-engine tests; the preceding baseline did not.
The combined statement/branch headline is **68.92%**: 28,553 of 39,300 statements and
9,019 of 15,212 branches execute. Reproduce this scan by prefixing the full command in
[coverage.md](coverage.md) with `CORE_PDF_TESSERACT_TESTS=1` in the pinned environment.
Generated HTML and JSON reports describe this final run.

Ruff lint/format, mypy, ty, all 19 import contracts, repository hooks, and diff checks
passed. The conservative top-level unused-code scan found no additional candidates.
Adaptive rescue, grid-cell fallback, packed vector recognition, and more varied real
scans still need broader tests; the remaining coverage gaps do not establish dead code.

## OCR and serialization pass

Source commit: `22706f2f`. This pass added deterministic tests for:

- JSON identity interning (shared versus equal objects, page-local IDs), all record kinds,
  invalid metadata locations, duplicate page IDs, and empty documents.
- HTML escaping and styled/list text, merged table cells and row bands, plus CSV/TEI
  escaping, geometry, ordered/deduplicated page selections, and invalid selections.
- Native/OCR routing and priority, fusion confidence thresholds and duplicates, image
  supplements, sparse native replacement, and coordinate preservation.
- Cross-page font votes, confidence and geometric rejection, ambiguous mappings,
  selection-local immutable overlays, seed recognition reuse, and stroked-alphabet fallback.
- Engine failure/timeout classification, API cleanup and same-image reuse, invalid or
  missing language data, probe failures, hOCR filtering, and bounded timeout retries.

A regression test demonstrated that cancelled cross-page stroked-text enrichment could
continue through cached recognition results. The loop now checks cancellation before each
selected page, including cached and structural-decode paths. Font enrichment's stored seed
indexes were redundant with its recognition-map keys and were removed, as was an unused
engine-loop counter. Markdown now calls the shared HTML table renderer directly, retaining
merged cells without a forwarding wrapper.

These tests use synthetic captures and a stub engine. They verify policy, integration
boundaries, and resource ownership, not real-engine recognition accuracy. Full raster/pass
orchestration and more complex vector-recognition paths remain candidates for further tests.

The full configured suite passed: **2,717 passed, 32 skipped in 201.77s**, adding 86
passing tests. All skips remain the opt-in installed veraPDF integration checks.

| Area | Statement coverage before → after | Branch coverage before → after |
| --- | ---: | ---: |
| OCR package | 17.92% → 38.34% | 1.60% → 19.14% |
| OCR routing/fusion | 19.34% → 77.90% | 3.85% → 66.67% |
| Cross-page enrichment | 22.28% → 89.58% | 0.00% → 88.64% |
| Tesseract adapter | 16.07% → 78.21% | 1.33% → 66.67% |
| Serialization | 32.03% → 98.25% | 17.11% → 93.42% |
| Entire workspace | 68.57% → 71.35% | 55.87% → 58.06% |

The combined statement/branch headline is **67.64%**. There are 28,037 covered statements
out of 39,297 and 8,830 covered branches out of 15,208. Compared with the preceding pass,
the denominator fell by only five statements while covered statements increased by 1,088
and covered branches by 333. Most of this improvement therefore comes from additional
execution, rather than deleting uncovered code. Six nonempty files remain wholly unexecuted.

Ruff lint/format, mypy, ty, all 19 import contracts, and diff checks passed. Generated HTML
and JSON coverage reports now describe this pass. The conservative static scan found no
new top-level removal candidates.

## Maintenance pass

Source commit: `42fbaae7`. The maintenance pass removed 117 net production source lines:

- The shared core/OCR CLI is Markdown-only, with `--mode`, `--plain`, and the Python
  helper's format argument removed. Parse-only, stdout, and file-output behavior remains.
- OCR imports shared contracts and the capture sentinel directly from their core owners.
  Recognition-specific contracts remain in the companion.
- OCR table reconciliation runs after layout, so layout still sees every original table
  obstacle. Core assembles the resulting page directly; the forwarding module and assembly
  dispatch aliases are gone.
- Selected-page assembly retains ordering and cancellation checks in one function.
- PNG prediction attempts the codec directly, retaining fallback and damaged-row behavior.

The audit deliberately retained strict/recovery resolver and predictor adapters: similarly
shaped code selects different coercion, decoding, or recovery behavior. Duplicate core/spec
geometry and buffer utilities remain where sharing would violate dependency contracts.
Validation retains its backend boundary, original-byte snapshots, and result enrichment.
No supported feature or public spec parsing extension was removed.

The new tests exercise CLI output/error behavior, removed-option rejection, document
selection/metadata/diagnostic assembly, cancellation, package isolation, predictor fallbacks,
and native/recognized OCR assembly with deterministic observations. They do not establish
coverage of real Tesseract execution or all recognition routes.

The expanded suite passed: **2,631 passed, 32 skipped in 198.62s**, adding 23 passing tests
to the preceding run. All skips are opt-in installed veraPDF checks.

| Package | Statements covered | Statement coverage | Branch coverage |
| --- | ---: | ---: | ---: |
| core-pdf | 18,368 / 25,892 | 70.94% | 56.46% |
| core-pdf-spec | 7,604 / 8,990 | 84.58% | 73.27% |
| core-pdf-ocr | 748 / 4,174 | 17.92% | 1.60% |
| core-pdf-validate | 229 / 246 | 93.09% | 80.49% |
| Total | 26,949 / 39,302 | 68.57% | 55.87% |

The combined statement/branch headline is **65.03%**. Relative to the prior rescan,
the measured statement count fell by 17 while covered statements increased by 951.
There are 12,353 missing statements and 16 nonempty files with no executed statements.
This increase combines newly exercised behavior with module imports; OCR's much lower
branch coverage makes clear that most recognition paths remain untested.
The core CLI reached 97.30% statement coverage, output serialization 32.03%, and the OCR
page pipeline 75.47%. The conservative static rescan found no new top-level definition
whose identifier occurs only at its definition in authored Python/TOML.

Ruff lint/format, mypy, ty, all 19 import contracts, and diff checks passed. Generated
`htmlcov/index.html` and `htmlcov/coverage.json` now represent this maintenance pass.

## Original audit history

Baseline: `fdd52fee`, with coverage configuration and test dependencies added in the
working tree. Run completed September 13, 2026 (America/Bogota), using Python 3.13.14,
coverage.py 7.16.1, pytest-cov 7.1.0, and eight xdist workers.

The [configured coverage run](coverage.md) passed: **2,608 passed, 32 skipped in 196.12s**.
The skipped tests are the opt-in installed veraPDF integration cases. This was the default
facade-owned differential matrix, not the exhaustive cross-corpus matrix.

## Post-removal rescan

Source commit: `302c55f1`. The same configured suite passed after all removals:
**2,608 passed, 32 skipped in 207.09s**. All skips require an installed veraPDF engine.

| Package | Statements covered | Statement coverage | Branch coverage |
| --- | ---: | ---: | ---: |
| core-pdf | 18,165 / 25,902 | 70.13% | 55.80% |
| core-pdf-spec | 7,604 / 8,990 | 84.58% | 73.27% |
| core-pdf-ocr | 0 / 4,181 | 0.00% | 0.00% |
| core-pdf-validate | 229 / 246 | 93.09% | 80.49% |
| Total | 25,998 / 39,319 | 66.12% | 55.29% |

The combined statement/branch headline is **63.10%**. Compared with the baseline,
629 measured statements were removed, including 598 previously missing statements.
There are now 13,321 missing statements and 37 nonempty files with no executed statements.
The percentage increase comes from removing code, not adding behavioral coverage.

The conservative static rescan found no remaining top-level production function or class
whose identifier occurs only at its definition in tracked authored Python/TOML. This does
not rule out unused methods, self-contained unused code groups, or names hidden by unrelated
same-name occurrences. Method candidates still include public APIs and framework callbacks;
none was promoted to a new high-confidence removal candidate. The OCR, serializer, CLI,
facade-method, and rendering test gaps below remain.

Ruff lint/format, mypy, ty, and all 19 import contracts passed. Every remaining root public
export resolves, and the removed package and API names are absent. The generated HTML and
JSON reports now describe this post-removal source state.

## Original coverage baseline

| Package | Statements covered | Statement coverage | Branch coverage |
| --- | ---: | ---: | ---: |
| core-pdf | 18,196 / 26,531 | 68.58% | 54.93% |
| core-pdf-spec | 7,604 / 8,990 | 84.58% | 73.27% |
| core-pdf-ocr | 0 / 4,181 | 0.00% | 0.00% |
| core-pdf-validate | 229 / 246 | 93.09% | 80.49% |
| Total | 26,029 / 39,948 | 65.16% | 54.73% |

Coverage.py's combined statement/branch headline is **62.26%**. There are 13,919 missing
statements and 38 nonempty source files with no executed statements. Vendor code is omitted;
coverage.py's default exclusions remain in effect (554 excluded statements).

## Removed internal helper

`src/core_pdf/impl/_impl/model/geometry.py:205` — `page_rotation_matrix`

- All nine executable statements and all six branches were uncovered.
- Its identifier occurred only at its definition across tracked authored Python,
  Markdown, and TOML files, including tests, scripts, and export tables.
- It was an internal implementation helper, not part of the public or spec extension API.
- It had no decorator or apparent registration mechanism.

The helper was subsequently removed after this baseline. Its defining statement and nine
body statements were deleted; no caller needed updating. The scan cannot establish whether
out-of-repository consumers improperly imported it, or exclude arbitrary constructed-name
dispatch.

## Removed after this baseline

The unsupported PyMuPDF facade contained **549 statements with zero coverage**. It and its
otherwise empty parent package have now been removed, along with the type-checker exclusion
and API/roadmap references. The third-party PyMuPDF test dependency and fixture corpus remain
because x-ray differential tests use them. The table above records the pre-removal baseline.

## Test gaps, not deletion candidates

`PdfPage.text_diagnostics()` and `Document.edit()` were also retired after
the baseline. Their diagnostic result records/exports and the otherwise unused internal
`DocumentEditor` were removed with them. Shared geometry-diagnostic helpers remain in use.
This is an intentional API removal, not an inference that uncovered public APIs are dead.

- **OCR: 4,181 statements uncovered.** The configured suite has no OCR test path, and core
  intentionally never imports the companion. Add tests using `core_pdf_ocr.PdfDocument`
  and representative raster/vector inputs before evaluating unused OCR internals.
- **Output serialization: 231 statements uncovered** in
  `src/core_pdf/impl/_impl/output/serialize.py`. Public `Document.to_json`, `to_html`,
  `to_markdown`, `to_csv`, and `to_tei` methods call these serializers. Exercise those
  public methods and validate output content.
- **Core CLI: 79 statements uncovered** in `src/core_pdf/cli.py`. It is a registered
  command entry point. Add command-level success/error and output-format checks.
- **pdfplumber facade:** 752 missing statements, with 45.23% statement coverage.
  In particular, `Page.to_image` and `Page.dedupe_chars` never execute. Broaden method-level
  differential coverage, not just the number of PDFs passed to existing methods.
- **Rendering:** `impl/_impl/render/patterns.py` has 216 missing statements and 10.74%
  statement coverage. Rendering helpers such as `fill_circle` and `internal_dash_subpath`
  have explicit callers despite uncovered bodies. Add pattern, shading, and stroke cases.

## Review limits and next steps

The initial static pass parsed top-level definitions and methods, then checked identifier
occurrences across tracked authored Python, Markdown, and TOML. The post-removal pass counted
Python/TOML references separately so mentions in this report cannot hide candidates, then
reviewed documentation and exports. It is a conservative reference heuristic,
not a complete call graph. Framework callbacks such as the OCR HTML parser's `handle_starttag`,
`handle_data`, and `handle_endtag` must be retained even when no explicit caller appears.
Public spec exports and documented extension methods are compatibility contracts.

Add native output/CLI and companion OCR coverage.
Then rerun coverage before reviewing deeper
implementation clusters. The HTML report at `htmlcov/index.html` and function-level JSON at
`htmlcov/coverage.json` are the detailed evidence for this baseline.
