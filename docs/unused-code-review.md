# Coverage and unused-code findings

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
