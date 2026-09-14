# Coverage and unused-code findings

Baseline: `fdd52fee`, with coverage configuration and test dependencies added in the
working tree. Run completed September 13, 2026 (America/Bogota), using Python 3.13.14,
coverage.py 7.16.1, pytest-cov 7.1.0, and eight xdist workers.

The [configured coverage run](coverage.md) passed: **2,608 passed, 32 skipped in 196.12s**.
The skipped tests are the opt-in installed veraPDF integration cases. This was the default
facade-owned differential matrix, not the exhaustive cross-corpus matrix.

## Coverage baseline

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

- All nine executable statements and all six branches are uncovered.
- Its identifier occurs only at its definition across tracked authored Python,
  Markdown, and TOML files, including tests, scripts, and export tables.
- It is an internal implementation helper, not part of the public or spec extension API.
- It has no decorator or apparent registration mechanism.

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

- **OCR: 4,181 statements uncovered.** The configured suite has no OCR test path, and core
  intentionally never imports the companion. Add tests using `core_pdf_ocr.PdfDocument`
  and representative raster/vector inputs before evaluating unused OCR internals.
- **Output serialization: 231 statements uncovered** in
  `src/core_pdf/impl/_impl/output/serialize.py`. Public `Document.to_json`, `to_html`,
  `to_markdown`, `to_csv`, and `to_tei` methods call these serializers. Exercise those
  public methods and validate output content.
- **Core CLI: 79 statements uncovered** in `src/core_pdf/cli.py`. It is a registered
  command entry point. Add command-level success/error and output-format checks.
- **Native API:** `PdfPage.text_diagnostics` and `Document.edit` never execute.
  Both expose supported behavior; their lack of internal callers does not make them dead.
- **pdfplumber facade:** 752 missing statements, with 45.23% statement coverage.
  In particular, `Page.to_image` and `Page.dedupe_chars` never execute. Broaden method-level
  differential coverage, not just the number of PDFs passed to existing methods.
- **Rendering:** `impl/_impl/render/patterns.py` has 216 missing statements and 10.74%
  statement coverage. Rendering helpers such as `fill_circle` and `internal_dash_subpath`
  have explicit callers despite uncovered bodies. Add pattern, shading, and stroke cases.

## Review limits and next steps

The static pass parsed top-level definitions and methods, then checked identifier occurrences
across tracked authored Python, Markdown, and TOML. It is a conservative reference heuristic,
not a complete call graph. Framework callbacks such as the OCR HTML parser's `handle_starttag`,
`handle_data`, and `handle_endtag` must be retained even when no explicit caller appears.
Public spec exports and documented extension methods are compatibility contracts.

Add native output/CLI and companion OCR coverage.
Then rerun coverage before reviewing deeper
implementation clusters. The HTML report at `htmlcov/index.html` and function-level JSON at
`htmlcov/coverage.json` are the detailed evidence for this baseline.
