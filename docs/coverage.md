# Coverage and unused-code review

The current statement and branch floors are the exact fractions in
[`coverage-baseline.json`](coverage-baseline.json), which CI enforces. The
[unused-code review log](unused-code-review.md) is a historical record of past reviews,
their coverage at the time, and the candidates they settled.

Set `CORE_PDF_VERAPDF` to the veraPDF 1.30.2 executable and install the
[pinned OCR prerequisites](../packages/core-pdf-ocr/tests/fixtures/README.md).
Run the complete configured suite from the repository root:

```sh
CORE_PDF_TESSERACT_TESTS=1 CORE_PDF_REQUIRE_VERAPDF=1 \
uv run --locked --all-packages --extra unstructured --group test --group vendor-test \
  pytest -n auto --cov --cov-config=pyproject.toml \
  --cov-report=term --cov-report=html --cov-report=json -ra
```

Open `htmlcov/index.html` for annotated source and branch results. The machine-readable
report is `htmlcov/coverage.json`; the underlying coverage.py database is `.coverage`.
These generated files are ignored by Git. A normal run replaces the previous results.

Coverage includes every authored source package listed in `[tool.coverage.run].source`, including modules never imported,
and excludes vendored code. Branch coverage is enabled. The headline coverage percentage
combines statements and branches; JSON also contains their separate percentages.
Use bare `--cov` to preserve the configured source roots. pytest-cov combines coverage
from xdist workers; `coverage run -m pytest -n auto` alone does not provide that integration.
See the [pytest-cov configuration](https://pytest-cov.readthedocs.io/en/stable/config.html)
and [xdist documentation](https://pytest-cov.readthedocs.io/en/latest/xdist.html).

The default test paths cover native core tests (including CLI and assembly checks),
compatibility differentials, deterministic OCR tests, strict spec tests, and validation
tests. OCR tests cover assembly, routing/fusion, cross-page enrichment, cancellation,
engine ownership/failures, timeout recovery, pass scheduling, raster budgets, rotation,
and coordinate remapping using fixed inputs and engine stubs.
Serialization tests cover JSON identity, escaping, merged cells, and CSV/TEI selections.
The [pinned Tesseract suite](../packages/core-pdf-ocr/tests/fixtures/README.md)
also checks actual recognition and cleanup with authored image-only PDFs and raster crops.
Default CI and the standard command above enable it with `CORE_PDF_TESSERACT_TESTS=1`;
focused local pytest runs can omit the flag to skip its four checks.
These tests do not exercise every native API or real OCR engine workflow. Real veraPDF execution
tests require `CORE_PDF_VERAPDF`; the standard command fails if it is unavailable.
Focused runs without either validator environment flag skip those integration checks.
The default differential matrix also omits selected expensive fixtures and uses each facade's own corpus. Prefix the command
with `CORE_PDF_COMPAT_DIFFERENTIAL_FULL=1` to run every facade against every fixture.

## Interpreting candidates

An uncovered line only establishes that this test run did not execute it. Before removing
a function or module:

1. Inspect its uncovered body in HTML or the JSON `functions` data. A covered `def` line
   only means that Python created the function.
2. Search callers, imports, export tables, string-based dispatch, documentation, and scripts.
3. Check public APIs, spec extension contracts, CLI entry points, inherited methods, and
   framework callbacks. External callers and dynamic dispatch can escape static searches.
4. Distinguish an unused internal implementation from a missing test or an intentionally
   optional workflow. Preserve mandated spec behavior even when fixtures never reach it.

Treat removal candidates as review findings, not automatic deletions. Add focused tests
for supported behavior; remove internal code only after establishing that it has no caller.

## Default CI and coverage floors

Every pull request runs the complete workspace suite, including pinned real Tesseract,
with the locked Unstructured extra and reference libraries. CI verifies the OCR binding's
linked engine and downloads the English model using the shared fixture manifest's checksum.
Real veraPDF tests run in their existing Java job. Both jobs must pass before their coverage
databases are combined under the existing `Differential / Python 3.14` status.

The `workspace-coverage` artifact contains HTML, JSON, and the combined `.coverage` database.
`scripts/check_coverage.py` rejects incomplete source inventories, missing branch coverage,
and regressions in either statement or branch coverage. `coverage-baseline.json` stores exact
fractions so display rounding cannot make an unchanged result fail. The floor records the
latest complete local run with both real engines enabled; confirm it on Linux CI and
ratchet it upward with subsequent complete results.
Do not update it from focused tests or lower it to accommodate a regression.

Focused local pytest invocations remain lightweight; coverage is not injected into pytest's
`addopts`. Keep every source root in that list (mirrored exactly by `docs/coverage-baseline.json`) and the existing vendor-only omit rule.
