# Coverage and unused-code review

See the [initial findings](unused-code-review.md) for the measured baseline and reviewed candidates.

Run the configured suite from the repository root:

```sh
uv run --locked --all-packages --extra unstructured --group test --group vendor-test \
  pytest -n auto --cov --cov-config=pyproject.toml \
  --cov-report=term --cov-report=html --cov-report=json -ra
```

Open `htmlcov/index.html` for annotated source and branch results. The machine-readable
report is `htmlcov/coverage.json`; the underlying coverage.py database is `.coverage`.
These generated files are ignored by Git. A normal run replaces the previous results.

Coverage includes all four authored source packages, including modules never imported,
and excludes vendored code. Branch coverage is enabled. The headline coverage percentage
combines statements and branches; JSON also contains their separate percentages.
Use bare `--cov` to preserve the configured source roots. pytest-cov combines coverage
from xdist workers; `coverage run -m pytest -n auto` alone does not provide that integration.
See the [pytest-cov configuration](https://pytest-cov.readthedocs.io/en/stable/config.html)
and [xdist documentation](https://pytest-cov.readthedocs.io/en/latest/xdist.html).

The default test paths cover compatibility differentials, strict spec tests, and validation
tests. They do not exercise every native API, CLI, or OCR workflow. Real veraPDF execution
tests require `CORE_PDF_VERAPDF`; otherwise they skip. The default differential matrix also
omits selected expensive fixtures and uses each facade's own corpus. Prefix the command
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
