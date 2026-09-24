# Repository Guidelines

## Project Structure & Module Organization

OCR and vector text recognition live in the separately installable uv workspace member
`packages/core-pdf-ocr/src/core_pdf_ocr`.
The companion depends on the exact matching core version. Core must never import or discover
it; users opt in through `core_pdf_ocr.PdfDocument` or the `core-pdf-ocr` command.

Third-party compatibility facades (pdfminer, pdfplumber, pypdf, pikepdf, unstructured,
llamaindex, x-ray) live in the workspace member `packages/core-pdf-compat/src/core_pdf_compat`,
one subpackage per facade plus the `_shared`, `_text_state`, and `_strict_page_tree` helpers
they share. The facades use `core_pdf.impl` internals by design, so the package pins the
exact matching core version like OCR; do not add a "public interfaces only" contract for it.
Core, spec, OCR, and validate must never import or discover it. Bumping core's version now
bumps `core-pdf-ocr` and `core-pdf-compat` with it.

The independently versioned `core-pdf-spec` workspace member lives at
`packages/core-pdf-spec/src/core_pdf_spec`. Core depends on its supported version range;
spec must never import core or OCR, including type-only imports. Spec exposes low-level
chapter APIs, not a document facade or CLI. Its exported symbols and documented parsing
extension methods form the cross-distribution compatibility contract.

Four standards packages sit beneath spec as the workspace floor, one per external
specification family: `core-postscript` (the PLRM calculator subset), `core-jbig2`
(ITU-T T.88), `core-pdf-crypto` (ciphers, RFC 5652 CMS, ISO/TS 32003 and 32004), and
`core-adobe-fonts` (CFF, Type 2, Type 1, CMaps, the Adobe Glyph List, and Core 14 metrics,
with the CMap data). Each lives at `packages/<name>/src/<name>/` and imports nothing from
core, spec, OCR, or validate, including type-only imports; the floor packages are also
independent of each other. The PNG and TIFF predictor kernels are the exception to the
one-package-per-standard rule: at ~180 lines, with spec as their only consumer, they live
in `core_pdf_spec.s_07_filters.predictors` beside the wrappers that call them, and the
sample helper both they and image decoding share is `core_pdf_spec.samples`. PDF glue (DecodeParms, PDF objects, spec exceptions, T.88 polarity
inversion) stays in the spec chapter that ISO 32000 assigns; kernels take bytes and plain
Python values. Spec pins each floor package's minor range; core pins the two it imports
directly. See `tests/fixtures/specifications/README.md` for the document-to-package map.

`packages/core-pdf-cythonized/src/core_pdf_cythonized` holds compiled kernels and is the
only workspace member shipping a C extension, so `core-pdf` is a compiled distribution and
needs a wheel or a compiler. There is no pure-Python fallback by design: when a kernel lands
there the Python it replaced is deleted, so nothing can silently diverge between a compiled
and an interpreted path. A kernel belongs there only if it is measured against a profiled
workload, its inner loop touches no Python objects (Cython loses to CPython's specializing
interpreter on object-heavy code, measured at roughly 2x slower for per-item construction),
and it is pinned by golden vectors generated from the original before deletion.
One kernel, `composite_knockout_element`, owns its algorithm outright: ISO 32000-2
11.4.x knockout compositing moved out of spec, with its conformance tests, so that
spec could stay pure Python rather than become a compiled distribution. That is the
only member of the package that is not a mirror of code owned elsewhere. The build
disables float contraction -- `-ffp-contract=off` on GCC/Clang, `/fp:precise` on MSVC,
chosen per compiler in `setup.py` because passing the wrong spelling would build with
semantics the golden vectors do not describe. The kernels must reproduce CPython float
semantics exactly, and compilers contract expressions into FMAs that shift results by an
ULP. `.github/workflows/wheels.yml` builds wheels with cibuildwheel and runs the golden
vectors against each one, which is the only cross-platform check this repo has.

The optional `core-pdf-validate` workspace member owns external validator adapters and reports.
Core, spec, and OCR must never import or discover it. Explicit validation uses original source
bytes; only declaration discovery depends on public core APIs. Validator execution, temporary
files, and report normalization stay outside spec. See `docs/standards.md`.

The Unstructured compatibility facade requires `core-pdf-compat[unstructured]`, including the
pinned `en_core_web_sm` model. Its import must fail clearly if the model cannot load; do not add
lexical fallbacks or runtime model installation. The `core_pdf_compat` package root stays lazy so
native core and other facades remain usable without the extra. Use `--all-packages --extra
unstructured` for differential runs (the extra belongs to the compat member, so root-only
commands cannot select it); ordinary `uv sync`/`uv run` commands may otherwise remove the model.

This is a Python 3.14+ PDF parsing engine using the `src` layout. Production code is in `src/core_pdf`; public entry points include `cli.py`, `__main__.py`, and `__init__.py`.
Inside `src/core_pdf/impl`, each subpackage owns one feature and code shared between them
lives in modules at the `impl` root. Root modules never import a subpackage; the pure helpers
(`scalars`, `array_views`, `execution`) import no spec either. A new root module must be added
to the root import contract in `pyproject.toml`, which `test_impl_root_contracts.py` enforces.

The authored test suite includes facade tests and differential comparisons under
`packages/core-pdf-compat/tests`, strict spec tests under
`packages/core-pdf-spec/tests`, each standards package's tests under its own
`packages/<name>/tests` (which import nothing from spec or core), and validation tests under
`packages/core-pdf-validate/tests`. Reference corpora remain in
`tests/fixtures`. `docs/` holds `api.md`, `standards.md`, `roadmap.md`, and
licensing material; maintenance scripts are in `scripts/`.

## Build, Test, and Development Commands

Use `uv` for environments and locked dependencies:

```sh
uv sync --all-packages --all-groups --extra unstructured                 # install development dependencies
uv run --locked --all-packages --extra unstructured --group test --group vendor-test pytest -n auto  # differential suite
uv run --all-packages --group lint ruff check .     # lint Python files
uv run --all-packages --group lint ruff format --check .
uv run --all-packages --group lint mypy             # static type checking
uv run --all-packages --group lint lint-imports     # architecture: layer and dependency contracts
uv run --all-packages --group lint --group test --group vendor-test ty check
prek run --all-files                 # run repository hooks across all files
```

Run the differential suite after changes affecting compatibility behavior. To focus a
run, pass a facade's test file under `packages/core-pdf-compat/tests/differential`.
The default matrix uses each facade's own reference corpus, with selected cross-corpus
cases for x-ray. Run every facade against every fixture explicitly with:

```sh
CORE_PDF_COMPAT_DIFFERENTIAL_FULL=1 \
  uv run --locked --all-packages --extra unstructured --group test --group vendor-test pytest -n auto
```

Initialize reference corpora with `git submodule update --init --recursive` before
running the suite. CI checks the lockfile and runs repository hooks alongside the
locked differential command.

## Dependency Management

Never edit `pyproject.toml` or `uv.lock` manually when adding or removing dependencies. Use `uv add --group <group> <package>` or `uv remove --group <group> <package>`; these commands update project metadata and the lockfile, and both generated changes should be reviewed and committed. Use the existing groups for their intended purpose: `test` for pytest and its runner, `vendor-test` for reference libraries used by differential tests, and `lint` for Ruff, mypy, and ty. For example, add a test dependency with `uv add --group test pytest-xdist`; do not create a new group when an existing group fits. A dependency of a workspace member is added with `uv add --package <member> <package>`; for another member, pass a version range such as `"core-jbig2>=0.1.0,<0.2.0"` and uv records the `{ workspace = true }` source itself.

## Testing Guidelines

Tests use pytest and pytest-xdist, and are named `test_*.py`, with test functions
named `test_<behavior>`. Facade tests in
`packages/core-pdf-compat/tests/differential` compare a facade with its reference
implementation over the same PDF. Strict tests in `packages/core-pdf-spec/tests` must
run without core or OCR installed.
Use spec citations for non-obvious mandated behavior and positive controls for valid defaults.
Preserve reference fixture contents and distinguish compatibility differences from failures
on both sides. Validate with `uv run --locked --all-packages --extra unstructured --group test --group vendor-test pytest -n auto`.

## Commit & Pull Request Guidelines

Use short Conventional Commit-style subjects such as `feat(ocr): ...`, `fix: ...`, `test(corpus): ...`, and `ci: ...`. Keep commits focused and explain the user-visible or correctness impact. Pull requests should describe the change, motivation, validation commands, fixture or compatibility impacts, and link related issues. Include representative output or screenshots when changing CLI behavior or documentation.

## Local and CI Validation

CI also installs each floor package and spec on its own (`uv sync --locked --package <member> --group test`), runs `scripts/check_package_isolation.py <member>` to prove the tiers above it are absent, and then runs that member's tests. Reproduce it locally the same way; it replaces `.venv`, so run a full `uv sync --all-packages` afterwards.

Local commands may use the installed environment directly. To reproduce CI’s locked dependency validation, use `uv run --locked` with the relevant group, such as `uv run --locked --all-packages --extra unstructured --group test --group vendor-test pytest -n auto` or `uv run --locked --group lint mypy`. Do not use `--locked` while intentionally changing dependencies; update them with `uv add` or `uv remove` first.
