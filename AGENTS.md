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

Five standards packages sit beneath spec as the workspace floor, one per external
specification family: `core-predictors` (PNG and TIFF predictors), `core-postscript` (the
PLRM calculator subset), `core-jbig2` (ITU-T T.88), `core-pdf-crypto` (ciphers, RFC 5652 CMS,
ISO/TS 32003 and 32004), and `core-adobe-fonts` (CFF, Type 2, Type 1, CMaps, the Adobe Glyph
List, and Core 14 metrics, with the CMap data). Each lives at `packages/<name>/src/<name>/`
and imports nothing from core, spec, OCR, or validate, including type-only imports; the floor
packages are independent of each other except that any may use `core-predictors`, which also
hosts the shared numpy sample-view helpers. PDF glue (DecodeParms, PDF objects, spec exceptions, T.88 polarity
inversion) stays in the spec chapter that ISO 32000 assigns; kernels take bytes and plain
Python values. Spec pins each floor package's minor range; core pins the three it imports
directly. See `tests/fixtures/specifications/README.md` for the document-to-package map.

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

This is a Python 3.14+ PDF parsing engine using the `src` layout. Production code is in `src/core_pdf`; public entry points include `cli.py`, `__main__.py`, and `__init__.py`. The public `PdfDocument` and `PdfPage` classes are the engine classes in `impl/document/document.py` and `impl/document/page.py`, re-exported by `core_pdf/__init__.py` together with `DocumentAdapter`. Internal implementation is organized under `src/core_pdf/impl`:

- `packages/core-pdf-spec/src/core_pdf_spec/` contains PDF-defined semantics and referenced-standard algorithms, one subpackage per spec chapter (`s_07_syntax`, `s_08_graphics`, `s_09_fonts`, …); code for a referenced external standard lives in the floor packages above and spec keeps only the PDF wrapper. Reader recovery, substitute fonts, Unicode guesses, capture products, selected device profiles, and raster preparation belong under `impl/`. This boundary also applies to type-only imports.
- `impl/document/` composes source/lifecycle, recovery adapters, page operations, and navigation/metadata/field/structure projections. `impl/capture/` records interpreter events into runs, glyph observations, page programs, and renderer commands. `impl/fonts/` owns font selection, Unicode recovery, raster adapters, and substitute font assets. `impl/graphics/` owns color output, raster preparation, codec selection, and tolerant filter/function adapters.
- `impl/types.py` defines core capture records and source protocols and re-exports spec-owned PDF primitive identities. `core_pdf_spec.types` and `core_pdf_spec.exceptions` own shared foundational types and errors. Keep module-level functions in their owning implementation modules; shared text normalization belongs in `impl/model/text.py`.
- `impl/extract/` is the native extraction pipeline. Recognition routing, learning, OCR tasks, and recognition-specific output policies belong to the companion. `extract/__init__.py` re-exports only the pipeline entry points; import stage internals from the owning submodule.
- `impl/render/` rasterizes; `impl/output/model.py` defines structured output and `impl/output/serialize.py` emits markdown/HTML/JSON. Import these defining modules directly; `output/__init__.py` is not a facade. `impl/model/` owns shared geometry/text models, text primitives, and page-selection normalization, and `impl/layout/` separates block construction, region partitioning, reading order, and text reconstruction. `impl/runtime/` holds engine-independent infrastructure and must not import from `core_pdf_spec` or the derived-processing packages beside it; it may use the standards packages, which sit below spec.
- `src/core_pdf/_vendor/fontTools` is vendored third-party code, excluded from linting, typing, and formatting.

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

### Compiled modules must not shadow sources

A Nuitka module build can leave a `<module>.cpython-*.so` next to its `.py` in
`src/`. Python's `ExtensionFileLoader` wins over `SourceFileLoader`, so the stale
binary is imported instead of the source — and it still reports the `.py` path as
`__file__`, so nothing looks wrong. Edits to that module then silently do nothing.
Remove any such `.so` from every source root listed in `[tool.coverage.run].source` before validating source changes; the root `conftest.py` reads that list and aborts the run if one is found.

### Two type checkers, contradictory advice

Both `mypy` and `ty` gate this repo, and they disagree about several `cast()`
calls: mypy reports them as redundant while `ty` requires them. Because of that,
mypy's `warn_redundant_casts` is deliberately left off. Run both before assuming
a typing change is an improvement.

## Dependency Management

Never edit `pyproject.toml` or `uv.lock` manually when adding or removing dependencies. Use `uv add --group <group> <package>` or `uv remove --group <group> <package>`; these commands update project metadata and the lockfile, and both generated changes should be reviewed and committed. Use the existing groups for their intended purpose: `test` for pytest and its runner, `vendor-test` for reference libraries used by differential tests, and `lint` for Ruff, mypy, and ty. For example, add a test dependency with `uv add --group test pytest-xdist`; do not create a new group when an existing group fits. A dependency of a workspace member is added with `uv add --package <member> <package>`; for another member, pass a version range such as `"core-jbig2>=0.1.0,<0.2.0"` and uv records the `{ workspace = true }` source itself.

## Coding Style & Naming Conventions

Write Python with four-space indentation, clear type annotations, and lines no longer than 100 characters. Ruff handles import sorting, linting, and formatting; run it before submitting. Use `snake_case` for modules, functions, and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.

There is no naming prefix for private symbols. The `internal_` prefix that used to mark them is gone; `__all__`, where a module declares one, is the signal, and nothing under `impl/` is public regardless. A leading underscore is used only for a backing field that sits behind a public property of the same name, such as `_closed` behind `PdfDocument.closed`. `EXPORTS` in `core_pdf/__init__.py` is the public export table that `install_lazy_module_exports` reads.

Dependency direction is enforced, not conventional. `import-linter` contracts in `[tool.importlinter]` (`pyproject.toml`) pin the derived-processing layering, the spec layering, the spec/reader policy boundary, and the foundational packages that must not depend upward (`impl/runtime/`, `impl/model/`, `core_pdf_spec.s_07_syntax`), and the standards-package floor (they import no other workspace package and not each other). They run in the `pre-push` prek stage that CI executes. If a change needs a new edge that a contract forbids, the edge is usually the bug -- read the contract's own description in `pyproject.toml` before editing it.

Third-party implementations belong in the owning package's `_vendor/` directory.
The fontTools backend stays in `src/core_pdf/_vendor/`; spec may contain attributed inert
standard tables, but no fontTools backend imports or reader recovery.
Spec algorithms have strict defaults: core owns retry/skip/substitute orchestration.
Specification-defined defaults and prescribed fallback behavior remain in spec.

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
