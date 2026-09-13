# Repository Guidelines

## Project Structure & Module Organization

OCR and vector text recognition live in the separately installable uv workspace member
`packages/core-pdf-ocr/src/core_pdf_ocr`.
The companion depends on the exact matching core version. Core must never import or discover
it; users opt in through `core_pdf_ocr.PdfDocument` or the `core-pdf-ocr` command.

The independently versioned `core-pdf-spec` workspace member lives at
`packages/core-pdf-spec/src/core_pdf_spec`. Core depends on its supported version range;
spec must never import core or OCR, including type-only imports. Spec exposes low-level
chapter APIs, not a document facade or CLI. Its exported symbols and documented parsing
extension methods form the cross-distribution compatibility contract.

The optional `core-pdf-validate` workspace member owns external validator adapters and reports.
Core, spec, and OCR must never import or discover it. Explicit validation uses original source
bytes; only declaration discovery depends on public core APIs. Validator execution, temporary
files, and report normalization stay outside spec. See `docs/standards.md`.

The Unstructured compatibility facade requires `core-pdf[unstructured]`, including the pinned
`en_core_web_sm` model. Its import must fail clearly if the model cannot load; do not add lexical
fallbacks or runtime model installation. The parent compatibility package stays lazy so native
core and other facades remain usable without the extra. Use `--extra unstructured` for
differential runs; ordinary `uv sync`/`uv run` commands may otherwise remove the model.

This is a Python 3.13+ PDF parsing engine using the `src` layout. Production code is in `src/core_pdf`; public entry points include `cli.py`, `__main__.py`, and `__init__.py`. `src/core_pdf/api/document.py` owns the public `PdfDocument` and `PdfPage` APIs; third-party compatibility facades live in `src/core_pdf/api/compat`. Nothing under `impl/` may import from `api/`. Internal implementation is organized under `src/core_pdf/impl`:

- `packages/core-pdf-spec/src/core_pdf_spec/` contains PDF-defined semantics and referenced-standard algorithms, one subpackage per spec chapter (`s_07_syntax`, `s_08_graphics`, `s_09_fonts`, …). Reader recovery, substitute fonts, Unicode guesses, capture products, selected device profiles, and raster preparation belong under `_impl/`. This boundary also applies to type-only imports.
- `impl/_impl/document/` composes source/lifecycle, recovery adapters, page operations, and navigation/metadata/field/structure projections. `impl/_impl/capture/` records interpreter events into runs, glyph observations, page programs, and renderer commands. `impl/_impl/fonts/` owns font selection, Unicode recovery, raster adapters, and substitute font assets. `impl/_impl/graphics/` owns color output, raster preparation, codec selection, and tolerant filter/function adapters.
- `impl/types.py` defines core capture records and source protocols and re-exports spec-owned PDF primitive identities. `core_pdf_spec.types` and `core_pdf_spec.exceptions` own shared foundational types and errors. Keep module-level functions in their owning implementation modules; shared text normalization belongs in `impl/_impl/model/text.py`.
- `impl/_impl/extract/` is the native extraction pipeline. Recognition routing, learning, OCR tasks, and recognition-specific output policies belong to the companion. `extract/__init__.py` re-exports only the pipeline entry points; import stage internals from the owning submodule.
- `impl/_impl/render/` rasterizes; `impl/_impl/output/model.py` defines structured output and `impl/_impl/output/serialize.py` emits markdown/HTML/JSON. Import these defining modules directly; `output/__init__.py` is not a facade. `impl/_impl/model/` owns shared geometry/text models, text primitives, and page-selection normalization, and `impl/_impl/layout/` separates block construction, region partitioning, reading order, and text reconstruction. `impl/_impl/runtime/` holds engine-independent infrastructure and must not import from `core_pdf_spec` or the derived-processing packages beside it.
- `src/core_pdf/_vendor/fontTools` is vendored third-party code, excluded from linting, typing, and formatting.

The authored test suite includes differential comparisons under
`tests/src/core_pdf/api/compat/differential` and strict package tests under
`packages/core-pdf-spec/tests` and `packages/core-pdf-validate/tests`.
Reference corpora remain in `tests/fixtures`. `docs/` holds `architecture.md`, `api.md`, `roadmap.md`, and
licensing material; maintenance scripts are in `scripts/`.

Start with `docs/architecture.md` — it describes the pipeline and how the source tree is organized.

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
run, pass a facade's test file under `tests/src/core_pdf/api/compat/differential`.
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
Remove any such `.so` from all four source roots before validating source changes.

### Two type checkers, contradictory advice

Both `mypy` and `ty` gate this repo, and they disagree about several `cast()`
calls: mypy reports them as redundant while `ty` requires them. Because of that,
mypy's `warn_redundant_casts` is deliberately left off. Run both before assuming
a typing change is an improvement.

## Dependency Management

Never edit `pyproject.toml` or `uv.lock` manually when adding or removing dependencies. Use `uv add --group <group> <package>` or `uv remove --group <group> <package>`; these commands update project metadata and the lockfile, and both generated changes should be reviewed and committed. Use the existing groups for their intended purpose: `test` for pytest and its runner, `vendor-test` for reference libraries used by differential tests, and `lint` for Ruff, mypy, and ty. For example, add a test dependency with `uv add --group test pytest-xdist`; do not create a new group when an existing group fits.

## Coding Style & Naming Conventions

Write Python with four-space indentation, clear type annotations, and lines no longer than 100 characters. Ruff handles import sorting, linting, and formatting; run it before submitting. Use `snake_case` for modules, functions, and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.

Module-level symbols that are not part of a module's interface are prefixed `internal_` rather than with a leading underscore — about 490 of them. Treat anything so prefixed as private. The convention is applied unevenly across subpackages, so its *absence* does not imply a symbol is public; nothing under `impl/` is. Where a module declares `__all__`, that is the more reliable signal. Two wrinkles worth knowing: `internal_EXPORTS` in `__init__.py` is the public export table (the prefix marks the variable as private, not its contents), and a handful of constants are spelled `internal_UPPER_CASE`.

Dependency direction is enforced, not conventional. `import-linter` contracts in `[tool.importlinter]` (`pyproject.toml`) pin the derived-processing layering, the spec layering, the spec/reader policy boundary, and the foundational packages that must not depend upward (`impl/_impl/runtime/`, `impl/_impl/model/`, `core_pdf_spec.s_07_syntax`). They run in the `pre-push` prek stage that CI executes. If a change needs a new edge that a contract forbids, the edge is usually the bug -- read the "Dependency direction" section of `docs/architecture.md` before editing the contract.

Third-party implementations belong in the owning package's `_vendor/` directory.
The fontTools backend stays in `src/core_pdf/_vendor/`; spec may contain attributed inert
standard tables, but no fontTools backend imports or reader recovery.
Spec algorithms have strict defaults: core owns retry/skip/substitute orchestration.
Specification-defined defaults and prescribed fallback behavior remain in spec.

## Testing Guidelines

Tests use pytest and pytest-xdist, and are named `test_*.py`, with test functions
named `test_<behavior>`. Facade tests in
`tests/src/core_pdf/api/compat/differential` compare a facade with its reference
implementation over the same PDF. Strict tests in `packages/core-pdf-spec/tests` must
run without core or OCR installed.
Use spec citations for non-obvious mandated behavior and positive controls for valid defaults.
Preserve reference fixture contents and distinguish compatibility differences from failures
on both sides. Validate with `uv run --locked --all-packages --extra unstructured --group test --group vendor-test pytest -n auto`.

## Commit & Pull Request Guidelines

Use short Conventional Commit-style subjects such as `feat(ocr): ...`, `fix: ...`, `test(corpus): ...`, and `ci: ...`. Keep commits focused and explain the user-visible or correctness impact. Pull requests should describe the change, motivation, validation commands, fixture or compatibility impacts, and link related issues. Include representative output or screenshots when changing CLI behavior or documentation.

## Local and CI Validation

Local commands may use the installed environment directly. To reproduce CI’s locked dependency validation, use `uv run --locked` with the relevant group, such as `uv run --locked --all-packages --extra unstructured --group test --group vendor-test pytest -n auto` or `uv run --locked --group lint mypy`. Do not use `--locked` while intentionally changing dependencies; update them with `uv add` or `uv remove` first.
