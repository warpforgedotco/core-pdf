# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.13+ PDF parsing engine using the `src` layout. Production code is in `src/core_pdf`; public entry points include `cli.py`, `__main__.py`, and `__init__.py`. Third-party compatibility facades live in `src/core_pdf/api/compat`. Internal implementation is organized under `src/core_pdf/impl`:

- `impl/engine/spec/` implements the PDF specification, one subpackage per spec chapter (`s_07_syntax`, `s_08_graphics`, `s_09_fonts`, …).
- `impl/engine/parse/` is the extraction pipeline, one module per stage (capture → route → fusion → tables → OCR → layout → emit). `parse/__init__.py` re-exports only the pipeline entry points and shared stage models; import stage internals from the owning submodule.
- `impl/engine/render/` rasterizes; `impl/engine/writing/` produces PDF output; `impl/engine/structured/` serializes to markdown/HTML/JSON; `impl/engine/model/` holds the capture records (geometry, text runs, glyphs) and `impl/engine/layout/` holds the heuristics that consume them. `impl/runtime/` holds engine-independent infrastructure and must not import from `impl/engine/`.
- `src/core_pdf/_vendor/fontTools` is vendored third-party code, excluded from linting, typing, and formatting.

Tests live under `tests/`: `tests/src` mirrors the package structure, while broader pipeline tests (`test_parse_*.py`, `test_rendering.py`, …) sit at the top level. Corpus fixtures are in `tests/fixtures`. `docs/` holds `architecture.md`, `api.md`, `roadmap.md`, and licensing material; maintenance scripts are in `scripts/`.

Start with `docs/architecture.md` — it describes the pipeline and how the source tree is organized.

## Build, Test, and Development Commands

Use `uv` for environments and locked dependencies:

```sh
uv sync --all-groups                 # install development dependencies
uv run pytest tests/ -n auto          # run the full test suite in parallel
uv run --group lint ruff check .     # lint Python files
uv run --group lint ruff format --check .
uv run --group lint mypy             # static type checking
uv run --group lint lint-imports     # architecture: layer and dependency contracts
uv run --group lint --group test --group benchmark ty check
prek run --all-files                 # run repository hooks across all files
```

After making broad changes, run the full suite with `uv run pytest tests/ -n auto`. Otherwise, test a subset covering the code and behavior affected by the changes, for example `uv run pytest tests/src/core_pdf/impl/engine/model/test_glyphs.py`. CI checks the lockfile, runs `prek` at the `pre-push` stage, and runs the test suite on Python 3.13 on Ubuntu.

### Coverage

```sh
uv run pytest tests/ -n auto --ignore=tests/benchmarks --cov --cov-report=term
```

Coverage is configured in `[tool.coverage]` and excludes vendored fontTools, so
the total describes code this project owns. `fail_under` is a ratchet set just
below the measured figure — raise it as gaps close rather than lowering it.

Rendering changes are additionally pinned by golden rasters; see the "Golden
rasters" section of `docs/architecture.md` before changing anything under
`impl/engine/render/`.

### Compiled modules must not shadow sources

A Nuitka module build can leave a `<module>.cpython-*.so` next to its `.py` in
`src/`. Python's `ExtensionFileLoader` wins over `SourceFileLoader`, so the stale
binary is imported instead of the source — and it still reports the `.py` path as
`__file__`, so nothing looks wrong. Edits to that module then silently do nothing.
`tests/conftest.py` fails the session if any such pair exists; delete the `.so`.

### Two type checkers, contradictory advice

Both `mypy` and `ty` gate this repo, and they disagree about several `cast()`
calls: mypy reports them as redundant while `ty` requires them. Because of that,
mypy's `warn_redundant_casts` is deliberately left off. Run both before assuming
a typing change is an improvement.

For performance-sensitive changes, capture a benchmark baseline first:

```sh
uv run --group benchmark pytest --benchmark-only -m benchmark_high_impact \
  --benchmark-save=baseline
uv run --group benchmark pytest --benchmark-only -m benchmark_high_impact \
  --benchmark-compare=baseline
```

The `benchmark_high_impact` tier is the routine pull-request suite. Run the complete benchmark
inventory explicitly with `uv run --group benchmark pytest --benchmark-only`; CI also runs it
weekly.

Timings on the heavy page-program and OCR benchmarks are noisy — several vary by more than 5% between runs of identical code — so treat a single run as weak evidence and rely on the invariants those benchmarks assert. See `docs/architecture.md` for details.

## Dependency Management

Never edit `pyproject.toml` or `uv.lock` manually when adding or removing dependencies. Use `uv add --group <group> <package>` or `uv remove --group <group> <package>`; these commands update project metadata and the lockfile, and both generated changes should be reviewed and committed. Use the existing groups for their intended purpose: `test` for pytest and test fixtures, `lint` for Ruff, mypy, and ty, `benchmark` for benchmark tooling, and `vendor` for vendoring tooling. For example, add a test dependency with `uv add --group test pytest-xdist`; do not create a new group when an existing group fits.

## Coding Style & Naming Conventions

Write Python with four-space indentation, clear type annotations, and lines no longer than 100 characters. Ruff handles import sorting, linting, and formatting; run it before submitting. Use `snake_case` for modules, functions, and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.

Module-level symbols that are not part of a module's interface are prefixed `internal_` rather than with a leading underscore — about 490 of them. Treat anything so prefixed as private. The convention is applied unevenly across subpackages, so its *absence* does not imply a symbol is public; nothing under `impl/` is. Where a module declares `__all__`, that is the more reliable signal. Two wrinkles worth knowing: `internal_EXPORTS` in `__init__.py` is the public export table (the prefix marks the variable as private, not its contents), and a handful of constants are spelled `internal_UPPER_CASE`.

Dependency direction is enforced, not conventional. `import-linter` contracts in `[tool.importlinter]` (`pyproject.toml`) pin the engine layering, the spec layering, and the three packages that must not depend upward (`impl/runtime/`, `impl/engine/model/`, `spec/s_07_syntax`). They run in the `pre-push` prek stage that CI executes. If a change needs a new edge that a contract forbids, the edge is usually the bug -- read the "Dependency direction" section of `docs/architecture.md` before editing the contract.

Third-party code belongs in `src/core_pdf/_vendor/`; do not add new third-party implementations elsewhere in `core-pdf`.

## Testing Guidelines

Tests use pytest and pytest-xdist, and are named `test_*.py`, with test functions named `test_<behavior>`. Place unit tests beside the corresponding implementation area under `tests/src/core_pdf`; broader pipeline and regression tests belong at the top level of `tests/`. Add or update fixtures and expected extraction output when behavior changes. For broad changes, use `uv run pytest tests/ -n auto`; for focused changes, run the subset of tests that exercises the recently modified code.

## Commit & Pull Request Guidelines

Use short Conventional Commit-style subjects such as `feat(ocr): ...`, `fix: ...`, `test(corpus): ...`, and `ci: ...`. Keep commits focused and explain the user-visible or correctness impact. Pull requests should describe the change, motivation, validation commands, fixture or compatibility impacts, and link related issues. Include representative output or screenshots when changing CLI behavior or documentation.

## Local and CI Validation

Local commands may use the installed environment directly. To reproduce CI’s locked dependency validation, use `uv run --locked` with the relevant group, such as `uv run --locked --group test pytest` or `uv run --locked --group lint mypy`. Do not use `--locked` while intentionally changing dependencies; update them with `uv add` or `uv remove` first.
