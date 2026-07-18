# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.11+ PDF parsing engine using the `src` layout. Production code is in `src/core_pdf`; public entry points include `cli.py`, `__main__.py`, and `__init__.py`. Internal implementation is organized under `src/core_pdf/impl`, including the PDF specification, extraction, rendering, writing, and tables. Reusable layout, glyph, and OCR components are separate workspace packages under `packages/`. Tests mirror the package structure in `tests/src`; corpus fixtures live under `tests/fixtures`. Documentation and licensing material are in `docs/`, and maintenance scripts are in `scripts/`.

## Build, Test, and Development Commands

Use `uv` for environments and locked dependencies:

```sh
uv sync --all-groups                 # install development dependencies
uv run pytest tests/ -n auto          # run the full test suite in parallel
uv run --group lint ruff check .     # lint Python files
uv run --group lint ruff format --check .
uv run --group lint mypy             # static type checking
uv run --group lint --group test --group benchmark ty check
prek run --all-files                 # run repository hooks across all files
```

After making broad changes, run the full suite with `uv run pytest tests/ -n auto`. Otherwise, test a subset covering the code and behavior affected by the changes, for example `uv run pytest tests/src/core_pdf/impl/engine/layout/test_glyphs.py`. CI also checks the lockfile and runs tests on Python 3.11–3.14 and Windows.

## Dependency Management

Never edit `pyproject.toml` or `uv.lock` manually when adding or removing dependencies. Use `uv add --group <group> <package>` or `uv remove --group <group> <package>`; these commands update project metadata and the lockfile, and both generated changes should be reviewed and committed. Use the existing groups for their intended purpose: `test` for pytest and test fixtures, `lint` for Ruff, mypy, and ty, `benchmark` for benchmark tooling, and `vendor` for vendoring tooling. For example, add a test dependency with `uv add --group test pytest-xdist`; do not create a new group when an existing group fits.

## Coding Style & Naming Conventions

Write Python with four-space indentation, clear type annotations, and lines no longer than 100 characters. Ruff handles import sorting, linting, and formatting; run it before submitting. Use `snake_case` for modules, functions, and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Third-party implementations do not belong in `core-pdf`; add reusable functionality to the appropriate workspace package instead.

## Testing Guidelines

Tests use pytest and pytest-xdist, and are named `test_*.py`, with test functions named `test_<behavior>`. Place new tests beside the corresponding implementation area under `tests/src/core_pdf`. Add or update fixtures and expected extraction output when behavior changes. For broad changes, use `uv run pytest tests/ -n auto`; for focused changes, run the subset of tests that exercises the recently modified code.

## Commit & Pull Request Guidelines

Use short Conventional Commit-style subjects such as `feat(ocr): ...`, `fix: ...`, `test(corpus): ...`, and `ci: ...`. Keep commits focused and explain the user-visible or correctness impact. Pull requests should describe the change, motivation, validation commands, fixture or compatibility impacts, and link related issues. Include representative output or screenshots when changing CLI behavior or documentation.

## Local and CI Validation

Local commands may use the installed environment directly. To reproduce CI’s locked dependency validation, use `uv run --locked` with the relevant group, such as `uv run --locked --group test pytest` or `uv run --locked --group lint mypy`. Do not use `--locked` while intentionally changing dependencies; update them with `uv add` or `uv remove` first.
