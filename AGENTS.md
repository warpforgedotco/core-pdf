# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.11+ PDF parsing engine using the `src` layout. Production code is in `src/core_pdf`; public entry points include `cli.py`, `__main__.py`, and `__init__.py`. Internal implementation is organized under `src/core_pdf/impl`, including the PDF specification, extraction, layout, OCR, tables, and vendored third-party code. Tests mirror the package structure in `tests/src`; compatibility and corpus fixtures live under `tests/fixtures`. Documentation and licensing material are in `docs/`, benchmarks in `core/benchmarks`, and maintenance scripts in `scripts/`.

## Build, Test, and Development Commands

Use `uv` for environments and locked dependencies:

```sh
uv sync --all-groups                 # install development dependencies
uv run pytest tests/ -n auto          # run the full test suite in parallel
uv run --group lint ruff check .     # lint Python files
uv run --group lint ruff format --check .
uv run --group lint mypy             # static type checking
uv run --group lint --group test --group benchmark ty check
```

After making broad changes, run the full suite with `uv run pytest tests/ -n auto`. Otherwise, test a subset covering the code and behavior affected by the changes, for example `uv run pytest tests/src/core_pdf/impl/engine/layout/test_glyphs.py`. CI also checks the lockfile and runs tests on Python 3.11–3.14 and Windows.

## Coding Style & Naming Conventions

Write Python with four-space indentation, clear type annotations, and lines no longer than 100 characters. Ruff handles import sorting, linting, and formatting; run it before submitting. Use `snake_case` for modules, functions, and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep changes out of `src/core_pdf/impl/third_party` unless explicitly updating vendored code.

## Testing Guidelines

Tests use pytest and pytest-xdist, and are named `test_*.py`, with test functions named `test_<behavior>`. Place new tests beside the corresponding implementation area under `tests/src/core_pdf`. Add or update fixtures and expected extraction output when behavior changes. For broad changes, use `uv run pytest tests/ -n auto`; for focused changes, run the subset of tests that exercises the recently modified code.

## Commit & Pull Request Guidelines

use short Conventional Commit-style subjects such as `feat(ocr): ...`, `fix: ...`, `test(corpus): ...`, and `ci: ...`. Keep commits focused and explain the user-visible or correctness impact. Pull requests should describe the change, motivation, validation commands, fixture or compatibility impacts, and link related issues. Include representative output or screenshots when changing CLI behavior or documentation.
