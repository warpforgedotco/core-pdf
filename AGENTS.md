# Build, Test, and Development Commands

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
