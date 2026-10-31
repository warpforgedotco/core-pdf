# Build and Development Commands

Use `uv` for environments and locked dependencies:

```sh
uv sync --all-packages --all-groups --extra unstructured                 # install development dependencies
uv run --all-packages --group lint ruff check .     # lint Python files
uv run --all-packages --group lint ruff format --check .
uv run --all-packages --group lint mypy             # static type checking
uv run --all-packages --group lint lint-imports     # architecture: layer and dependency contracts
uv run --all-packages --group lint --group test ty check
prek run --all-files                 # run repository hooks across all files
```

## Dependency Management

Never edit `pyproject.toml` or `uv.lock` manually when adding or removing dependencies. Use `uv add --group <group> <package>` or `uv remove --group <group> <package>`; these commands update project metadata and the lockfile, and both generated changes should be reviewed and committed. Use the existing groups for their intended purpose: `lint` for Ruff, mypy, and ty, and `compiling` for Nuitka. For example, add a lint dependency with `uv add --group lint ruff`; do not create a new group when an existing group fits. A dependency of a workspace member is added with `uv add --package <member> <package>`; for another member, pass a version range such as `"core-jbig2>=0.1.0,<0.2.0"` and uv records the `{ workspace = true }` source itself.

## Commit & Pull Request Guidelines

Use short Conventional Commit-style subjects such as `feat(ocr): ...`, `fix: ...`, and `ci: ...`. Keep commits focused and explain the user-visible or correctness impact. Pull requests should describe the change, motivation, validation commands, compatibility impacts, and link related issues. Include representative output or screenshots when changing CLI behavior or documentation.

## Local and CI Validation

To run a reference library's own test suite against its `core_pdf_compat` facade, initialize the submodules under `tests/upstream` with `git submodule update --init --recursive --depth 1`, then run `uv run --group test python scripts/run_upstream_suite.py <suite>`. Add `--reference` to run the same tests on the real library, which separates environment failures from facade gaps. Reports go to `.upstream-results/`.

To prove a floor package or spec works alone, install it on its own (`uv sync --locked --package <member>`) and run `scripts/check_package_isolation.py <member>`, which checks that the tiers above it are absent. This replaces `.venv`, so run a full `uv sync --all-packages` afterwards.

To validate against the locked dependencies, use `uv run --locked` with the relevant group, such as `uv run --locked --group lint mypy`. Do not use `--locked` while intentionally changing dependencies; update them with `uv add` or `uv remove` first.
