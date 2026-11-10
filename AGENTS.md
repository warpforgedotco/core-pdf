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

To run a reference library's own test suite against its `core_pdf_compat` facade, initialize the submodules under `tests/upstream` with `git submodule update --init --recursive --depth 1`, then run `uv run --group test python scripts/run_upstream_suite.py <suite>`. Add `--reference` to run the same tests on the real library, which separates environment failures from facade gaps. Reports go to `.upstream-results/`. For pypdf, `--network` first downloads the files its tests fetch, listed in `tests/compat/pypdf_test_files.tsv`, and also runs the tests marked `enable_socket`; add `-m enable_socket` to run only those.

To prove a floor package or spec works alone, install it on its own (`uv sync --locked --package <member>`) and run `scripts/check_package_isolation.py <member>`, which checks that the tiers above it are absent. This replaces `.venv`, so run a full `uv sync --all-packages` afterwards.

To validate against the locked dependencies, use `uv run --locked` with the relevant group, such as `uv run --locked --group lint mypy`. Do not use `--locked` while intentionally changing dependencies; update them with `uv add` or `uv remove` first.

## Benchmarks

`tests/benchmarks` holds CodSpeed benchmarks of core's saves and each facade's main calls, on upstream test PDFs (it needs the pdfplumber, pypdf, unstructured and x-ray submodules). Run them locally with `uv run --all-packages --group benchmark --extra unstructured pytest tests/benchmarks --codspeed`. The `CodSpeed` workflow runs them under CPU simulation on every pull request and on main, and reports each change against main. To compare a facade's speed with the real library's, use `scripts/benchmark_facades.py`.

## Compiled modules

`core-pdf-cythonized` compiles its Python files with Cython, and a build can leave `.so` files beside them in `packages/core-pdf-cythonized/src/`. Those shadow later edits to the `.py` files. After changing that package, rebuild it with `uv sync --all-packages --reinstall-package core-pdf-cythonized` before testing.

## Lazy imports

CI and development run on Python 3.15; the packages still support 3.14. Modules list the imports a text extraction does not need in `__lazy_modules__`, which 3.15 loads on first use and 3.14 ignores, so do not use the `lazy` keyword (a syntax error on 3.14). The list names the module in the `from` clause: `from importlib import resources` needs `"importlib"`. An eager `from module import name` of a name that module imported lazily loads it, so import such names from where they are defined. `scripts/check_lazy_imports.py` fails if a text extraction loads a module that should stay deferred.

## Upstream versions

The facades support the latest release of each library they follow (pypdf, pdfminer.six, pdfplumber, unstructured, llama-index-readers-file, x-ray). The upstream suites in `scripts/run_upstream_suite.py` pin those latest versions, and the comparison scripts (`scripts/check_*_output.py`, `check_pypdf_bytes.py`, `check_pypdf_reads.py`, `check_encrypt_readers.py`) compare against the same pins. The `Upstream releases` workflow checks the pins against PyPI each week and keeps one open issue while any is behind; `uv run --no-project python scripts/check_upstream_versions.py` does the same check locally.

To move to a new release: move the library's submodule under `tests/upstream` to that release's tag, change its pins in `run_upstream_suite.py`, the comparison scripts and `.github/workflows/tests.yml`, update its tag and commit in `RELEASES` next to the pins in `run_upstream_suite.py`, run the suite with `--reference` first, then the facade and the comparison scripts, and fix the facade until both match. For pypdf, also run `uv run --group test python scripts/fetch_pypdf_test_files.py --record --write`, which reruns pypdf's suite to list the files its tests download in `tests/compat/pypdf_test_files.tsv`; CI fetches that list and fails on any file it cannot get. Read the library's diff for the code the facade mirrors; a test run alone can miss behaviour changes.

The suites run the tests in the checkout, so each checkout must sit at the commit of the pinned release's tag. CI's quality job runs `uv run --no-project python scripts/check_upstream_versions.py --checkouts`, which needs no network: it compares the commits `git ls-tree HEAD tests/upstream/` records with `RELEASES` and fails on any checkout elsewhere. llama_index tags only llama-index-core, so its entry is the tag of the pinned core release, v0.14.25, whose `llama-index-readers-file` directory is that package's 0.7.0 release; when moving either pin, pick the core tag whose readers-file directory matches the pinned readers-file sdist.

## Submodules

The upstream checkouts under `tests/upstream` are submodules, and `git pull` does not move them. After pulling, run `git submodule update --init --recursive --depth 1`, or test against stale tests. Don't commit with `git commit -a` while a submodule is out of date: it records the stale commit and moves the checkout back for everyone. Stage files by name, and stage a submodule only when moving it on purpose.
